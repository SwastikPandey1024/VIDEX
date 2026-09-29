"""Phase 4.0 — Audio Intelligence & Timestamped ASR Acceptance Test.

This is the hermetic acceptance test for the audio pipeline. It validates
the full chain using MockASRProvider only:

    test_video
        -> VideoReader (DecodedFrame + FrameTimestamp provenance)
        -> AudioPipeline.process_video()
            -> extract_audio_stream() [PyAV]
            -> MockASRProvider.transcribe()
            -> TemporalTranscriptFusion.fuse()
            -> AudioPipeline._generate_evidence()
        -> AudioPipelineResult (audio_metadata, segments, evidence)

The test does NOT require faster-whisper weights to run in CI.
Real-model acceptance is handled by test_faster_whisper_acceptance (skippable).
"""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import av
import cv2
import numpy as np
import pytest

from videx.audio.base import MockASRConfig
from videx.audio.extraction import (
    NoAudioStreamError,
    extract_audio_metadata,
    extract_audio_stream,
)
from videx.audio.fusion import TemporalTranscriptFusion, TemporalTranscriptFusionConfig
from videx.audio.mock import MockASRProvider
from videx.audio.normalization import normalize_transcript
from videx.audio.pipeline import AudioPipeline, AudioPipelineResult
from videx.domain.schemas import EvidenceType, TranscriptSegment


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_silent_mp4(path: Path, fps: float = 25.0, duration: float = 2.0) -> None:
    n_frames = int(fps * duration)
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    try:
        for _ in range(n_frames):
            out.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        out.release()


def _make_video_with_audio(
    path: Path, fps: float = 25.0, duration: float = 3.0
) -> None:
    fps_frac = Fraction(fps).limit_denominator(1000)
    sample_rate = 16000
    n_frames = int(fps * duration)
    n_audio_samples = int(sample_rate * duration)

    container = av.open(str(path), mode="w")

    v_stream = container.add_stream("mpeg4", rate=fps_frac)
    v_stream.width = 320
    v_stream.height = 240
    v_stream.pix_fmt = "yuv420p"

    try:
        a_stream = container.add_stream("aac", rate=sample_rate)
    except Exception:  # noqa: BLE001
        a_stream = container.add_stream("mp2", rate=sample_rate)

    for i in range(n_frames):
        img = np.full((240, 320, 3), 50, dtype=np.uint8)
        v_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        v_frame = v_frame.reformat(format="yuv420p")
        v_frame.pts = i
        for packet in v_stream.encode(v_frame):
            container.mux(packet)

    t_arr = np.linspace(0, duration, n_audio_samples, endpoint=False, dtype=np.float32)
    sine_wave = (0.3 * np.sin(2 * np.pi * 440.0 * t_arr)).astype(np.float32)
    a_frame = av.AudioFrame(format="fltp", layout="mono", samples=n_audio_samples)
    a_frame.sample_rate = sample_rate
    a_frame.planes[0].update(sine_wave.tobytes())
    a_frame.pts = 0
    for packet in a_stream.encode(a_frame):
        container.mux(packet)

    for packet in v_stream.encode():
        container.mux(packet)
    for packet in a_stream.encode():
        container.mux(packet)

    container.close()


@pytest.fixture(scope="module")
def video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "test_audio.mp4"
    _make_video_with_audio(path)
    return path


@pytest.fixture(scope="module")
def video_silent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "test_silent.mp4"
    _make_silent_mp4(path)
    return path


@pytest.fixture(scope="module")
def mock_pipeline() -> AudioPipeline:
    return AudioPipeline.create_mock(
        config=MockASRConfig(
            canned_segments=[
                {
                    "raw_text": "Speed camera ahead",
                    "start_timestamp_seconds": 0.0,
                    "end_timestamp_seconds": 2.0,
                    "confidence": 0.94,
                    "language": "en",
                },
                {
                    "raw_text": "Signal ahead slow down",
                    "start_timestamp_seconds": 2.1,
                    "end_timestamp_seconds": 4.0,
                    "confidence": 0.91,
                    "language": "en",
                },
            ]
        )
    )


# ── Audio Extraction ──────────────────────────────────────────────────────────


class TestAudioExtraction:
    """Validates audio stream extraction from video containers."""

    def test_extract_audio_metadata(self, video_with_audio: Path) -> None:
        meta = extract_audio_metadata(video_with_audio)
        assert meta is not None
        assert meta.sample_rate == 16000
        assert meta.channels in (1, 2)
        assert meta.duration_seconds > 0.0
        assert meta.codec_name in ("aac", "mp2", "mp3")

    def test_extract_audio_metadata_silent_raises(self, video_silent: Path) -> None:
        with pytest.raises(NoAudioStreamError):
            extract_audio_metadata(video_silent)

    def test_extract_audio_stream_dtype(self, video_with_audio: Path) -> None:
        arr, meta = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    def test_extract_audio_stream_length(self, video_with_audio: Path) -> None:
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        assert len(arr) > 0, "Audio stream must contain samples"

    def test_extract_audio_stream_amplitude(self, video_with_audio: Path) -> None:
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        max_abs = float(np.max(np.abs(arr)))
        assert max_abs <= 1.0, f"Audio values must be in [-1.0, 1.0], got max_abs={max_abs}"

    def test_extract_audio_stream_silent_raises(self, video_silent: Path) -> None:
        with pytest.raises(NoAudioStreamError):
            extract_audio_stream(video_silent)


# ── Normalization ─────────────────────────────────────────────────────────────


class TestTranscriptNormalization:
    """Validates conservative Unicode NFKC normalization of transcript text."""

    def test_collapses_multiple_spaces(self) -> None:
        assert normalize_transcript("Hello   World") == "Hello World"

    def test_preserves_smart_quotes(self) -> None:
        assert normalize_transcript("India\u2019s best") == "India\u2019s best"

    def test_preserves_accented_chars(self) -> None:
        assert normalize_transcript("caf\u00e9") == "caf\u00e9"

    def test_preserves_devanagari(self) -> None:
        raw = "\u0928\u092e\u0938\u094d\u0924\u0947  \u092f\u0939  \u0939\u0948"
        result = normalize_transcript(raw)
        assert "\u0928\u092e\u0938\u094d\u0924\u0947" in result

    def test_empty_string(self) -> None:
        assert normalize_transcript("") == ""

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert normalize_transcript("  hello  ") == "hello"


# ── MockASRProvider ───────────────────────────────────────────────────────────


class TestMockASRProvider:
    """Validates deterministic MockASRProvider transcript generation."""

    def test_returns_list_of_transcript_segments(
        self, video_with_audio: Path
    ) -> None:
        provider = MockASRProvider()
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        segments = provider.transcribe(audio_input=arr)
        assert isinstance(segments, list)
        assert len(segments) > 0

    def test_canned_segments_returned_verbatim(
        self, video_with_audio: Path
    ) -> None:
        provider = MockASRProvider(
            MockASRConfig(
                canned_segments=[
                    {
                        "raw_text": "VIDEX test",
                        "start_timestamp_seconds": 0.0,
                        "end_timestamp_seconds": 2.0,
                        "confidence": 0.99,
                        "language": "en",
                    }
                ]
            )
        )
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        segments = provider.transcribe(audio_input=arr)
        assert len(segments) == 1
        assert segments[0].raw_text == "VIDEX test"
        assert segments[0].confidence == pytest.approx(0.99)
        assert segments[0].language == "en"

    def test_segment_temporal_fields(self, video_with_audio: Path) -> None:
        provider = MockASRProvider()
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        segments = provider.transcribe(audio_input=arr)
        for seg in segments:
            assert isinstance(seg, TranscriptSegment)
            assert seg.start_timestamp_seconds >= 0.0
            assert seg.end_timestamp_seconds > seg.start_timestamp_seconds
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.raw_text
            assert seg.normalized_text
            assert seg.provider
            assert seg.language

    def test_segment_video_id_propagated(self, video_with_audio: Path) -> None:
        from uuid import uuid4

        provider = MockASRProvider()
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        vid_id = uuid4()
        segments = provider.transcribe(audio_input=arr, video_id=vid_id)
        for seg in segments:
            assert seg.video_id == vid_id


# ── TemporalTranscriptFusion ──────────────────────────────────────────────────


class TestTemporalTranscriptFusion:
    """Validates temporal deduplication of ASR transcript segments."""

    def _make_seg(
        self,
        text: str,
        start: float,
        end: float,
        lang: str = "en",
        provider: str = "mock",
        video_id: UUID | None = None,
    ) -> TranscriptSegment:
        from uuid import uuid4

        return TranscriptSegment(
            video_id=video_id or uuid4(),
            start_timestamp_seconds=start,
            end_timestamp_seconds=end,
            raw_text=text,
            normalized_text=text,
            language=lang,
            confidence=0.9,
            provider=provider,
        )

    def test_empty_input(self) -> None:
        fusion = TemporalTranscriptFusion()
        assert fusion.fuse([]) == []

    def test_single_segment_passthrough(self) -> None:
        from uuid import uuid4

        fusion = TemporalTranscriptFusion()
        seg = self._make_seg("hello", 0.0, 2.0, video_id=uuid4())
        result = fusion.fuse([seg])
        assert len(result) == 1

    def test_identical_adjacent_segments_merged(self) -> None:
        from uuid import uuid4

        vid = uuid4()
        fusion = TemporalTranscriptFusion(
            TemporalTranscriptFusionConfig(max_gap_seconds=0.5)
        )
        segs = [
            self._make_seg("VIDEX test", 0.0, 2.0, video_id=vid),
            self._make_seg("VIDEX test", 2.1, 4.0, video_id=vid),
            self._make_seg("other text", 5.0, 7.0, video_id=vid),
        ]
        result = fusion.fuse(segs)
        assert len(result) == 2

    def test_different_segments_not_merged(self) -> None:
        from uuid import uuid4

        vid = uuid4()
        fusion = TemporalTranscriptFusion()
        segs = [
            self._make_seg("segment one", 0.0, 2.0, video_id=vid),
            self._make_seg("segment two", 2.1, 4.0, video_id=vid),
        ]
        result = fusion.fuse(segs)
        assert len(result) == 2

    def test_supporting_segment_ids_in_attributes(self) -> None:
        from uuid import uuid4

        vid = uuid4()
        fusion = TemporalTranscriptFusion(
            TemporalTranscriptFusionConfig(max_gap_seconds=0.5)
        )
        segs = [
            self._make_seg("test text", 0.0, 2.0, video_id=vid),
            self._make_seg("test text", 2.1, 4.0, video_id=vid),
        ]
        result = fusion.fuse(segs)
        assert "supporting_segment_ids" in result[0].attributes


# ── AudioPipeline ─────────────────────────────────────────────────────────────


class TestAudioPipeline:
    """Validates full AudioPipeline orchestration and Evidence generation."""

    def test_process_video_returns_result(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio, language="en")
        assert isinstance(result, AudioPipelineResult)

    def test_process_video_video_id_set(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        assert result.video_id is not None
        assert isinstance(result.video_id, UUID)

    def test_process_video_audio_metadata_present(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        assert result.audio_metadata is not None
        assert result.audio_metadata.sample_rate == 16000

    def test_process_video_segments_non_empty(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        assert len(result.raw_segments) > 0
        assert len(result.fused_segments) > 0

    def test_process_video_segment_fields(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        for seg in result.fused_segments:
            assert seg.video_id == result.video_id
            assert seg.raw_text
            assert seg.normalized_text
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.start_timestamp_seconds < seg.end_timestamp_seconds
            assert seg.provider

    def test_process_video_evidence_non_empty(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        assert len(result.evidence) > 0

    def test_process_video_evidence_type(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        for ev in result.evidence:
            assert ev.evidence_type == EvidenceType.AUDIO_TRANSCRIPT

    def test_process_video_evidence_video_id(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        for ev in result.evidence:
            assert ev.video_id == result.video_id

    def test_process_video_evidence_temporal_fields(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        for ev in result.evidence:
            assert ev.timestamp_seconds >= 0.0
            assert "end_timestamp_seconds" in (ev.metadata or {})
            end_ts = ev.metadata["end_timestamp_seconds"]
            assert end_ts > ev.timestamp_seconds

    def test_process_video_silent_returns_empty(
        self, mock_pipeline: AudioPipeline, video_silent: Path
    ) -> None:
        result = mock_pipeline.process_video(video_silent, language="en")
        assert result.raw_segments == []
        assert result.fused_segments == []
        assert result.evidence == []
        assert result.audio_metadata is None

    def test_process_video_video_id_override(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        from uuid import uuid4

        custom_id = uuid4()
        result = mock_pipeline.process_video(video_with_audio, video_id=custom_id)
        assert result.video_id == custom_id
        for ev in result.evidence:
            assert ev.video_id == custom_id

    def test_processing_time_positive(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        result = mock_pipeline.process_video(video_with_audio)
        assert result.processing_time_seconds > 0.0

    def test_process_audio_direct(
        self, mock_pipeline: AudioPipeline, video_with_audio: Path
    ) -> None:
        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        result = mock_pipeline.process_audio(arr, language="en")
        assert isinstance(result, AudioPipelineResult)
        assert len(result.fused_segments) > 0
        assert len(result.evidence) > 0


# ── CI-safe Real-model acceptance ─────────────────────────────────────────────


def _faster_whisper_available() -> bool:
    """Return True only if faster-whisper is installed AND the tiny model is locally cached."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False

    # Also check the model is locally cached to avoid HF Hub network calls in CI
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-untyped]

        cached = try_to_load_from_cache(
            "Systran/faster-whisper-tiny", "model.bin"
        )
        return cached is not None
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not _faster_whisper_available(),
    reason="faster-whisper tiny model not locally cached — real model acceptance skipped in CI",
)
class TestFasterWhisperAcceptance:
    """Real faster-whisper inference acceptance — skipped when weights absent."""

    def test_real_transcription(self, video_with_audio: Path) -> None:
        from videx.audio.base import FasterWhisperConfig
        from videx.audio.whisper import FasterWhisperASRProvider

        cfg = FasterWhisperConfig(
            model_size_or_path="tiny", device="cpu", compute_type="int8"
        )
        provider = FasterWhisperASRProvider(cfg)
        provider.warmup()

        arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        segments = provider.transcribe(audio_input=arr, language="en")

        # Sine wave has no speech; model may return empty — that's acceptable
        for seg in segments:
            assert seg.raw_text
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.start_timestamp_seconds < seg.end_timestamp_seconds

