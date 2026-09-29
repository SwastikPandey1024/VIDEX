"""Unit tests for VIDEX Audio Intelligence & Timestamped ASR subsystem (Phase 4.0)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import av
import cv2
import numpy as np
import pytest

from videx.audio.base import (
    ASRProvider,
    AudioProvider,
    FasterWhisperConfig,
    MockASRConfig,
)
from videx.audio.extraction import (
    CorruptAudioError,
    NoAudioStreamError,
    extract_audio_bytes_wav,
    extract_audio_metadata,
    extract_audio_stream,
    extract_audio_to_wav,
)
from videx.audio.fusion import TemporalTranscriptFusion, TemporalTranscriptFusionConfig
from videx.audio.mock import MockASRProvider
from videx.audio.normalization import normalize_transcript
from videx.audio.pipeline import AudioPipeline, AudioPipelineResult
from videx.audio.whisper import FasterWhisperASRProvider
from videx.domain.schemas import (
    AudioMetadata,
    Evidence,
    EvidenceType,
    TranscriptSegment,
)


@pytest.fixture
def synthetic_video_with_audio(tmp_path: Path) -> Path:
    """Create a 2.0-second synthetic MP4 video with a 16kHz sine wave AAC audio track."""
    video_path = tmp_path / "synthetic_audio_video.mp4"
    container = av.open(str(video_path), "w")

    # Add video stream
    v_stream = container.add_stream("mpeg4", rate=25)
    v_stream.width = 160
    v_stream.height = 120
    v_stream.pix_fmt = "yuv420p"

    # Add audio stream
    a_stream = container.add_stream("aac", rate=16000)

    # Write 50 video frames (2 seconds at 25 fps)
    for _ in range(50):
        frame = av.VideoFrame(160, 120, "yuv420p")
        for packet in v_stream.encode(frame):
            container.mux(packet)

    # Write 2.0 seconds of 440Hz sine wave audio at 16kHz
    num_samples = int(16000 * 2.0)
    t = np.linspace(0, 2.0, num_samples, endpoint=False, dtype=np.float32)
    sine_wave = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Packetize into 1024-sample audio frames
    chunk_size = 1024
    for offset in range(0, num_samples, chunk_size):
        chunk = sine_wave[offset : offset + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        a_frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
        a_frame.sample_rate = 16000
        for packet in a_stream.encode(a_frame):
            container.mux(packet)

    # Flush encoders
    for packet in v_stream.encode(None):
        container.mux(packet)
    for packet in a_stream.encode(None):
        container.mux(packet)

    container.close()
    return video_path


@pytest.fixture
def synthetic_video_no_audio(tmp_path: Path) -> Path:
    """Create a synthetic MP4 video without an audio stream."""
    video_path = tmp_path / "video_no_audio.mp4"
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (160, 120))
    for _ in range(25):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


class TestAudioMetadataAndExtraction:
    """Tests for audio metadata extraction and format normalization."""

    def test_extract_audio_metadata_success(self, synthetic_video_with_audio: Path) -> None:
        meta = extract_audio_metadata(synthetic_video_with_audio)
        assert isinstance(meta, AudioMetadata)
        assert meta.sample_rate == 16000
        assert meta.channels in (1, 2)
        assert meta.duration_seconds > 0.0
        assert meta.codec_name is not None

    def test_extract_audio_metadata_no_audio(self, synthetic_video_no_audio: Path) -> None:
        with pytest.raises(NoAudioStreamError):
            extract_audio_metadata(synthetic_video_no_audio)

    def test_extract_audio_metadata_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_audio_metadata(tmp_path / "non_existent.mp4")

    def test_extract_audio_metadata_corrupt_file(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"NOT_A_VALID_VIDEO_OR_AUDIO")
        with pytest.raises(CorruptAudioError):
            extract_audio_metadata(corrupt)

    def test_extract_audio_stream_array(self, synthetic_video_with_audio: Path) -> None:
        arr, meta = extract_audio_stream(synthetic_video_with_audio, target_sample_rate=16000)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.ndim == 1
        assert len(arr) > 0
        # Duration should be approximately 2.0s
        assert 1.8 <= len(arr) / 16000.0 <= 2.2
        assert meta.sample_rate == 16000

    def test_extract_audio_bytes_wav(self, synthetic_video_with_audio: Path) -> None:
        wav_bytes, meta = extract_audio_bytes_wav(synthetic_video_with_audio)
        assert isinstance(wav_bytes, bytes)
        assert wav_bytes.startswith(b"RIFF")
        assert b"WAVE" in wav_bytes[:16]

    def test_extract_audio_to_wav_file(
        self, synthetic_video_with_audio: Path, tmp_path: Path
    ) -> None:
        out_wav = tmp_path / "extracted.wav"
        res_path = extract_audio_to_wav(synthetic_video_with_audio, out_wav)
        assert res_path.exists()
        assert res_path.stat().st_size > 0


class TestTranscriptNormalization:
    """Tests for conservative Unicode transcript normalization."""

    def test_whitespace_and_nfkc(self) -> None:
        raw = "  Hello   world\n\tthis  is a   test  "
        norm = normalize_transcript(raw)
        assert norm == "Hello world this is a test"

    def test_strips_bom_and_zero_width_space(self) -> None:
        raw = "\ufeffHello\u200bworld\u2060!"
        norm = normalize_transcript(raw)
        assert norm == "Helloworld!"

    def test_preserves_indic_devanagari_zwj_and_zwnj(self) -> None:
        # Hindi: चिट्ठी or राष्ट्र with explicit ZWJ/ZWNJ
        raw_hindi = "नमस्ते \u200d भारत \u200c"
        norm = normalize_transcript(raw_hindi, preserve_indic_zw=True)
        assert "\u200d" in norm
        assert "\u200c" in norm
        assert "नमस्ते" in norm
        assert "भारत" in norm

    def test_case_folding_option(self) -> None:
        raw = "VIDEX Speech Intelligence"
        assert normalize_transcript(raw, lowercase=False) == "VIDEX Speech Intelligence"
        assert normalize_transcript(raw, lowercase=True) == "videx speech intelligence"

    def test_empty_string(self) -> None:
        assert normalize_transcript("") == ""


class TestASRProviderProtocols:
    """Tests for protocol compliance of ASR providers."""

    def test_mock_asr_provider_protocol(self) -> None:
        provider = MockASRProvider()
        assert isinstance(provider, ASRProvider)
        assert isinstance(provider, AudioProvider)
        assert provider.provider_name == "mock_asr"
        assert "en" in provider.supported_languages
        assert "hi" in provider.supported_languages
        provider.warmup()

    def test_faster_whisper_provider_protocol(self) -> None:
        provider = FasterWhisperASRProvider()
        assert isinstance(provider, ASRProvider)
        assert isinstance(provider, AudioProvider)
        assert provider.provider_name == "faster_whisper"
        assert "en" in provider.supported_languages
        assert "hi" in provider.supported_languages


class TestMockASRProvider:
    """Tests for deterministic MockASRProvider execution."""

    def test_mock_transcribe_synthetic_array(self) -> None:
        provider = MockASRProvider(MockASRConfig(segment_duration_seconds=2.0))
        audio_arr = np.zeros(32000, dtype=np.float32)  # 2.0s
        v_id = uuid4()
        segments = provider.transcribe(audio_arr, video_id=v_id, language="en")
        assert len(segments) >= 1
        seg = segments[0]
        assert isinstance(seg, TranscriptSegment)
        assert seg.video_id == v_id
        assert seg.language == "en"
        assert seg.start_timestamp_seconds == 0.0
        assert seg.end_timestamp_seconds == 2.0
        assert len(seg.words) > 0
        assert seg.confidence == 0.95
        assert seg.raw_text != ""
        assert seg.normalized_text != ""

    def test_mock_transcribe_hindi_language(self) -> None:
        provider = MockASRProvider()
        audio_arr = np.zeros(16000, dtype=np.float32)
        segments = provider.transcribe(audio_arr, language="hi")
        assert len(segments) >= 1
        seg = segments[0]
        assert seg.language == "hi"
        assert any("\u0900" <= c <= "\u097f" for c in seg.raw_text)

    def test_mock_transcribe_canned_segments(self) -> None:
        canned = [
            {
                "start_timestamp_seconds": 0.0,
                "end_timestamp_seconds": 1.5,
                "raw_text": "Emergency vehicle approaching",
                "language": "en",
                "confidence": 0.98,
            },
            {
                "start_timestamp_seconds": 2.0,
                "end_timestamp_seconds": 3.5,
                "raw_text": "Please clear the intersection",
                "language": "en",
                "confidence": 0.96,
            },
        ]
        provider = MockASRProvider(MockASRConfig(canned_segments=canned))
        segments = provider.transcribe(b"dummy")
        assert len(segments) == 2
        assert segments[0].raw_text == "Emergency vehicle approaching"
        assert segments[0].confidence == 0.98
        assert segments[1].raw_text == "Please clear the intersection"
        assert segments[1].confidence == 0.96


class TestTemporalTranscriptFusion:
    """Tests for temporal fusion and deduplication of transcript segments."""

    def test_fuse_identical_overlapping_segments(self) -> None:
        v_id = uuid4()
        fusion = TemporalTranscriptFusion(TemporalTranscriptFusionConfig(max_gap_seconds=0.5))

        # Two consecutive segments with identical text (sliding window repeat)
        seg1 = TranscriptSegment(
            video_id=v_id,
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=2.0,
            raw_text="MH 12 AB 1234",
            language="en",
            confidence=0.92,
            provider="test_asr",
        )
        seg2 = TranscriptSegment(
            video_id=v_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=3.0,
            raw_text="MH 12 AB 1234",
            language="en",
            confidence=0.96,
            provider="test_asr",
        )

        fused = fusion.fuse([seg1, seg2])
        assert len(fused) == 1
        assert fused[0].raw_text == "MH 12 AB 1234"
        assert fused[0].start_timestamp_seconds == 0.0
        assert fused[0].end_timestamp_seconds == 3.0
        # Preserves supporting segment IDs
        assert "supporting_segment_ids" in fused[0].attributes
        assert len(fused[0].attributes["supporting_segment_ids"]) == 2

    def test_fuse_distinct_non_overlapping_segments(self) -> None:
        v_id = uuid4()
        fusion = TemporalTranscriptFusion(TemporalTranscriptFusionConfig(max_gap_seconds=0.5))

        seg1 = TranscriptSegment(
            video_id=v_id,
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=2.0,
            raw_text="Hello world",
            language="en",
            confidence=0.9,
            provider="test_asr",
        )
        seg2 = TranscriptSegment(
            video_id=v_id,
            start_timestamp_seconds=5.0,
            end_timestamp_seconds=7.0,
            raw_text="Goodbye world",
            language="en",
            confidence=0.9,
            provider="test_asr",
        )

        fused = fusion.fuse([seg1, seg2])
        assert len(fused) == 2
        assert fused[0].raw_text == "Hello world"
        assert fused[1].raw_text == "Goodbye world"

    def test_fuse_empty_input(self) -> None:
        fusion = TemporalTranscriptFusion()
        assert fusion.fuse([]) == []


class TestAudioPipeline:
    """Tests for the full AudioPipeline orchestration and Evidence creation."""

    def test_process_video_with_audio(self, synthetic_video_with_audio: Path) -> None:
        pipeline = AudioPipeline.create_mock()
        res = pipeline.process_video(synthetic_video_with_audio)

        assert isinstance(res, AudioPipelineResult)
        assert res.audio_metadata is not None
        assert res.audio_metadata.sample_rate == 16000
        assert len(res.raw_segments) > 0
        assert len(res.fused_segments) > 0
        assert len(res.evidence) > 0

        # Check evidence records
        for ev in res.evidence:
            assert isinstance(ev, Evidence)
            assert ev.evidence_type == EvidenceType.AUDIO_TRANSCRIPT
            assert ev.video_id == res.video_id
            assert ev.timestamp_seconds is not None
            assert ev.confidence >= 0.0
            assert "raw_text" in ev.raw_payload
            assert "normalized_text" in ev.raw_payload
            assert "audio" in ev.tags
            assert "asr" in ev.tags

    def test_process_video_without_audio_stream(self, synthetic_video_no_audio: Path) -> None:
        pipeline = AudioPipeline.create_mock()
        res = pipeline.process_video(synthetic_video_no_audio)

        assert isinstance(res, AudioPipelineResult)
        assert res.audio_metadata is None
        assert len(res.raw_segments) == 0
        assert len(res.fused_segments) == 0
        assert len(res.evidence) == 0

    def test_process_pre_extracted_audio(self) -> None:
        pipeline = AudioPipeline.create_mock()
        audio_arr = np.zeros(32000, dtype=np.float32)  # 2.0s
        v_id = uuid4()
        res = pipeline.process_audio(audio_arr, video_id=v_id, language="hi")

        assert res.video_id == v_id
        assert res.language == "hi"
        assert len(res.fused_segments) > 0
        assert len(res.evidence) > 0
        assert res.evidence[0].video_id == v_id


class TestFasterWhisperProviderMocked:
    """Test FasterWhisperASRProvider with mocked internal model."""

    def test_transcribe_mocked_faster_whisper(self) -> None:
        provider = FasterWhisperASRProvider(FasterWhisperConfig(device="cpu", compute_type="int8"))

        mock_seg = MagicMock()
        mock_seg.start = 0.5
        mock_seg.end = 2.5
        mock_seg.text = " Testing faster whisper integration "
        mock_seg.avg_logprob = -0.1
        mock_seg.no_speech_prob = 0.02

        mock_word = MagicMock()
        mock_word.word = "Testing"
        mock_word.start = 0.5
        mock_word.end = 1.0
        mock_word.probability = 0.95
        mock_seg.words = [mock_word]

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = ([mock_seg], mock_info)

        provider._model = mock_whisper_model

        audio_arr = np.zeros(16000, dtype=np.float32)
        v_id = uuid4()
        results = provider.transcribe(audio_arr, video_id=v_id, language="en")

        assert len(results) == 1
        seg = results[0]
        assert seg.video_id == v_id
        assert seg.raw_text == "Testing faster whisper integration"
        assert seg.normalized_text == "Testing faster whisper integration"
        assert seg.start_timestamp_seconds == 0.5
        assert seg.end_timestamp_seconds == 2.5
        assert len(seg.words) == 1
        assert seg.words[0].word == "Testing"
        assert seg.words[0].confidence == 0.95
        assert seg.confidence is not None and seg.confidence > 0.8
