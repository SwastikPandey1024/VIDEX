"""Phase 4.0 Audio Intelligence Pipeline Smoke Test Script.

Validates the complete ASR/Audio foundation:
1. AudioExtraction: Extracts audio from synthetic video via PyAV without corruption.
2. AudioMetadata: Captures sample_rate, channels, duration, codec provenance.
3. MockASRProvider: Produces deterministic TranscriptSegments with temporal provenance.
4. FasterWhisperASRProvider: Skips gracefully when model not present (CI-safe).
5. TemporalTranscriptFusion: Deduplicates consecutive repeated segments.
6. AudioPipeline: Orchestrates extraction -> ASR -> fusion -> Evidence generation.
7. Evidence: Verifies correct EvidenceType, temporal fields, provider tags.
8. Telemetry: Measures wall-clock latency for extraction and transcription.
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

# Force UTF-8 stdout to handle non-ASCII characters (Hindi, smart quotes) on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import av
import cv2
import numpy as np

from videx.audio.base import FasterWhisperConfig, MockASRConfig
from videx.audio.extraction import (
    NoAudioStreamError,
    extract_audio_metadata,
    extract_audio_stream,
)
from videx.audio.fusion import TemporalTranscriptFusion, TemporalTranscriptFusionConfig
from videx.audio.mock import MockASRProvider
from videx.audio.normalization import normalize_transcript
from videx.audio.pipeline import AudioPipeline
from videx.domain.schemas import EvidenceType

# ── ANSI helpers ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
SKIP = f"{YELLOW}[SKIP]{RESET}"
INFO = f"{BLUE}[INFO]{RESET}"


def section(title: str) -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  -> {detail}"
    print(msg)
    return ok


def skip(label: str, reason: str = "") -> None:
    msg = f"  {SKIP}  {label}"
    if reason:
        msg += f"  -> {reason}"
    print(msg)


# ── Video / Audio fixture creation ────────────────────────────────────────────


def create_silent_video(path: Path, fps: float = 25.0, duration: float = 2.0) -> None:
    """Create a synthetic MP4 with no audio track (for NoAudioStreamError tests)."""
    n_frames = int(fps * duration)
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    try:
        for _ in range(n_frames):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()


def create_video_with_audio(path: Path, fps: float = 25.0, duration: float = 3.0) -> None:
    """Create a synthetic MP4 with both video and a sine-wave audio track."""
    from fractions import Fraction

    fps_frac = Fraction(fps).limit_denominator(1000)
    sample_rate = 16000
    n_frames = int(fps * duration)
    n_audio_samples = int(sample_rate * duration)

    container = av.open(str(path), mode="w")

    # Video stream
    v_stream = container.add_stream("mpeg4", rate=fps_frac)
    v_stream.width = 320
    v_stream.height = 240
    v_stream.pix_fmt = "yuv420p"

    # Audio stream — use libmp3lame fallback (widely available) or mp2 if aac fails
    try:
        a_stream = container.add_stream("aac", rate=sample_rate)
    except Exception:  # noqa: BLE001
        a_stream = container.add_stream("mp2", rate=sample_rate)

    # Write video frames using sequential PTS
    for i in range(n_frames):
        img = np.full((240, 320, 3), 50, dtype=np.uint8)
        v_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        v_frame = v_frame.reformat(format="yuv420p")
        v_frame.pts = i
        for packet in v_stream.encode(v_frame):
            container.mux(packet)

    # Write audio as one bulk frame (wav-style)
    t_arr = np.linspace(0, duration, n_audio_samples, endpoint=False, dtype=np.float32)
    sine_wave = (0.3 * np.sin(2 * np.pi * 440.0 * t_arr)).astype(np.float32)

    a_frame = av.AudioFrame(format="fltp", layout="mono", samples=n_audio_samples)
    a_frame.sample_rate = sample_rate
    a_frame.planes[0].update(sine_wave.tobytes())
    a_frame.pts = 0
    for packet in a_stream.encode(a_frame):
        container.mux(packet)

    # Flush encoders
    for packet in v_stream.encode():
        container.mux(packet)
    for packet in a_stream.encode():
        container.mux(packet)

    container.close()


# ── Section 1: Audio metadata extraction ─────────────────────────────────────


def test_audio_metadata(video_with_audio: Path, video_silent: Path) -> int:
    section("1. Audio Metadata Extraction")
    failures = 0

    t0 = time.perf_counter()
    meta = extract_audio_metadata(video_with_audio)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not check("AudioMetadata extracted", meta is not None):
        failures += 1

    ok = check("sample_rate == 16000", meta.sample_rate == 16000, f"{meta.sample_rate}")
    if not ok:
        failures += 1

    ok = check("channels in (1, 2)", meta.channels in (1, 2), f"{meta.channels}")
    if not ok:
        failures += 1

    ok = check("duration_seconds > 0", meta.duration_seconds > 0, f"{meta.duration_seconds:.2f}s")
    if not ok:
        failures += 1

    ok = check("codec_name set", bool(meta.codec_name), f"'{meta.codec_name}'")
    if not ok:
        failures += 1

    print(f"  {INFO}  Extraction latency: {elapsed_ms:.1f} ms")

    # Silent video should raise NoAudioStreamError
    try:
        extract_audio_metadata(video_silent)
        check("NoAudioStreamError raised for silent video", False)
        failures += 1
    except NoAudioStreamError:
        check("NoAudioStreamError raised for silent video", True)

    return failures


# ── Section 2: Audio stream extraction ────────────────────────────────────────


def test_audio_stream(video_with_audio: Path) -> int:
    section("2. Audio Stream Extraction (float32 PCM)")
    failures = 0

    t0 = time.perf_counter()
    audio_arr, meta = extract_audio_stream(video_with_audio, target_sample_rate=16000)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ok = check("Returns float32 numpy array", audio_arr.dtype == np.float32, str(audio_arr.dtype))
    if not ok:
        failures += 1

    ok = check("Array length > 0", len(audio_arr) > 0, f"{len(audio_arr)} samples")
    if not ok:
        failures += 1

    ok = check(
        "Values in [-1.0, 1.0]",
        float(np.max(np.abs(audio_arr))) <= 1.0,
        f"max_abs={float(np.max(np.abs(audio_arr))):.4f}",
    )
    if not ok:
        failures += 1

    ok = check("sample_rate == 16000", meta.sample_rate == 16000, f"{meta.sample_rate}")
    if not ok:
        failures += 1

    print(f"  {INFO}  Extraction: {elapsed_ms:.1f} ms | Samples: {len(audio_arr):,}")

    return failures


# ── Section 3: Transcript normalization ──────────────────────────────────────


def test_normalization() -> int:
    section("3. Transcript Normalization")
    failures = 0

    cases = [
        ("Hello   World!", "Hello World!"),
        ("INDIA\u2019S BEST", "INDIA\u2019S BEST"),  # Smart quote preserved
        ("caf\u00e9", "caf\u00e9"),  # Accented character preserved
        (
            "\u0928\u092e\u0938\u094d\u0924\u0947  \u092f\u0939  \u0939\u0948",
            "\u0928\u092e\u0938\u094d\u0924\u0947 \u092f\u0939 \u0939\u0948",
        ),
        ("", ""),
    ]

    for raw, expected in cases:
        result = normalize_transcript(raw)
        ok = check(
            f"normalize_transcript({ascii(raw[:30])})",
            result == expected,
            f"got {ascii(result[:30])}",
        )
        if not ok:
            failures += 1

    return failures


# ── Section 4: MockASRProvider ────────────────────────────────────────────────


def test_mock_provider(video_with_audio: Path) -> int:
    section("4. MockASRProvider — Deterministic Transcription")
    failures = 0

    provider = MockASRProvider(
        MockASRConfig(
            canned_segments=[
                {
                    "raw_text": "Speed camera ahead",
                    "start_timestamp_seconds": 0.0,
                    "end_timestamp_seconds": 2.5,
                    "confidence": 0.94,
                    "language": "en",
                },
                {
                    "raw_text": "\u0906\u0917\u0947 \u0915\u0948\u092e\u0930\u093e \u0939\u0948",
                    "start_timestamp_seconds": 2.6,
                    "end_timestamp_seconds": 4.1,
                    "confidence": 0.87,
                    "language": "hi",
                },
            ]
        )
    )

    audio_arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
    t0 = time.perf_counter()
    segments = provider.transcribe(audio_input=audio_arr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ok = check("Returned 2 segments", len(segments) == 2, f"{len(segments)}")
    if not ok:
        failures += 1

    for i, seg in enumerate(segments):
        ok = check(
            f"seg[{i}].confidence in [0,1]",
            seg.confidence is not None and 0.0 <= seg.confidence <= 1.0,
            f"{seg.confidence:.3f}" if seg.confidence is not None else "None",
        )
        if not ok:
            failures += 1
        ok = check(
            f"seg[{i}].raw_text non-empty",
            bool(seg.raw_text),
            f"'{seg.raw_text[:30]}'",
        )
        if not ok:
            failures += 1
        ok = check(
            f"seg[{i}].start < end",
            seg.start_timestamp_seconds < seg.end_timestamp_seconds,
            f"{seg.start_timestamp_seconds:.2f} < {seg.end_timestamp_seconds:.2f}",
        )
        if not ok:
            failures += 1
        ok = check(
            f"seg[{i}].normalized_text set",
            bool(seg.normalized_text),
        )
        if not ok:
            failures += 1
        ok = check(
            f"seg[{i}].video_id set",
            seg.video_id is not None,
        )
        if not ok:
            failures += 1

    print(f"  {INFO}  Transcription latency: {elapsed_ms:.1f} ms")

    return failures


# ── Section 5: TemporalTranscriptFusion ───────────────────────────────────────


def test_fusion() -> int:
    section("5. TemporalTranscriptFusion — Deduplication")
    failures = 0

    from uuid import uuid4

    from videx.domain.schemas import TranscriptSegment

    v_id = uuid4()
    fusion = TemporalTranscriptFusion(TemporalTranscriptFusionConfig(max_gap_seconds=0.5))

    # Two identical segments close together => should merge
    seg_a = TranscriptSegment(
        video_id=v_id,
        start_timestamp_seconds=0.0,
        end_timestamp_seconds=2.0,
        raw_text="Hello VIDEX",
        normalized_text="Hello VIDEX",
        language="en",
        confidence=0.9,
        provider="mock",
    )
    seg_b = TranscriptSegment(
        video_id=v_id,
        start_timestamp_seconds=2.1,
        end_timestamp_seconds=4.0,
        raw_text="Hello VIDEX",
        normalized_text="Hello VIDEX",
        language="en",
        confidence=0.9,
        provider="mock",
    )
    seg_c = TranscriptSegment(
        video_id=v_id,
        start_timestamp_seconds=5.0,
        end_timestamp_seconds=7.0,
        raw_text="Different text",
        normalized_text="Different text",
        language="en",
        confidence=0.8,
        provider="mock",
    )

    fused = fusion.fuse([seg_a, seg_b, seg_c])
    ok = check("Duplicate adjacent segments merged (3→2)", len(fused) == 2, f"{len(fused)}")
    if not ok:
        failures += 1

    ok = check("Empty input returns empty", fusion.fuse([]) == [])
    if not ok:
        failures += 1

    ok = check("Single segment passthrough", len(fusion.fuse([seg_c])) == 1)
    if not ok:
        failures += 1

    return failures


# ── Section 6: AudioPipeline end-to-end ──────────────────────────────────────


def test_pipeline(video_with_audio: Path, video_silent: Path) -> int:
    section("6. AudioPipeline — End-to-End with MockASR")
    failures = 0

    pipeline = AudioPipeline.create_mock(
        config=MockASRConfig(
            default_text="VIDEX video intelligence test",
            segment_duration_seconds=1.0,
        )
    )

    # Process video with audio
    t0 = time.perf_counter()
    result = pipeline.process_video(video_with_audio, language="en")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ok = check("video_id set", result.video_id is not None)
    if not ok:
        failures += 1

    ok = check("audio_metadata present", result.audio_metadata is not None)
    if not ok:
        failures += 1

    n_raw = len(result.raw_segments)
    ok = check("raw_segments non-empty", n_raw > 0, f"{n_raw}")
    if not ok:
        failures += 1

    ok = check(
        "fused_segments non-empty", len(result.fused_segments) > 0, f"{len(result.fused_segments)}"
    )
    if not ok:
        failures += 1

    ok = check("evidence records generated", len(result.evidence) > 0, f"{len(result.evidence)}")
    if not ok:
        failures += 1

    ok = check(
        "processing_time_seconds > 0",
        result.processing_time_seconds > 0,
        f"{result.processing_time_seconds * 1000:.1f} ms",
    )
    if not ok:
        failures += 1

    # Validate evidence records
    for ev in result.evidence:
        ok = check(
            "evidence.evidence_type == AUDIO_TRANSCRIPT",
            ev.evidence_type == EvidenceType.AUDIO_TRANSCRIPT,
            str(ev.evidence_type),
        )
        if not ok:
            failures += 1
            break
        ok = check(
            "evidence.video_id == pipeline.video_id",
            ev.video_id == result.video_id,
        )
        if not ok:
            failures += 1
            break

    print(f"  {INFO}  Pipeline (video_with_audio): {elapsed_ms:.1f} ms total")

    # Silent video: should return empty result, not raise
    t0 = time.perf_counter()
    silent_result = pipeline.process_video(video_silent, language="en")
    elapsed_ms_silent = (time.perf_counter() - t0) * 1000

    ok = check(
        "Silent video returns empty result (no error)",
        silent_result.raw_segments == [] and silent_result.evidence == [],
    )
    if not ok:
        failures += 1

    print(f"  {INFO}  Pipeline (silent_video): {elapsed_ms_silent:.1f} ms")

    return failures


# ── Section 7: FasterWhisper acceptance ───────────────────────────────────────


def _get_smoke_whisper_model() -> str | None:
    local_dir = Path("models/faster-whisper-tiny")
    if (
        local_dir.is_dir()
        and (local_dir / "model.bin").is_file()
        and (local_dir / "config.json").is_file()
    ):
        return str(local_dir)
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache("Systran/faster-whisper-tiny", "model.bin")
        if cached is not None:
            return "tiny"
    except Exception:  # noqa: BLE001
        return None
    return None


def test_faster_whisper(video_with_audio: Path) -> int:
    section("7. FasterWhisper Real Model (Runtime Acceptance)")
    failures = 0

    try:
        import faster_whisper  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        skip("faster-whisper not installed — skipping real model test")
        return 0

    model_ref = _get_smoke_whisper_model()
    if model_ref is None:
        skip("faster-whisper tiny model not locally cached — skipping")
        return 0

    try:
        cfg = FasterWhisperConfig(
            model_size_or_path=model_ref,
            device="cpu",
            compute_type="int8",
            word_timestamps=True,
        )
        from videx.audio.whisper import FasterWhisperASRProvider

        provider = FasterWhisperASRProvider(cfg)
        provider.warmup()
        check("Model loaded and warmed up", True)

        # 1. Real speech test if deterministic fixture exists
        english_fixture = Path("tests/fixtures/audio/english_speech.wav")
        if english_fixture.is_file():
            t0 = time.perf_counter()
            speech_segments = provider.transcribe(str(english_fixture), language="en")
            speech_elapsed_ms = (time.perf_counter() - t0) * 1000

            ok = check(
                "Real speech: non-empty transcript produced",
                len(speech_segments) > 0,
                f"{len(speech_segments)} segments",
            )
            if not ok:
                failures += 1

            for s in speech_segments:
                valid_ts = (
                    s.start_timestamp_seconds >= 0.0
                    and s.end_timestamp_seconds > s.start_timestamp_seconds
                    and s.language == "en"
                )
                ts_detail = f"[{s.start_timestamp_seconds:.2f}s - {s.end_timestamp_seconds:.2f}s]"
                ok = check(
                    "Speech segment: valid timestamps & language=='en'",
                    valid_ts,
                    f"{ts_detail} lang={s.language}",
                )
                if not ok:
                    failures += 1
                ok = check(
                    "Speech segment: word timestamps preserved",
                    len(s.words) > 0,
                    f"{len(s.words)} words extracted",
                )
                if not ok:
                    failures += 1
                has_expected = (
                    "speech recognition" in s.normalized_text.lower()
                    or "speech" in s.normalized_text.lower()
                )
                ok = check(
                    "Speech segment: raw_text matches expected content",
                    has_expected,
                    f"'{s.raw_text}'",
                )
                if not ok:
                    failures += 1

            print(f"  {INFO}  Real speech inference latency: {speech_elapsed_ms:.1f} ms")

        # 2. Non-speech sine audio test
        audio_arr, _ = extract_audio_stream(video_with_audio, target_sample_rate=16000)
        t0 = time.perf_counter()
        segments = provider.transcribe(audio_input=audio_arr, language="en")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        check("FasterWhisper: non-speech audio handled gracefully", True)
        print(f"  {INFO}  Sine inference: {elapsed_ms:.1f} ms | {len(segments)} segments")

    except Exception as exc:  # noqa: BLE001
        check(f"FasterWhisper test encountered error: {exc}", False)
        failures += 1

    return failures


# ── Section 8: Benchmarks ─────────────────────────────────────────────────────


def run_benchmarks(video_with_audio: Path) -> None:
    section("8. Micro-Benchmarks & RTF Measurement")

    # 1. Mock pipeline micro-benchmark
    pipeline = AudioPipeline.create_mock(config=MockASRConfig(segment_duration_seconds=1.0))
    pipeline.process_video(video_with_audio, language="en")

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        pipeline.process_video(video_with_audio, language="en")
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    print(f"  {INFO}  AudioPipeline (mock, 3s video, 5 runs):")
    print(f"         avg={avg_ms:.1f} ms | min={min_ms:.1f} ms | max={max_ms:.1f} ms")

    # 2. Real Faster-Whisper RTF benchmark (if model available)
    model_ref = _get_smoke_whisper_model()
    english_fixture = Path("tests/fixtures/audio/english_speech.wav")
    if model_ref is not None and english_fixture.is_file():
        import wave

        with wave.open(str(english_fixture), "rb") as w:
            audio_duration = w.getnframes() / float(w.getframerate())

        cfg = FasterWhisperConfig(
            model_size_or_path=model_ref,
            device="cpu",
            compute_type="int8",
        )
        from videx.audio.whisper import FasterWhisperASRProvider

        provider = FasterWhisperASRProvider(cfg)
        provider.warmup()

        real_times = []
        last_segs = []
        for _ in range(3):
            t0 = time.perf_counter()
            last_segs = provider.transcribe(str(english_fixture), language="en")
            real_times.append(time.perf_counter() - t0)

        avg_proc_time = sum(real_times) / len(real_times)
        rtf = avg_proc_time / audio_duration
        total_words = sum(len(s.words) for s in last_segs)

        print(f"\n  {BOLD}Real Faster-Whisper Benchmark Metrics:{RESET}")
        print(f"    • Model:           tiny ({model_ref})")
        print(f"    • Device:          {cfg.device}")
        print(f"    • Compute type:    {cfg.compute_type}")
        print(f"    • Audio duration:  {audio_duration:.2f} s")
        print(f"    • Processing time: {avg_proc_time:.3f} s (avg of 3 runs)")
        rtf_desc = "Real-time capable (RTF < 1.0)" if rtf < 1.0 else "Slower than real-time"
        print(f"    • RTF:             {rtf:.3f} ({rtf_desc})")
        print(f"    • Segments:        {len(last_segs)}")
        print(f"    • Words:           {total_words}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}VIDEX Phase 4.0 — Audio Intelligence Smoke Test{RESET}")
    print(f"{BLUE}Testing AudioExtraction, ASR, Fusion, and Pipeline...{RESET}")

    total_failures = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        video_with_audio = tmppath / "test_audio.mp4"
        video_silent = tmppath / "test_silent.mp4"

        print(f"\n{INFO}  Creating synthetic test fixtures...")
        create_video_with_audio(video_with_audio, fps=25.0, duration=3.0)
        create_silent_video(video_silent, fps=25.0, duration=2.0)
        print(f"  {INFO}  video_with_audio: {video_with_audio.stat().st_size:,} bytes")
        print(f"  {INFO}  video_silent:     {video_silent.stat().st_size:,} bytes")

        total_failures += test_audio_metadata(video_with_audio, video_silent)
        total_failures += test_audio_stream(video_with_audio)
        total_failures += test_normalization()
        total_failures += test_mock_provider(video_with_audio)
        total_failures += test_fusion()
        total_failures += test_pipeline(video_with_audio, video_silent)
        total_failures += test_faster_whisper(video_with_audio)
        run_benchmarks(video_with_audio)

    section("Final Result")
    if total_failures == 0:
        print(f"  {PASS}  All smoke tests passed!\n")
        sys.exit(0)
    else:
        print(f"  {FAIL}  {total_failures} smoke test(s) FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
