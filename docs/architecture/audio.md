# VIDEX Phase 4.0 — Audio Intelligence Architecture

## Overview

Phase 4.0 introduces the **Audio Intelligence Pipeline** for VIDEX: a fully
provider-abstracted, temporally-anchored speech recognition and acoustic event
detection system. It integrates natively with the established Phase 1–3
ingestion, perception, and OCR evidence architecture.

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Provider abstraction** | `ASRProvider` and `SoundEventProvider` protocols — swappable backends |
| **Temporal provenance** | Every `TranscriptSegment` carries explicit `start_timestamp_seconds` / `end_timestamp_seconds` |
| **No visual sync** | ASR results are never derived from `frame_number` or estimated fps |
| **Immutable segments** | `TranscriptSegment` is a frozen Pydantic model |
| **Conservative normalization** | Unicode NFKC + ZWJ/ZWNJ preservation; `raw_text` always retained |
| **CI hermeticity** | All unit and acceptance tests pass without downloading model weights |
| **Evidence propagation** | `AudioPipeline` converts fused segments into `Evidence(EvidenceType.AUDIO_TRANSCRIPT)` |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    VideoReader (Phase 1)                        │
│         DecodedFrame → FrameTimestamp (authoritative)           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ video path / Video domain object
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AudioPipeline                               │
│                                                                 │
│   1. extract_audio_stream()     — PyAV demux, 16kHz float32    │
│   2. ASRProvider.transcribe()   — faster-whisper / mock        │
│   3. normalize_transcript()     — NFKC, ZWJ/ZWNJ safe          │
│   4. TemporalTranscriptFusion() — deduplication                │
│   5. _generate_evidence()       — EvidenceType.AUDIO_TRANSCRIPT│
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
          AudioPipelineResult          Evidence[]
          ├── video_id                 ├── evidence_type
          ├── audio_metadata           ├── video_id
          ├── raw_segments             ├── timestamp_seconds
          ├── fused_segments           ├── metadata.end_timestamp_seconds
          ├── evidence                 ├── supporting_observation_ids
          └── processing_time_seconds  └── tags: [audio, asr, lang, provider]
```

---

## Module Map

```
src/videx/audio/
├── __init__.py         — Public re-exports
├── base.py             — Configs (FasterWhisperConfig, MockASRConfig) +
│                         resolve_audio_input() + ASRProvider/SoundEventProvider protocols
├── extraction.py       — PyAV-based audio stream extraction
│                         extract_audio_metadata(), extract_audio_stream(),
│                         extract_audio_bytes_wav(), extract_audio_to_wav()
├── normalization.py    — normalize_transcript() — NFKC + whitespace collapse
│                         Preserves Indic ZWJ (U+200D), ZWNJ (U+200C)
│                         Strips BOM (U+FEFF), zero-width space (U+200B)
├── mock.py             — MockASRProvider: deterministic, zero-weight ASR
├── whisper.py          — FasterWhisperASRProvider: CTranslate2-accelerated
│                         word-level timestamps, VAD, language routing
├── fusion.py           — TemporalTranscriptFusion: temporal deduplication
│                         of overlapping/repeating sliding-window segments
└── pipeline.py         — AudioPipeline orchestrator: wraps all of the above
                          process_video() / process_audio() → AudioPipelineResult
```

---

## Domain Schema

### `AudioMetadata`
Captures provenance metadata from the audio stream header.

| Field | Type | Description |
|---|---|---|
| `sample_rate` | `int` | Decoded sample rate in Hz |
| `channels` | `int` | Number of audio channels |
| `duration_seconds` | `float` | Stream duration |
| `codec_name` | `str \| None` | Container codec (e.g. `aac`, `mp2`) |
| `bit_rate` | `int \| None` | Bit rate in bps |
| `layout_name` | `str \| None` | Channel layout (e.g. `mono`, `stereo`) |

### `WordTimestamp`
Word-level alignment from ASR provider.

| Field | Type | Description |
|---|---|---|
| `word` | `str` | Token surface form |
| `start_timestamp_seconds` | `float` | Word start (provider-authoritative) |
| `end_timestamp_seconds` | `float` | Word end (provider-authoritative) |
| `confidence` | `float \| None` | Per-word confidence [0, 1] |

### `TranscriptSegment`
Core ASR result unit. Immutable, self-contained.

| Field | Type | Description |
|---|---|---|
| `segment_id` | `UUID` | Unique segment identifier |
| `video_id` | `UUID \| None` | Parent video association |
| `start_timestamp_seconds` | `float` | Segment start (ASR-authoritative) |
| `end_timestamp_seconds` | `float` | Segment end (ASR-authoritative) |
| `raw_text` | `str` | Original ASR output verbatim |
| `normalized_text` | `str` | NFKC-normalized transcript |
| `language` | `str` | ISO 639-1 language code |
| `confidence` | `float` | Segment confidence [0, 1] |
| `provider` | `str` | ASR backend identifier |
| `words` | `list[WordTimestamp]` | Word-level timestamps |
| `speaker_id` | `str \| None` | Speaker diarization label |
| `no_speech_prob` | `float \| None` | No-speech probability [0, 1] |
| `attributes` | `dict` | Provider-specific extras |

### `SoundObservation`
Acoustic event detection hook.

| Field | Type | Description |
|---|---|---|
| `video_id` | `UUID` | Parent video |
| `start/end_timestamp_seconds` | `float` | Event window |
| `label` | `str` | Event class (e.g. `siren`, `applause`) |
| `confidence` | `float` | Detection confidence [0, 1] |
| `provider` | `str` | Acoustic event backend |

---

## Provider Protocols

Both are `typing.Protocol` with `@runtime_checkable`:

### `ASRProvider`
```python
class ASRProvider(Protocol):
    @property
    def provider_name(self) -> str: ...
    @property
    def supported_languages(self) -> tuple[str, ...] | list[str]: ...
    def warmup(self) -> None: ...
    def transcribe(
        self,
        audio_input: Any,  # Path | bytes | np.ndarray[float32]
        video_id: UUID | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[TranscriptSegment]: ...
```

### `SoundEventProvider`
```python
class SoundEventProvider(Protocol):
    @property
    def provider_name(self) -> str: ...
    def warmup(self) -> None: ...
    def detect_events(
        self,
        audio_input: Any,
        video_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[SoundObservation]: ...
```

---

## Extraction Layer

`extract_audio_stream()` uses **PyAV** (libav wrapper) for non-corrupting
demuxing and format conversion:

1. Opens the container read-only
2. Locates the first audio stream
3. Resamples to `target_sample_rate` Hz (default 16 kHz), mono, `fltp` format
4. Concatenates frames into a single `np.ndarray[float32]`
5. Returns `(audio_array, AudioMetadata)`

**Error hierarchy:**
- `AudioExtractionError` — base class
- `NoAudioStreamError` — container has no audio (video-only source)
- `CorruptAudioError` — decoding failure
- `EmptyAudioError` — stream decoded 0 samples

---

## ASR Providers

### `FasterWhisperASRProvider`
- Backend: `faster-whisper` (CTranslate2-accelerated)
- Supports `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
- Word-level timestamps via `word_timestamps=True`
- VAD filtering for silence removal
- Confidence from `avg_logprob` (exp-transformed)
- No-speech probability from `no_speech_prob`
- Language detection from `TranscriptionInfo.language`

**Key config fields (`FasterWhisperConfig`):**
| Field | Default | Description |
|---|---|---|
| `model_size_or_path` | `"large-v3"` | Model name or local path |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `compute_type` | `"int8"` | Quantization type |
| `language` | `None` | Language hint; `None` = auto-detect |
| `word_timestamps` | `True` | Word-level alignment |
| `vad_filter` | `True` | VAD silence filtering |
| `beam_size` | `5` | Beam search width |

### `MockASRProvider`
- Zero-weight deterministic provider for hermetic testing
- Supports `canned_segments` for explicit segment injection
- Generates synthetic segments with word timestamps
- Supports Hindi (Devanagari) and English

---

## Temporal Transcript Fusion

`TemporalTranscriptFusion` merges ASR segments that represent the same
speech due to sliding-window overlap or repeated output:

**Merge criteria (all must hold):**
1. Same `video_id` and `language`
2. Temporal gap ≤ `max_gap_seconds` (default: 0.3s)
3. One of:
   - Identical `normalized_text` (sliding-window repeat)
   - Overlapping segments where one is a substring of the other

**Merge result:**
- `start_timestamp_seconds` = minimum across cluster
- `end_timestamp_seconds` = maximum across cluster
- `raw_text` / `normalized_text` from highest-confidence segment
- `confidence` = mean across cluster
- `supporting_segment_ids` stored in `attributes`

---

## Evidence Generation

`AudioPipeline._generate_evidence()` converts each fused `TranscriptSegment`
into an `Evidence` record:

```python
Evidence(
    evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
    source_module=f"asr_{seg.provider}",
    video_id=video_id,
    frame_id=None,  # Audio evidence is not frame-anchored
    timestamp_seconds=seg.start_timestamp_seconds,
    confidence=seg.confidence,
    description=f"Audio transcript [{lang}]: {normalized_text}",
    raw_payload={  # Full provenance payload
        "raw_text": ...,
        "normalized_text": ...,
        "start_timestamp_seconds": ...,
        "end_timestamp_seconds": ...,
        "duration_seconds": ...,
        "language": ...,
        "provider": ...,
        "words": [...],
    },
    supporting_observation_ids=[seg.segment_id, ...],
    tags=["audio", "asr", lang, provider],
    metadata={"end_timestamp_seconds": seg.end_timestamp_seconds},
)
```

> **No visual frame anchoring**: `frame_id` is always `None` for audio evidence.
> Temporal position is expressed via `timestamp_seconds` (segment start) and
> `metadata["end_timestamp_seconds"]` (segment end).

---

## Text Normalization

`normalize_transcript()` applies conservative Unicode normalization:

| Operation | Detail |
|---|---|
| NFKC normalization | Decomposes compatibility characters, recomposes canonically |
| Whitespace collapse | Multiple spaces → single space |
| Strip leading/trailing | `str.strip()` |
| BOM removal | U+FEFF stripped |
| Zero-width space removal | U+200B stripped |
| **ZWJ/ZWNJ preserved** | U+200D and U+200C retained (Indic script integrity) |

`raw_text` is **always preserved** unchanged alongside `normalized_text`.

---

## Configuration (`videx.config.Settings`)

| Field | Default | Description |
|---|---|---|
| `asr_device` | `"cpu"` | Inference device |
| `asr_compute_type` | `"int8"` | Quantization type |
| `asr_model_size` | `"large-v3"` | Model name or path |
| `asr_language` | `"en"` | Primary language hint |
| `asr_word_timestamps` | `True` | Word-level timestamp extraction |
| `asr_vad_filter` | `True` | VAD-based silence filtering |
| `asr_beam_size` | `5` | Beam search width |

---

## Performance Benchmarks (Phase 4.0 & Phase 4.0R)

Measured on a 3-second synthetic video (440 Hz sine wave, 16 kHz, AAC) and 4.10-second English speech fixture:

| Operation | Latency / RTF |
|---|---|
| `extract_audio_metadata()` | ~6–8 ms |
| `extract_audio_stream()` (3s) | ~22 ms |
| `MockASRProvider.transcribe()` | ~0.6–2 ms |
| `TemporalTranscriptFusion.fuse()` | <1 ms |
| `AudioPipeline.process_video()` (mock, 3s) | ~23–25 ms avg |

### Real Faster-Whisper Runtime Benchmark (Phase 4.0R)
Measured on CPU with int8 quantization:
- **Model:** `tiny` (`Systran/faster-whisper-tiny` / `models/faster-whisper-tiny`)
- **Device:** `cpu`
- **Compute type:** `int8`
- **Audio duration:** `4.10 s`
- **Processing time:** `0.985 s` (3-run average)
- **Real-Time Factor (RTF):** `0.240` (processing_time / audio_duration < 1.0; ~4.2x faster than real-time)
- **Segments produced:** `1`
- **Words produced:** `10`

---

## Sound Event Status (Phase 4.0R)

> **Important architectural demarcation:**
> - `SoundEventProvider` is strictly an **interface / hook protocol** (`@runtime_checkable Protocol`) defining the standard contract: `detect_events(audio_input, video_id, **kwargs) -> list[SoundObservation]`.
> - **Real sound-event detector models** (e.g. YAMNet, AudioSpectrogramTransformer) are explicitly scheduled for a **future phase**.
> - No acoustic sound-event model is loaded or executed in Phase 4.0R.

---

## Testing

| Test Suite | Location | Count |
|---|---|---|
| Unit tests | `tests/unit/test_audio.py` | 24 passed |
| Integration acceptance | `tests/integration/test_audio_acceptance.py` | 39 passed, 1 skipped (Hindi fixture check) |
| Smoke test | `scripts/smoke_test_audio.py` | 8 sections passed |

**Real-model gate** (`TestFasterWhisperAcceptance`):
- Executes real speech transcription on `tests/fixtures/audio/english_speech.wav`
- Verifies full execution chain: `video -> extraction -> FasterWhisperASRProvider -> TranscriptSegment -> Evidence`
- Verifies word timestamps, valid start/end bounds, normalized text sync, and zero visual-FPS drift.
- Hindi acceptance tests honest local availability: skips gracefully when no local Hindi audio fixture is present rather than fabricating a false acceptance.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `av` | ≥14.0 | PyAV audio demuxing and resampling |
| `numpy` | ≥1.26 | Float32 audio arrays |
| `faster-whisper` | ≥1.2.1 | CTranslate2 ASR inference |
| `pydantic` | ≥2.x | Domain model validation |

---

## Integration with Prior Phases

| Phase | Role in Phase 4 |
|---|---|
| Phase 1 (Ingestion) | `Video` domain object passed to `AudioPipeline.process_video()` |
| Phase 2 (Perception) | `Evidence` schema shared; `EvidenceType.AUDIO_TRANSCRIPT` added |
| Phase 3 (OCR) | Provider protocol pattern replicated; evidence format identical |

---

## Future Extension Points

- **Sound event detection**: `SoundEventProvider` protocol hook is defined; concrete detector model implementation in future phase.
- **Speaker diarization**: `TranscriptSegment.speaker_id` is already a first-class field.
- **Multimodal fusion**: `Evidence.supporting_observation_ids` links ASR evidence to OCR/detection evidence from overlapping time windows.
