# ADR-004 — Audio & Hindi ASR Engine Selection

**Date:** 2026-09-28  
**Status:** Proposed  
**Deciders:** Swastik Pandey  

---

## Context

VIDEX requires an Automatic Speech Recognition (ASR) and audio processing component to:

1. Transcribe speech from video audio streams, focusing on Hindi, English, and Hinglish (code-switched Hindi-English).
2. Generate precise segment-level and word-level timestamps aligned to the video master clock.
3. Run locally without mandatory third-party cloud services or per-minute API fees.
4. Integrate behind the `ASRProvider` protocol (`src/videx/providers/base.py`).
5. Execute efficiently on CPU for development, scaling to GPU in production.

No formal speech recognition benchmarks have been performed on domain-specific Hindi video tracks. This ADR records the canonical baseline candidate, the conditional criteria for specialized Hindi ASR, alternative engines, and promotion benchmarks.

---

## Decision Drivers

| Priority | Criterion | Description |
|:---:|---|---|
| 1 | Word Error Rate (WER) | High transcription accuracy on Hindi, conversational speech, and code-mixing |
| 2 | Timestamp Granularity | Accurate word and segment timestamps for temporal alignment |
| 3 | Real-Time Factor (RTF) | Fast inference on CPU (dev) and high throughput on GPU (prod) |
| 4 | Memory Footprint | Manageable model size (VRAM/RAM) without starving concurrent perception models |
| 5 | Protocol Ergonomics | Simple wrapping behind `ASRProvider` |
| 6 | License Compatibility | Permissive open-source licensing (MIT / Apache 2.0 preferred) |

---

## Canonical Candidates

### Candidate 1 (Baseline Candidate) — faster-whisper (CTranslate2)

- **Role:** Primary, baseline ASR candidate for all general speech transcription.
- **Architecture:** CTranslate2 reimplementation of OpenAI's Whisper model (quantized 8-bit/16-bit Transformer encoder-decoder).
- **Key Strengths:**
  - Up to 4× faster than standard PyTorch Whisper on CPU with significantly lower memory usage.
  - Native word-level and segment-level timestamp generation.
  - Strong multilingual pretraining covering Hindi and English code-switching out of the box.
  - MIT license.
- **Model Sizes for Evaluation:** `medium` (balance for CPU dev) and `large-v3` (high accuracy on GPU).
- **Status:** **Proposed** (Awaiting benchmark validation).

### Candidate 2 (Conditional Candidate) — Specialized Hindi ASR (e.g., IndicWhisper / AI4Bharat)

- **Role:** Specialized vernacular ASR route.
- **Condition for Activation:** **Explicitly conditional.** Adopted *only* if empirical benchmarking of `faster-whisper` demonstrates unacceptable WER (e.g., WER > 25%) on colloquial regional Hindi dialects, heavy accents, or specific vernacular domain jargon.
- **Trade-offs:** Requires additional fine-tuned model checkpoints and custom tokenizer configurations, increasing operational maintenance.
- **Status:** **Proposed** (Conditional route pending benchmark need).

---

## Alternatives Considered & Retired References

### Alternative A — Vanilla OpenAI Whisper (`openai-whisper`)
- **Strengths:** Reference PyTorch implementation, extensive documentation.
- **Weaknesses:** Substantially higher CPU latency and memory usage compared to `faster-whisper`.
- **Verdict:** Superseded by `faster-whisper`; kept only as a reference baseline.

### Alternative B — Meta MMS (Massively Multilingual Speech)
- **Strengths:** Covers thousands of languages using wav2vec2 backbones.
- **Weaknesses:** Weak code-switching handling; CC BY-NC 4.0 license restricts commercial deployment.
- **Verdict:** Excluded due to licensing and performance on conversational Indian speech.

### Alternative C — Managed Cloud APIs (Google Cloud Speech, Azure Speech)
- **Strengths:** Excellent accuracy, managed scaling.
- **Weaknesses:** External network dependency, recurring API costs, data privacy issues for confidential video footage.
- **Verdict:** Unsuitable for offline/air-gapped local VIDEX architecture.

---

## Decision Status & Promotion Criteria

**Current Status:** **PROPOSED** (No ASR model is marked benchmark-proven or Accepted).

### Required Benchmarking Protocol for Promotion

Before promoting `faster-whisper` (or activating a specialized Hindi ASR candidate) to `Accepted`:

1. **Benchmark Audio Clips:** Extract 10–15 audio segments (30–90 seconds each) from target domain videos covering:
   - Clear studio speech / voiceovers.
   - Conversational Hindi with ambient background noise (street/traffic/indoor).
   - Code-switched Hinglish dialogue.
2. **Evaluation Metrics:**
   - Word Error Rate (WER) and Character Error Rate (CER) against human-verified transcripts.
   - Timestamp alignment drift: delta (in milliseconds) between predicted word boundaries and ground-truth audio cues.
   - Real-Time Factor (RTF = processing_time / audio_duration) on CPU and GPU.
   - Peak RAM / VRAM allocation during audio transcription.
3. **Condition Check for Specialized Route:**
   - If `faster-whisper` achieves WER < 15% on conversational Hindi/Hinglish, it is promoted as sole ASR provider.
   - If `faster-whisper` fails to achieve acceptable WER on accented/colloquial speech, benchmark IndicWhisper on the failure set.
4. **Promotion Rule:** Promotion requires verified WER and RTF compliance on the target deployment profile.

---

## Architecture Alignment

- **Provider Protocol:** Abstracted via `ASRProvider`.
- **Pipeline Position:** Executes in parallel with visual decoding in the Audio Pipeline; feeds timestamped `AudioSegment` records into the **Temporal Event Engine** and persists raw audio evidence to the **Evidence Store**.
- **Related PUML Diagrams:**
  - [02-processing-pipeline.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/02-processing-pipeline.puml)
  - [03-technology-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/03-technology-architecture.puml)
  - [04-evidence-graph.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/04-evidence-graph.puml)
  - [06-processing-sequence.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/06-processing-sequence.puml)
