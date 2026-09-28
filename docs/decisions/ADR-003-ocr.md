# ADR-003 — OCR Architecture & Engine Selection

**Date:** 2026-09-28  
**Status:** Proposed  
**Deciders:** Swastik Pandey  

---

## Context

VIDEX requires an Optical Character Recognition (OCR) subsystem to detect, extract, and track text in video frames. Key domain constraints:

1. Target videos feature bilingual or multilingual text: Hindi (Devanagari script), English (Latin script), and mixed code overlays (e.g., news chyrons, signage, scene text).
2. Video text differs sharply from document text: low resolution, compression artifacts, dynamic backgrounds, motion blur, and brief on-screen lifespans.
3. Text must be temporally tracked and fused across sequential frames rather than treated as independent per-frame noise.
4. The subsystem must remain model-agnostic and wrap cleanly behind `OCRProvider` (`src/videx/providers/base.py`).
5. Execution must support CPU during development and scale efficiently in production.

No empirical benchmarks on target video text have been conducted. This ADR establishes the canonical **OCR Router** architecture, primary engine candidates, temporal OCR fusion strategy, and promotion criteria.

---

## Decision Drivers

| Priority | Criterion | Description |
|:---:|---|---|
| 1 | Devanagari & Latin Recognition | High character/word accuracy on complex Hindi video overlays |
| 2 | Text Localization Quality | Accurate bounding boxes for multi-line and rotated scene text |
| 3 | Temporal Consistency | Ability to fuse text detections across consecutive frames |
| 4 | Routing Flexibility | Modular script-based routing without monolithic dependencies |
| 5 | Latency & Resource Footprint | Efficient CPU/GPU execution without blocking the pipeline |
| 6 | License Compatibility | Permissive open-source licensing (Apache 2.0 / MIT) |

---

## Canonical Architecture: The OCR Router

Rather than forcing a single OCR engine to handle all scripts and frame conditions, VIDEX adopts a routed, multi-stage OCR architecture:

```
Video Frames
     │
     ▼
┌──────────────┐
│  OCR Router  │ ── Selects engine based on script, scene type & frame quality
└──────┬───────┘
       ├─────────────────────────────────┐
       ▼                                 ▼
┌──────────────┐            ┌─────────────────────────┐
│   PP-OCRv6   │            │   Hindi / Indic OCR     │
│  (Supported  │            │   (Specialized Route    │
│   Scripts)   │            │    for Devanagari)      │
└──────┬───────┘            └────────────┬────────────┘
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           ┌─────────────────────────┐
           │   Temporal OCR Fusion   │ ── Aggregates, deduplicates & tracks text
           └────────────┬────────────┘
                        ▼
                 Evidence Store
```

### 1. Route A (Supported Scripts Candidate) — PP-OCRv6
- **Role:** High-efficiency, general-purpose text detection and recognition for standard supported scripts (Latin, bilingual layouts).
- **Architecture:** Compact Differentiable Binarization (DB) detector + lightweight SVTR-based recognizer.
- **Strengths:** Optimized mobile and server models, Apache 2.0 license, excellent speed/accuracy trade-off on structured text overlays.
- **Status:** **Proposed** (Awaiting benchmark validation).

### 2. Route B (Specialized Indic Route) — Dedicated Hindi/Indic OCR
- **Role:** Targeted recognition for complex Devanagari ligatures, compound consonants (samyuktaksar), and low-contrast vernacular signage.
- **Candidate Engines:** IndicOCR / fine-tuned PaddleOCR Devanagari models.
- **Status:** **Proposed** (Conditional on script detection and benchmark results).

### 3. Temporal OCR Fusion Layer
- **Role:** Post-recognition temporal aggregator.
- **Mechanism:** Text bounding boxes and recognized strings from consecutive frames are tracked across time. Transcripts are fused via Levenshtein distance clustering, confidence voting, and temporal windowing.
- **Benefit:** Eliminates single-frame OCR flicker, repairs partial misrecognitions, and produces a consolidated `OCRObservation` with `first_seen` and `last_seen` timestamps.

---

## Alternatives Considered & Fallback Candidates

### Alternative C — EasyOCR
- **Strengths:** 80+ supported languages, straightforward PyTorch implementation, Apache 2.0.
- **Weaknesses:** CRAFT text detector is computationally heavy on CPU; recognition accuracy degrades on stylized Hindi fonts.
- **Verdict:** Valid development fallback, but secondary to PP-OCRv6.

### Alternative D — Surya
- **Strengths:** Modern transformer architecture; excellent layout parsing and line detection.
- **Weaknesses:** High VRAM footprint, requires GPU, dual-license (GPL-3.0 with commercial restrictions).
- **Verdict:** Excluded from baseline due to license constraints and resource weight.

### Alternative E — Tesseract 5 + Devanagari (`hin`)
- **Strengths:** Ultra-lightweight CPU execution, widely available packages.
- **Weaknesses:** Severe degradation on unconstrained video overlays and dynamic backgrounds.
- **Verdict:** Insufficient for primary video OCR; legacy fallback only.

---

## Decision Status & Promotion Criteria

**Current Status:** **PROPOSED** (No OCR engine or routing configuration is marked benchmark-proven or Accepted).

### Required Benchmarking Protocol for Promotion

Before promoting PP-OCRv6, the specialized Indic route, or Temporal OCR Fusion to `Accepted`:

1. **Benchmark Test Set:** 20–30 video clips containing:
   - News chyrons and crawling headlines (Hindi + English).
   - Vernacular storefronts and traffic signage.
   - Low-resolution and motion-blurred video scenes.
2. **Evaluation Metrics:**
   - Character Error Rate (CER) and Word Error Rate (WER) against ground-truth text annotations.
   - Text Bounding Box IoU and detection recall.
   - Temporal stability score (reduction in transient false alarms via Temporal OCR Fusion).
   - Inference latency per keyframe on CPU and GPU.
3. **Promotion Rule:** Promotion requires verified CER/WER thresholds on both Hindi and English video text sets while satisfying CPU/GPU latency budgets.

---

## Architecture Alignment

- **Provider Protocol:** Abstracted behind `OCRProvider`.
- **Pipeline Position:** Operates inside the Multimodal Perception stage; fed by the Frame Router, streaming fused observations into the **Temporal Event Engine** and persisting to the **Evidence Store**.
- **Related PUML Diagrams:**
  - [02-processing-pipeline.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/02-processing-pipeline.puml)
  - [03-technology-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/03-technology-architecture.puml)
  - [04-evidence-graph.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/04-evidence-graph.puml)
