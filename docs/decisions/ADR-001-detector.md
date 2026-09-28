# ADR-001 — Object Detector Selection

**Date:** 2026-09-28  
**Status:** Proposed  
**Deciders:** Swastik Pandey  

---

## Context

VIDEX requires an object detection component that:

1. Runs locally without mandatory cloud vendor dependency.
2. Supports CPU execution for development/testing, scaling to GPU in production.
3. Detects core surveillance and scene-understanding classes (persons, vehicles, common physical objects).
4. Wraps cleanly behind the abstract `DetectionProvider` protocol (`src/videx/providers/base.py`).
5. Maintains Python ≥ 3.11 compatibility.

No empirical benchmarks have been executed on representative video clips yet. This ADR documents the canonical candidates, alternatives, conditional activation rules, and the benchmark protocol required before promoting any candidate to `Accepted`.

---

## Decision Drivers

| Priority | Criterion | Description |
|:---:|---|---|
| 1 | Detection Accuracy | mAP50 / mAP50-95 on target domain objects |
| 2 | Latency & Throughput | FPS on CPU (dev) and GPU (prod) |
| 3 | Protocol Ergonomics | Seamless integration behind `DetectionProvider` |
| 4 | Vocabulary Flexibility | Fixed-vocabulary efficiency vs open-vocabulary zero-shot capabilities |
| 5 | Operational Footprint | Memory / VRAM consumption and dependency stability |
| 6 | License Compatibility | Permissive (MIT / Apache 2.0 preferred) |

---

## Canonical Candidates

### Candidate 1 (Baseline Candidate) — YOLO26

- **Role:** Default, primary object detection backbone for known object classes.
- **Characteristics:** State-of-the-art balance of real-time throughput and bounding-box accuracy on standard object vocabularies.
- **Integration:** Wrappable behind `DetectionProvider` returning standardized `Detection` domain models with normalized `BoundingBox`.
- **Status:** **Proposed** (Awaiting benchmark validation).

### Candidate 2 (Conditional Candidate) — YOLOE-26 (Open-Vocabulary)

- **Role:** Conditional open-vocabulary detector.
- **Condition for Activation:** **NOT mandatory.** Activated *only* when evaluation demonstrates that a fixed vocabulary is a measurable limitation for target intelligence queries, or when arbitrary user-prompted object classes must be detected zero-shot.
- **Trade-off:** Additional embedding latency and higher resource overhead compared to YOLO26.
- **Status:** **Proposed** (Conditional route).

---

## Alternatives Considered & Retired References

### Alternative A — PP-YOLOE+ (PaddleDetection)
- **Strengths:** Apache 2.0 license, strong accuracy, efficient inference runtimes.
- **Weaknesses:** Requires PaddlePaddle runtime environment, introducing ecosystem fragmentation unless PP-OCRv6 is co-deployed.
- **Verdict:** Retained as a secondary fallback alternative.

### Alternative B — YOLOv8 / YOLO-World (Ultralytics)
- **Strengths:** Established developer ecosystem, extensive community tooling.
- **Weaknesses:** AGPL-3.0 licensing constraints, superseded by YOLO26 architectural efficiencies.
- **Verdict:** Reference baseline only; not the primary forward candidate.

### Retired Reference — EVA (BAAI)
- **Reason for Removal:** Stale reference. The massive transformer backbone imposes prohibitive operational and VRAM costs, requiring dedicated GPU infrastructure that conflicts with VIDEX's lightweight, decoupled edge/hybrid deployment goals. Removed from active consideration.

---

## Decision Status & Promotion Criteria

**Current Status:** **PROPOSED** (No model is marked benchmark-proven or Accepted).

### Required Benchmarking Protocol for Promotion

Before promoting YOLO26 (or conditionally enabling YOLOE-26) to `Accepted`:

1. **Benchmark Dataset:** Assemble a representative dataset of 10–20 video clips (diverse resolutions, lighting conditions, occlusion densities, motion blur).
2. **Evaluation Metrics:**
   - mAP50 and mAP50:95 against annotated target classes.
   - Inference latency per frame (ms) on CPU (Intel/AMD) and GPU (NVIDIA CUDA).
   - Peak RAM and VRAM footprint during sustained video processing.
   - Bounding-box coordinate jitter across sequential frames.
3. **Open-Vocabulary Threshold Test (YOLOE-26):**
   - Measure detection recall for out-of-vocabulary terms against latency penalty.
   - Verify if text prompt embeddings justify the operational cost overhead.
4. **Promotion Rule:** Promotion requires reproducible benchmark results demonstrating target accuracy at or above SLA without exceeding memory thresholds.

---

## Architecture Alignment

- **Provider Protocol:** Isolated strictly behind `DetectionProvider`.
- **Pipeline Position:** Invoked by the Frame Router during the Multimodal Perception stage, streaming detections into the **Temporal Event Engine** and persisting raw items into the **Evidence Store**.
- **Related PUML Diagram:** [03-technology-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/03-technology-architecture.puml).
