# ADR-002 — Multi-Object Tracker Selection

**Date:** 2026-09-28  
**Status:** Proposed  
**Deciders:** Swastik Pandey  

---

## Context

VIDEX requires a multi-object tracking (MOT) component that:

1. Associates frame-level detections into persistent spatio-temporal tracks (`Track` domain models).
2. Maintains track identities across temporary occlusions, camera movements, and frame gaps.
3. Computes trajectory vectors, velocities, and lifecycle states (birth, active, lost, deleted).
4. Decouples completely from the detector implementation via the `TrackingProvider` protocol (`src/videx/providers/base.py`).
5. Executes efficiently on CPU during development, with scalable GPU support in production.

No formal multi-object tracking benchmarks have been executed on target VIDEX footage. This ADR defines the canonical tracker candidates, conditional ReID requirements, and the benchmarking criteria required before promotion to `Accepted`.

---

## Decision Drivers

| Priority | Criterion | Description |
|:---:|---|---|
| 1 | Identity Consistency | Low ID switch rate (IDSW) across occlusions |
| 2 | Tracking Accuracy | Higher HOTA and MOTA metrics |
| 3 | Execution Throughput | Real-time or near real-time frame rates on CPU / GPU |
| 4 | ReID Modularity | Ability to run lightweight motion-only mode or conditional appearance ReID |
| 5 | Protocol Ergonomics | Simple integration behind `TrackingProvider` |
| 6 | License Compatibility | Permissive open-source licensing (MIT / Apache 2.0) |

---

## Canonical Candidates

### Candidate 1 (Default Candidate) — BoT-SORT

- **Role:** Default multi-object tracking candidate.
- **Architecture:** Combines Kalman filtering, camera motion compensation (CMC), and Hungarian matching, with an optional appearance ReID feature extractor.
- **Operational Mode:**
  - **Motion-Only Mode (Default):** Runs Kalman filter + IoU + CMC without appearance embeddings for high-throughput CPU/GPU processing.
  - **ReID Mode (Conditional):** **Explicitly conditional.** ReID feature extraction is enabled *only* if empirical evaluation on domain footage demonstrates unacceptable ID switching during prolonged occlusions or cross-object crossovers.
- **License:** MIT.
- **Status:** **Proposed** (Awaiting benchmark validation).

### Candidate 2 (Alternative Candidate) — ByteTrack

- **Role:** Fallback / secondary high-speed tracking alternative.
- **Architecture:** Pure motion-and-association tracker that retains low-confidence detections in a secondary association step to recover occluded objects.
- **Strengths:** Extremely fast on CPU (no neural network inference inside tracker loop), minimal memory footprint, zero ReID overhead.
- **Weaknesses:** Lacks appearance cues; vulnerable to identity swaps when objects have overlapping trajectories and varying velocities.
- **License:** MIT.
- **Status:** **Proposed** (Alternative candidate).

---

## Other Alternatives Considered

### Alternative C — OC-SORT (Observation-Centric SORT)
- **Strengths:** Re-observation mechanisms reduce error accumulation during long occlusions; pure motion-based.
- **Weaknesses:** Integration ecosystem is less mature than BoT-SORT/ByteTrack.
- **Verdict:** Secondary candidate if benchmark scenes exhibit severe occlusions without appearance variance.

### Alternative D — StrongSORT
- **Strengths:** Advanced appearance matching and temporal error correction.
- **Weaknesses:** Deep feature extractor runs continuously on every detection, causing substantial latency penalties that violate CPU throughput goals.
- **Verdict:** Rejected as default; too computationally heavy for baseline pipeline.

### Legacy — DeepSORT
- **Verdict:** Superseded by ByteTrack and BoT-SORT; retained purely as a historical reference point.

---

## Decision Status & Promotion Criteria

**Current Status:** **PROPOSED** (No tracker is marked benchmark-proven or Accepted).

### Required Benchmarking Protocol for Promotion

Before promoting BoT-SORT (or adopting ByteTrack as default) to `Accepted`:

1. **Benchmark Footage:** Select 5–10 representative clips with:
   - High object density (crowded scenes).
   - Temporary occlusions (objects passing behind obstacles or other objects).
   - Moving camera / pan-tilt-zoom shifts.
2. **Evaluation Metrics:**
   - HOTA (Higher Order Tracking Accuracy) and MOTA (Multiple Object Tracking Accuracy).
   - ID Switches (IDSW) and Fragmentations (Frag).
   - Association accuracy (IDF1).
   - Tracking loop latency (FPS on CPU and GPU).
3. **ReID Justification Test:**
   - Benchmark BoT-SORT in motion-only mode vs BoT-SORT with ReID embeddings.
   - ReID may only be activated in production if it reduces IDSW by a statistically significant margin (e.g., >30%) that justifies the added GPU/compute latency.
4. **Promotion Rule:** Promotion requires verified HOTA/IDSW benchmarks meeting VIDEX accuracy targets without exceeding latency budgets.

---

## Architecture Alignment

- **Provider Protocol:** Abstracted via `TrackingProvider`.
- **Pipeline Position:** Follows detection in Multimodal Perception; outputs `Track` objects and `TrajectoryPoint`s to the **Temporal Event Engine** and saves raw tracking data to the **Evidence Store**.
- **Related PUML Diagrams:**
  - [02-processing-pipeline.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/02-processing-pipeline.puml)
  - [03-technology-architecture.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/03-technology-architecture.puml)
  - [04-evidence-graph.puml](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/architecture/04-evidence-graph.puml)
