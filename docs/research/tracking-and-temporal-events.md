# Research Note: Multi-Object Tracking & Temporal Event Reasoning

**Domain:** Spatio-Temporal Association & Event Intelligence  
**Target Component:** `src/videx/providers/tracker.py` & `src/videx/events/`  
**Status:** In Progress / Active Reference  
**Relevant ADR:** [ADR-002 — Multi-Object Tracker Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-002-tracker.md)

---

## 1. Problem and Engineering Question

A single-frame detection is an isolated, instantaneous observation: "a person is at coordinates $[x_1, y_1, x_2, y_2]$ at timestamp $t$." Real-world video intelligence requires understanding continuous temporal behavior:
- Is this the same worker who entered the hazardous zone at $t=0$?
- How long has the individual remained in the scene without protective headgear?
- What was the trajectory and velocity vector prior to an incident?

The core engineering questions are:
1. **Identity Association under Occlusion:** How can VIDEX maintain persistent object identities (`Track` models) through temporary visual occlusions, scale shifts, and camera movements without incurring heavy neural Re-Identification (ReID) latency overhead on CPU?
2. **Temporal Reasoning vs. Frame Sampling:** How does video sampling frequency (e.g. 30 FPS raw vs. 2–5 FPS keyframe extraction) affect Kalman filter motion models and Hungarian matching accuracy?
3. **Event Aggregation:** How should frame-level track states be aggregated into discrete, bounded temporal events (loitering, zone transitions, compliance lapses) suitable for evidence graph ingestion?

---

## 2. Current VIDEX Implementation and Evidence

### Architecture & Implementation
- **Provider Protocol:** Abstracted behind `TrackingProvider` in `src/videx/providers/base.py`, implemented via BoT-SORT in `src/videx/providers/tracker.py`.
- **Domain Models:** Converts raw detections into persistent `Track` entities containing unique UUIDs, trajectory histories, bounding box sequences, and activity states (`active`, `lost`, `terminated`).
- **Operational Mode:** Motion-only tracking (Kalman Filter + IoU association + Camera Motion Compensation). Appearance ReID features are disabled by default.

### Empirical Baseline Evidence (`Sample_Videos/ppe-1.mp4`)
From the baseline profiling report (`outputs/profiling/cpu_baseline_report.md`):
- **Total Tracked Frames:** 343 frames (11.44s).
- **Execution Latency:**
  - Mean: **$0.486\text{ ms/frame}$** ($\approx 2,049\text{ FPS}$)
  - Median: **$0.357\text{ ms}$**
  - P90: **$0.626\text{ ms}$**
  - P99: **$4.209\text{ ms}$**
  - Total Wall Time: $0.167\text{ s}$ across the entire video.
- **Resource Footprint:** Consumes less than $0.2\%$ of perception stage execution time.
- **Identity Stability:** Maintained a single, continuous track identity (`track_id=1`) across all 343 frames with $0$ ID switches and $0$ fragmentations on the single-worker PPE video.

---

## 3. Official Documentation and Primary Papers

### Primary Papers
1. **BoT-SORT (Aharon et al., 2022):**
   - *Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022).* "BoT-SORT: Robust Associations for Multi-Pedestrian Tracking." [arXiv:2206.14651](https://arxiv.org/abs/2206.14651).
   - *Key Mechanics:* Integrates Kalman filtering with Camera Motion Compensation (CMC) using optical flow / RANSAC affine transformation, refined bounding-box width-height state vectors ($[x, y, w, h]$ instead of $[x, y, a, h]$), and optional deep ReID feature fusion.
2. **ByteTrack (Zhang et al., 2022):**
   - *Zhang, Y., et al. (2022).* "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." *ECCV 2022*. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864).
   - *Key Mechanics:* Retains low-score detection boxes (confidence $0.10 \le \text{conf} < 0.60$) instead of discarding them. Associates high-confidence detections first, then uses low-confidence boxes in a secondary matching pass to recover partially occluded or blurred targets.
3. **Tracking Evaluation & Metrics (TrackEval / HOTA):**
   - *Luiten, J., et al. (2020).* "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking." *International Journal of Computer Vision*. [GitHub: TrackEval](https://github.com/JonathonLuiten/TrackEval).
   - *Metric Breakdown:*
     - **HOTA (Higher Order Tracking Accuracy):** Balances detection accuracy ($\text{DetA}$) and association accuracy ($\text{AssA}$).
     - **IDF1 (Identification F1):** Measures identity preservation over time, penalizing ID switches.
     - **MOTA (Multiple Object Tracking Accuracy):** Classical metric combining False Positives, False Negatives, and ID switches.
4. **Temporal Action & Event Recognition:**
   - *SlowFast Networks (Feichtenhofer et al., 2019):* Two-stream pathways capturing spatial semantics at low frame rates and rapid motion transitions at high frame rates. [arXiv:1812.03982](https://arxiv.org/abs/1812.03982).
   - *Temporal Segment Networks (Wang et al., 2016):* Long-range temporal structure modeling using sparse segment sampling. [arXiv:1705.02953](https://arxiv.org/abs/1705.02953).

---

## 4. Candidate Approaches and Hardware Requirements

| Candidate | Architecture / Mechanism | Hardware Overhead | Target Scenarios | Advantages | Disadvantages |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **BoT-SORT Motion-Only (Default)** | Kalman Filter + CMC + Hungarian Matching | Extremely Low (<1ms CPU) | Fixed / panning surveillance cameras | Camera motion resilience via CMC; zero neural overhead | Vulnerable to prolonged complete occlusion |
| **ByteTrack (Alternative)** | Two-stage low-score IoU association | Minimal (<0.5ms CPU) | High-density crowd scenes | Recovers occluded/blurred low-score boxes | Sensitive to large camera jerks without CMC |
| **BoT-SORT + ReID (Conditional)** | Motion + Deep Appearance Embedding (OSNet/FastReID) | High (+15–40ms/frame on CPU) | Multi-person crossover with prolonged exits | Re-acquires targets after long departures | Multiplies CPU/GPU inferencing cost significantly |

### Hardware Feasibility
- **Motion-Only Tracking:** Pure CPU execution requires minimal RAM (<5 MB state memory) and negligible CPU compute. Easily sustained on low-power devices.
- **Deep ReID Tracking:** Requires batching person crops through a feature extractor network, demanding either GPU acceleration or an optimized ONNX/OpenVINO embedding pipeline.

---

## 5. Expected Trade-offs and Failure Modes

1. **Sampling Frequency vs. Kalman Predictability:**
   - *Failure Mode:* Downsampling video to 1–2 FPS causes linear motion assumptions in Kalman filters to break down, resulting in failed Hungarian matching and rampant ID switches.
   - *Mitigation:* Decouple frame decoding rate from detection rate: run tracking at native video frame rate or maintain an adaptive velocity model that scales covariance matrices with $\Delta t$.
2. **Camera Motion Compensation (CMC) CPU Cost:**
   - *Failure Mode:* Calculating full-frame dense optical flow or multi-point homography transformations on high-resolution ($1080\text{p}/4\text{K}$) frames on CPU can spike tracking latency from $0.5\text{ ms}$ to $40+\text{ ms}$.
   - *Mitigation:* Downsample image resolution to $320\text{p}$ for optical flow feature point extraction, or disable CMC if camera is stationary.
3. **Identity Switching (IDSW) in Dense Crowds:**
   - *Failure Mode:* Two individuals with intersecting walking vectors swap track IDs during crossover.
   - *Mitigation:* Strict association thresholds; when IoU overlap is ambiguous, transition track state to `occluded` and suspend trajectory predictions rather than forcing a greedy assignment.

---

## 6. Proposed Experiment and Acceptance Criteria

### Experiment Protocol
1. **Benchmark Footage Collection:**
   - Assemble a MOT test suite consisting of 5 multi-person surveillance clips featuring:
     - Cross-camera occlusion (worker walking behind pillars or machinery).
     - Object-object crossing (two workers passing each other in opposing directions).
     - Scale changes (worker moving from foreground to deep background).
2. **TrackEval Integration:**
   - Run TrackEval harness against human-annotated MOT ground truth.
   - Compare BoT-SORT Motion-Only vs. ByteTrack across HOTA, MOTA, IDF1, and IDSW.

### Acceptance Criteria
- **HOTA:** $\ge 65.0$ on target surveillance scenes.
- **IDF1:** $\ge 75.0$ identity consistency.
- **ID Switch Rate:** $\le 2$ ID switches per minute of active multi-person video.
- **Execution Speed:** $\ge 1,000\text{ FPS}$ on multi-core CPU.

---

## 7. Actual Experiment Results (Kept Separate from Hypotheses)

### Single-Target Baseline (`Sample_Videos/ppe-1.mp4`, Recorded 2026-10-21)
- **Algorithm:** BoT-SORT Motion-Only (`src/videx/providers/tracker.py`).
- **Frames Processed:** 343 frames.
- **Throughput:** $2,049.32\text{ FPS}$ ($0.486\text{ ms}$ mean per frame).
- **Track Continuity:** 1 unique track generated, 0 ID switches, 0 false track splits.
- **Conclusion:** On single-subject footage with clean background contrast, motion-only tracking is virtually free in computational cost and maintains flawless trajectory continuity. Multi-person stress testing remains the next empirical milestone.

---

## 8. Decision, Rationale, and Link to Relevant ADR

- **Decision:** Maintain **BoT-SORT Motion-Only** as the primary tracking engine. Appearance ReID is designated as strictly conditional per ADR-002.
- **Rationale:** At $0.486\text{ ms/frame}$, BoT-SORT motion tracking is 500× faster than the current object detector, introducing zero pipeline bottleneck while delivering camera-motion compensated trajectory vectors.
- **Relevant ADR:** [ADR-002 — Multi-Object Tracker Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-002-tracker.md).
