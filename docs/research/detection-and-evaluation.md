# Research Note: Object Detection & Evaluation Science

**Domain:** Video Perception & Ground-Truth Integrity  
**Target Component:** `src/videx/providers/detector.py` (`DetectionProvider`)  
**Status:** In Progress / Active Reference  
**Relevant ADR:** [ADR-001 — Object Detector Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-001-detector.md)

---

## 1. Problem and Engineering Question

In VIDEX, the object detector is the entry-point perceptual sensor for video frames. Every downstream component—multi-object tracking, spatial evidence extraction, temporal event triggers, and semantic evidence graph nodes—relies on bounding box coordinates and class predictions emitted at this stage.

The core engineering questions are:
1. **Model Selection & Throughput:** How can VIDEX achieve real-time or near-real-time throughput (≥10 FPS) on standard x86 CPU hardware while maintaining high bounding-box fidelity for surveillance, PPE, and workplace scenes?
2. **Evaluation Integrity:** How do we establish a trustworthy ground-truth evaluation harness that prevents data leakage, enforces strict one-to-one matching, avoids sampling bias, and separates detector accuracy from runtime engine parity?

---

## 2. Current VIDEX Implementation and Evidence

### Architecture & Implementation
- **Provider Protocol:** Abstracted behind `DetectionProvider` in `src/videx/providers/base.py`, implemented in `src/videx/providers/detector.py`.
- **Default Checkpoint:** `models/yolo26n.pt` (Ultralytics YOLO26 Nano PyTorch CPU).
- **Execution Settings:** Default confidence threshold $0.25$, sequential frame decode via OpenCV (`cv2.VideoCapture`).
- **Domain Output:** Emits standardized `Detection` domain objects with normalized coordinates $[x_1, y_1, x_2, y_2] \in [0, 1]$.

### Empirical Baseline Evidence (`Sample_Videos/ppe-1.mp4`)
From the isolated baseline profiling run (`outputs/profiling/cpu_baseline_ppe1.json`):
- **Target Video:** 1280×720 @ 29.97 fps, 343 frames, duration 11.44s.
- **Hardware:** Intel 12 Physical / 16 Logical Cores, Windows 11, PyTorch 2.14.0+cpu.
- **Inference Latency (Isolated YOLO26n):**
  - Mean: **$280.23\text{ ms}$** ($\approx 3.57\text{ FPS}$)
  - Median: **$225.72\text{ ms}$**
  - P90: **$407.79\text{ ms}$**
  - P99: **$1117.48\text{ ms}$**
  - Total Wall Time: $96.12\text{ s}$ across 343 frames.
- **Perception Time Share:** YOLO26n inference accounts for over $95\%$ of isolated perception compute (compared to $7.98\text{ ms}$ decode and $0.486\text{ ms}$ tracking).
- **Ground-Truth Audit Status:** 12 keyframes audited (`docs/evaluation/annotation-verification-report.md`). Initial human labels exhibited boundary truncation (omitting hard hats and extended hands), prompting an audit review under Policy A (`docs/evaluation/annotation-guidelines.md`).

---

## 3. Official Documentation and Primary Papers

### Primary Papers
1. **YOLO (You Only Look Once):**
   - *Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).* "You Only Look Once: Unified, Real-Time Object Detection." *IEEE CVPR 2016*. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640).
   - *Significance:* Establishes one-stage spatial regression over dense anchor/anchor-free grid cells, trading marginal two-stage AP for order-of-magnitude FPS gains.
2. **Ultralytics YOLO Evolution (YOLOv8 to YOLO26):**
   - *Ultralytics Framework Architecture:* Anchor-free decoupled head architecture separating objectness, bounding-box regression (Distribution Focal Loss), and classification branches.
   - *Official Documentation:*
     - [Ultralytics Predict Mode](https://docs.ultralytics.com/modes/predict/)
     - [Ultralytics Export Guide](https://docs.ultralytics.com/modes/export/)
     - [Ultralytics Model Validation](https://docs.ultralytics.com/modes/val/)
3. **COCO Detection Evaluation Standard:**
   - *Lin, T. Y., et al. (2014).* "Microsoft COCO: Common Objects in Context." *ECCV 2014*. [COCO Detection Evaluation Resource](https://cocodataset.org/dataset/detection-eval.htm).
   - *Significance:* Defines standard IoU thresholds ($0.50$ to $0.95$ in steps of $0.05$), 101-point interpolated Average Precision (AP), and scale-stratified evaluation ($\text{AP}_S, \text{AP}_M, \text{AP}_L$).

---

## 4. Candidate Approaches and Hardware Requirements

| Candidate | Architecture / Backbone | Hardware Target | Expected mAP50 | Strengths | Weaknesses |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **YOLO26n (Default)** | Lightweight ConvNet backbone + decoupled head | x86 CPU (AVX2/AVX-512) | 40.9–57.5 (COCO) | Minimal memory footprint (~6 MB weights), rapid warm-up | Compute-bound on CPU ($280\text{ ms}$ in raw PyTorch) |
| **YOLO26s (Scaled)** | Deeper feature pyramid + wider channels | GPU (CUDA/TensorRT) | Higher AP (~62.0) | Superior recall on distant/small objects | Prohibitive CPU latency (~600+ ms/frame) |
| **YOLOE-26 (Open-Vocab)** | Text-vision embedding fusion (CLIP-based) | High-end GPU / VRAM | Variable | Zero-shot detection on arbitrary user queries | Massive embedding latency, high operational cost |
| **PP-YOLOE+ (Paddle)** | CSPRepResNet backbone | CPU with OpenVINO / Paddle | High | Strong performance on industrial edge devices | Introduces Paddle inference dependency |

### Hardware Feasibility
- **Local Dev / CI:** 12-core x86 CPU, 16 GB RAM. Must not mandate GPU availability for basic testing or development.
- **Production Server:** NVIDIA T4 / L4 or modern Intel Xeon / AMD EPYC with AMX/AVX-512 acceleration.

---

## 5. Expected Trade-offs and Failure Modes

1. **Precision vs. Recall under Confidence Thresholding:**
   - Setting $\text{conf}=0.25$ generates 803 raw detections on `ppe-1.mp4`. Raising $\text{conf}=0.50$ reduces false positives (e.g., background furniture misclassified as equipment) but risks dropping partially occluded workers during fast motion.
2. **Duplicate Box Credit (Greedy vs Hungarian Matching):**
   - *Failure Mode:* A naive evaluation script awarding true-positive credit to multiple predicted boxes overlapping a single ground-truth object inflates precision and recall.
   - *Mitigation:* Strict 1-to-1 matching: once a prediction is assigned to a ground-truth object with highest IoU, any remaining overlapping predictions for that object are classified as False Positives.
3. **Boundary Truncation & Guideline Misalignment (Policy A):**
   - *Failure Mode:* Ground-truth annotators bounding only the torso/clothing, whereas the detector encompasses the safety hard hat and reaching hands.
   - *Mitigation:* Explicit annotation contracts defining full anatomical bounds (hard hat to boots, fingertips) while excluding handheld tools or detached safety equipment.
4. **Sampling Bias from Dense Video Frames:**
   - Adjacent video frames (e.g. frame 10 and frame 11 @ 30 FPS) are temporally autocorrelated. Evaluating a model on 12 consecutive frames from a single clip yields high variance and zero cross-domain validity. Evaluation requires keyframe sampling (e.g. 1 Hz) across diverse environments.

---

## 6. Proposed Experiment and Acceptance Criteria

### Experiment Design
1. **Annotation Expansion:** Expand the verified 12-keyframe suite on `ppe-1.mp4` to a multi-clip validation set of 10–20 short surveillance/workplace clips (500+ annotated keyframes across variable lighting and zoom levels).
2. **One-to-One Matching Benchmark:** Implement an automated validation harness calculating:
   - Precision, Recall, and F1 at $\text{IoU} \in \{0.50, 0.75\}$
   - $\text{mAP}_{50}$ and $\text{mAP}_{50:95}$ across all annotated classes.
3. **Threshold Sweep:** Evaluate confidence sweeps $\text{conf} \in [0.10, 0.70]$ in increments of $0.05$ to find the optimal F1 operating point for the surveillance domain.

### Acceptance Criteria for Promotion to Accepted
- $\text{mAP}_{50} \ge 0.85$ on target domain surveillance classes (person, helmet/PPE, vest).
- Strict one-to-one matching enforced in code.
- Zero data leakage between evaluation videos and any training/fine-tuning datasets.
- Deterministic inference output with reproducible bounding boxes ($\text{jitter} < 2\text{ px}$ across static frames).

---

## 7. Actual Experiment Results (Kept Separate from Hypotheses)

### Baseline Measurements (Recorded 2026-10-21 / 2026-10-22)
- **Target Clip:** `Sample_Videos/ppe-1.mp4` (11.44s, 343 frames).
- **Detector:** `models/yolo26n.pt` on PyTorch CPU (`threads=12`).
- **Total Detections Generated:** 803 detections (at $\text{conf}=0.25$).
- **Inference Time:** $96.12\text{ s}$ total ($280.2\text{ ms/frame}$ mean).
- **Ground Truth Audit Findings:**
  - Initial 12-keyframe human export had mean IoU of $0.793$ against model proposals.
  - Qualitative visual review confirmed that in 10 of 12 frames, the detector's proposed boxes accurately included the worker's yellow hard hat, which human annotators had clipped.
  - Formal audit trail created in `outputs/annotation_correction_review/audit_trail.json` with status `AWAITING_APPROVAL`.

---

## 8. Decision, Rationale, and Link to Relevant ADR

- **Decision:** Retain **YOLO26n** as the canonical baseline detector wrapped behind `DetectionProvider`.
- **Rationale:** YOLO26n provides the optimal parameter-to-speed ratio for real-time video intelligence. While raw PyTorch CPU inference ($280\text{ ms}$) is compute-bound, the model architecture is directly exportable to graph-optimized runtimes (ONNX Runtime, OpenVINO) without altering pipeline interfaces. Open-vocabulary models (YOLOE-26) are deferred until fixed-vocabulary limitations are demonstrably breached.
- **Relevant ADR:** [ADR-001 — Object Detector Selection](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/decisions/ADR-001-detector.md).
