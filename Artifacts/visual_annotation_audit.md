# VIDEX Phase 3 — Visual Annotation Audit & Contract Verification

> **Dataset:** `tests/fixtures/evaluation/ppe1_annotations.json` (`video_id="video_ppe1_001"`)  
> **Source Video:** `Sample_Videos/ppe-1.mp4` ($1280 \times 720$ px, 29.97003 fps)  
> **Evaluated Frames:** Frame 0 ($t = 0.000\,\text{s}$) & Frame 90 ($t = 3.003\,\text{s}$)  
> **Detection Model:** YOLO26 Nano (`models/yolo26n.pt`, default conf=0.25, CPU)  

---

## 1. Executive Summary

This visual audit investigates the Frame 0 detection result, where YOLO26 detected the worker with **confidence 0.9171** but only achieved an **IoU of 0.4301** against the ground truth, causing it to fail the $\theta_{\text{IoU}} = 0.50$ matching threshold.

### Core Conclusion
- **Software Contract Status:** **Defect-Free**. The bounding box coordinate system (top-left $x, y, w, h$ in pixel space), Ultralytics model un-letterboxing, OpenCV frame decoding, Kuhn-Munkres optimal bipartite matching, and container timestamp handling are operating exactly as specified.
- **Root Cause:** **Ground-Truth Annotation Defect**. The ground truth annotation in `tests/fixtures/evaluation/ppe1_annotations.json` for Frame 0 is displaced **$\approx 125$ pixels to the left** ($x \in [450, 770]$) of the worker's physical body ($x \in [575, 873]$). It encloses empty background wall on the left and clips the worker's right arm and torso on the right.
- **Model Behavior:** The YOLO26 detector correctly localized the worker's visible physical contours.

---

## 2. Visual Audit Overlays

### Frame 0 ($t = 0.000\,\text{s}$) — Mismatch (IoU 0.4301)
Below is the annotated Frame 0 overlay comparing Ground Truth (Green, $[450, 120, 320, 550]$) against YOLO26 Predictions (Red: `person` conf=0.917 $[575, 140, 298, 504]$, Orange: `suitcase` conf=0.464 $[1071, 642, 129, 77]$).

![Frame 0 Annotation Audit Overlay](C:/Users/Swastik Pandey/.gemini/antigravity-ide/brain/ea8fc563-fda2-4427-ad4b-771750eb5d1f/frame_0_audit.png)

### Frame 0 Zoomed Region Audit
Shaded overlay showing the discrepancy: **Green tint ($x \in [450, 575]$)** is empty background wall enclosed only by GT. **Red tint ($x \in [770, 873]$)** is worker body enclosed only by the detector.

![Frame 0 Zoomed Crop Audit](C:/Users/Swastik Pandey/.gemini/antigravity-ide/brain/ea8fc563-fda2-4427-ad4b-771750eb5d1f/frame_0_crop_audit.png)

### Frame 90 ($t = 3.003\,\text{s}$) — Matched True Positive (IoU 0.5497)
Below is the annotated Frame 90 overlay. As the worker walked leftward toward the static annotation zone, IoU rose to $0.5497$, clearing the $0.50$ gate.

![Frame 90 Annotation Audit Overlay](C:/Users/Swastik Pandey/.gemini/antigravity-ide/brain/ea8fc563-fda2-4427-ad4b-771750eb5d1f/frame_90_audit.png)

---

## 3. Ground Truth vs. Prediction Table

| Dimension / Metric | Frame 0 Ground Truth | Frame 0 YOLO26 Prediction | Frame 90 Ground Truth | Frame 90 YOLO26 Prediction |
| :--- | :---: | :---: | :---: | :---: |
| **Object Class** | `person` | `person` | `person` | `person` |
| **Confidence** | Human Reference | **$0.9171$** | Human Reference | **$0.9022$** |
| **Top-Left $x$** | $450.0$ | **$575.42$** | $460.0$ | **$551.10$** |
| **Top-Left $y$** | $120.0$ | **$139.46$** | $125.0$ | **$133.90$** |
| **Width $w$** | $320.0$ | **$297.90$** | $315.0$ | **$308.96$** |
| **Height $h$** | $550.0$ | **$504.05$** | $545.0$ | **$539.52$** |
| **Horizontal Range $[x_1, x_2]$** | $[450.0, 770.0]$ | **$[575.4, 873.3]$** | $[460.0, 775.0]$ | **$[551.1, 860.1]$** |
| **Vertical Range $[y_1, y_2]$** | $[120.0, 670.0]$ | **$[139.5, 643.5]$** | $[125.0, 670.0]$ | **$[133.9, 673.4]$** |
| **Spatial IoU** | — | **$0.4301$** | — | **$0.5497$** |
| **Match Evaluation ($\theta_{\text{IoU}} = 0.50$)** | **Unmatched (1 FN)** | **Unmatched (1 FP)** | **Matched (1 TP)** | **Matched (1 TP)** |
| **Physical Visual Plausibility** | **IMPLAUSIBLE** (shifted 125 px left onto blank wall) | **ACCURATE** (encloses body) | **MARGINAL** (shifted 91 px left onto blank wall) | **ACCURATE** (encloses body) |

---

## 4. Annotation and Evaluation Contract Verification

- **Coordinate Convention:** Both Ground Truth and YOLO predictions use top-left origin $[x, y, w, h]$ in native pixel space. (Verified in [`schemas.py`](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/src/videx/domain/schemas.py) and [`detection.py`](file:///c:/Users/Swastik%20Pandey/Downloads/VIDEX/src/videx/perception/detection.py)).
- **Width & Height Interpretation:** Width and height are non-negative pixel spans ($w = x_2 - x_1$, $h = y_2 - y_1$), not bottom-right coordinates.
- **Coordinate Scaling & Letterbox:** Ultralytics letterbox transformation maps from $640 \times 640$ model forward pass back to native $1280 \times 720$ pixel space. Verified: Suitcase detection at $x=1070.7$ and person at $x=575.4$ reside in native $1280 \times 720$ coordinate space.
- **Class Matching:** Case-insensitive string matching (`gt.class_label.lower() == pred.class_label.lower()`).
- **Bipartite Matching:** Kuhn-Munkres minimum cost assignment guarantees one-to-one matching, correctly penalizing surplus predictions as False Positives.
- **Timestamp & Frame Alignment:** Frame 0 container timestamp is $0.000\,\text{s}$ (GT: $0.0\,\text{s}$). Frame 90 container timestamp is $3.003\,\text{s}$ (GT: $3.0\,\text{s}$, offset $3\,\text{ms}$, well within tolerance $\Delta t = 0.100\,\text{s}$).
