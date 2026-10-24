# VIDEX — PPE-1 Annotation Validation and Benchmark Gate Report

> **Target Asset:** `Sample_Videos/ppe-1.mp4` ($1280 \times 720$, 29.97003 fps, 343 frames, $11.445\,\text{s}$)  
> **Legacy Fixture:** `tests/fixtures/evaluation/ppe1_annotations.json` (v1)  
> **Audited Fixture:** `tests/fixtures/evaluation/ppe1_annotations_v2.json` (v2, provisional candidate)  
> **Evaluator Invariants:** 16/16 Unit Tests Passed  

---

## 1. Visual 4-Panel Evidence Comparisons

Below are the 4-panel side-by-side composite images comparing:
1. **Raw Frame**
2. **v1 Ground Truth** (Red, $[450, 120, 320, 550]$)
3. **v2 Ground Truth** (Green, $[574, 138, 300, 506]$)
4. **YOLO26 Prediction** (Cyan/Yellow, $[575.4, 139.5, 297.9, 504.1]$, conf=0.917)

### Frame 0 ($t = 0.000\,\text{s}$) — 4-Panel Validation
![Frame 0 4-Panel Validation](C:/Users/Swastik Pandey/.gemini/antigravity-ide/brain/ea8fc563-fda2-4427-ad4b-771750eb5d1f/frame_0_4panel_validation.png)

### Frame 90 ($t = 3.003\,\text{s}$) — 4-Panel Validation
![Frame 90 4-Panel Validation](C:/Users/Swastik Pandey/.gemini/antigravity-ide/brain/ea8fc563-fda2-4427-ad4b-771750eb5d1f/frame_90_4panel_validation.png)

---

## 2. Pairwise Spatial IoU Analysis

| Comparison Pair | Frame 0 IoU | Frame 0 Decision ($\theta_{\text{IoU}} = 0.50$) | Frame 90 IoU | Frame 90 Decision ($\theta_{\text{IoU}} = 0.50$) |
| :--- | :---: | :---: | :---: | :---: |
| **v1 GT vs. Model Prediction** | **$0.4300$** | **REJECTED** (1 FN, 1 FP) | **$0.5498$** | **MATCHED** (1 TP) |
| **v2 GT vs. Model Prediction** | **$0.9892$** | **MATCHED** (1 TP) | **$0.9938$** | **MATCHED** (1 TP) |
| **v1 GT vs. v2 GT** | **$0.4287$** | Disjoint reference bounds | **$0.5518$** | Disjoint reference bounds |

---

## 3. Deterministic 1 Hz Sampling Manifest (0.0s to 11.0s)

A deterministic sampling schedule was created extracting 1 frame every second ($1.0\,\text{Hz}$) across the video:

| Sample | Target Time | Decoded Frame | Container PTS | Offset ($\Delta t$) | Status in v1 | Status in v2 | Human Review Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | $0.0\,\text{s}$ | 0 | $0.000000\,\text{s}$ | $0.0\,\text{ms}$ | Annotated | Annotated (Audited) | Verified |
| 1 | $1.0\,\text{s}$ | 30 | $1.001000\,\text{s}$ | $+1.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 2 | $2.0\,\text{s}$ | 60 | $2.002000\,\text{s}$ | $+2.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 3 | $3.0\,\text{s}$ | 90 | $3.003000\,\text{s}$ | $+3.0\,\text{ms}$ | Annotated | Annotated (Audited) | Verified |
| 4 | $4.0\,\text{s}$ | 120 | $4.004000\,\text{s}$ | $+4.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 5 | $5.0\,\text{s}$ | 150 | $5.005000\,\text{s}$ | $+5.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 6 | $6.0\,\text{s}$ | 180 | $6.006000\,\text{s}$ | $+6.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 7 | $7.0\,\text{s}$ | 210 | $7.007000\,\text{s}$ | $+7.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 8 | $8.0\,\text{s}$ | 240 | $8.008000\,\text{s}$ | $+8.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 9 | $9.0\,\text{s}$ | 270 | $9.009000\,\text{s}$ | $+9.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 10 | $10.0\,\text{s}$ | 300 | $10.010000\,\text{s}$ | $+10.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |
| 11 | $11.0\,\text{s}$ | 330 | $11.011000\,\text{s}$ | $+11.0\,\text{ms}$ | *Unannotated* | *Unannotated* | NEEDS_ANNOTATION |

> [!NOTE]
> Frames 1 through 11 (indices 30, 60, 120, ..., 330) are extracted as raw PNGs in `outputs/evaluation/benchmark_gate/sampled_keyframes/`. In strict compliance with VIDEX missing-semantics rules, they are **NOT** treated as negative examples or pseudo-labeled with model outputs.
