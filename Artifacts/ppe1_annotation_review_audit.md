# VIDEX — PPE-1 Candidate Annotation 4-Boundary Audit Report (Policy A)

> **Document Status:** `AWAITING_HUMAN_APPROVAL` (Strict Safety Protocol)  
> **Target Video:** `Sample_Videos/ppe-1.mp4` (1280x720, 29.97 fps, 343 frames)  
> **Keyframe Indices (12):** `[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]`  
> **Governing Policy:** Policy A — Visible Person Only ([`annotation_guidelines_ppe1_v1.md`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/annotation_guidelines_ppe1_v1.md))  
> **Fixture Gate:** `ppe1_annotations_v3_independent.json` **NOT WRITTEN** (Blocked until 12/12 human approvals)  

---

## 1. Executive Summary & Safety Guarantees

1. **Zero Model Exposure & Quarantined Ground Truth:**  
   Review was conducted strictly on raw decoded frames ([`outputs/evaluation/benchmark_gate_v2/frames/`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/evaluation/benchmark_gate_v2/frames/)) without detector predictions, tracker overlays, or confidence scores.
2. **Preservation of Originals & Hashes:**  
   All original fixtures (`v1`, `v2`), human export, and audit trails remain unmodified. Hashes verified 100% byte-for-byte.
3. **No Automatic Batch Approval:**  
   None of the 12 candidate boxes have been approved automatically. Every frame's status is preserved as `AWAITING_APPROVAL`.
4. **Independent 4-Boundary Audit:**  
   Every proposed box was inspected across all 4 boundaries independently ($x_1, y_1, x_2, y_2$), specifically verifying right-side arm/hand coverage and ensuring strict exclusion of unrelated objects (tools, clipboard, pen, countertop, overhead lamp).

---

## 2. File Integrity & SHA-256 Hash Verification

All historical fixtures, human exports, and benchmark files match their recorded SHA-256 hashes exactly:

| Asset Identifier | File Path | Verified SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :--- |
| **Historical v1 Fixture** | [`tests/fixtures/evaluation/ppe1_annotations.json`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/tests/fixtures/evaluation/ppe1_annotations.json) | `edd062d360c42850a55b7de94c5f8febd475683ef0fa53f2d9b5018d27af934b` | **PRESERVED [OK]** |
| **Historical v2 Fixture** | [`tests/fixtures/evaluation/ppe1_annotations_v2.json`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/tests/fixtures/evaluation/ppe1_annotations_v2.json) | `8fcd58897e00b500e73a89e5ab7e91b2022fc3efbc3044687385e44ab526b4b7` | **PRESERVED [OK]** |
| **Human Annotator Export** | [`outputs/evaluation/benchmark_gate_v2/ppe1_independent_annotations_completed.json`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/evaluation/benchmark_gate_v2/ppe1_independent_annotations_completed.json) | `60d75d0b403ca8f289790450dbc7f472968eaa11c3d2eaab2efbf0da4d4a02e1` | **PRESERVED [OK]** |
| **Pending v3 Fixture** | `tests/fixtures/evaluation/ppe1_annotations_v3_independent.json` | *None (File intentionally withheld per Rule 7)* | **LOCKED [OK]** |

---

## 3. Detailed Frame-by-Frame 4-Boundary Audit

### Frame Summary Table

| Frame | PTS (s) | Submitted Box `[x,y,w,h]` | Proposed Box `[x,y,w,h]` | IoU | Boundary Audit Summary | Review Status |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **0** | 0.000 | `[562, 143, 334, 517]` | `[560, 105, 300, 550]` | 0.824 | **Top:** Hard hat apex captured at $y=105$; lamp excluded ($y<95$). **Right:** $x_2=860$ trims 36px empty wall from submitted $x_2=896$. *Refinement note:* Outer sleeve contour extends to $x=868$ (candidate option `[560, 105, 308, 550]`). | `AWAITING_APPROVAL` |
| **30** | 1.001 | `[534, 153, 312, 458]` | `[520, 90, 330, 555]` | 0.780 | **Top:** Full hard hat apex at $y=90$ (submitted clipped 63px). **Left:** Covers extended right forearm holding tape at $x=520$. **Bottom:** Encloses hands ($y_2=645$) above tape tool/clipboard. | `AWAITING_APPROVAL` |
| **60** | 2.002 | `[568, 150, 226, 501]` | `[475, 90, 365, 580]` | 0.535 | **Left & Top:** Reaching overhead left arm/hand completely omitted in submitted box ($x_1=568\to 475$, $y_1=150\to 90$). Stops at hand, excluding lamp fixture ($y<90$). **Right:** $x_2=840$ recovers clipped shoulder. | `AWAITING_APPROVAL` |
| **90** | 3.003 | `[549, 169, 243, 491]` | `[490, 88, 360, 585]` | 0.567 | **Left & Top:** Reaching arm enclosed at $x_1=490, y_1=88$; submitted truncated hard hat by 81px. **Right:** $x_2=850$ encloses right shoulder. **Bottom:** Countertop occlusion at $y_2=673$. | `AWAITING_APPROVAL` |
| **120** | 4.004 | `[545, 154, 258, 511]` | `[495, 88, 355, 585]` | 0.635 | **Left & Top:** Encloses raised left arm/hand at $x_1=495, y_1=88$ without enclosing lamp. **Right:** Fits right upper arm at $x_2=850$. | `AWAITING_APPROVAL` |
| **150** | 5.005 | `[581, 195, 202, 396]` | `[500, 85, 345, 585]` | 0.396 | **Severe Submitted Error:** Submitted truncated 110px of hard hat and completely clipped overhead hand. Proposed recovers $x_1=500, y_1=85, x_2=845, y_2=670$. | `AWAITING_APPROVAL` |
| **180** | 6.006 | `[508, 227, 300, 408]` | `[505, 118, 315, 550]` | 0.706 | **Top:** Leaning hard hat apex enclosed at $y_1=118$ (submitted clipped 109px). **Left:** Leaning left forearm enclosed at $x_1=505$. **Right:** Right arm at $x_2=820$. | `AWAITING_APPROVAL` |
| **210** | 7.007 | `[604, 264, 253, 353]` | `[510, 150, 355, 520]` | 0.484 | **Left:** Submitted box ($x_1=604$) completely omitted the leaning left arm extending to $x_1=510$. **Top:** Hard hat apex at $y_1=150$. **Right:** Back/hips at $x_2=865$. | `AWAITING_APPROVAL` |
| **240** | 8.008 | `[645, 305, 229, 359]` | `[520, 178, 355, 500]` | 0.463 | **Severe Submitted Error:** Submitted omitted 127px of hard hat ($y_1=305\to 178$) and omitted leaning writing arm ($x_1=645\to 520$). **Right:** Back at $x_2=875$ excluding wall. | `AWAITING_APPROVAL` |
| **270** | 9.009 | `[635, 237, 245, 433]` | `[520, 192, 360, 490]` | 0.601 | **Left:** Leaning writing arm recovered ($x_1=520$). **Top:** Hard hat apex at $y_1=192$. **Right:** $x_2=880$ snugly fits back without wall overextension. | `AWAITING_APPROVAL` |
| **300** | 10.010 | `[644, 268, 220, 388]` | `[520, 195, 355, 485]` | 0.496 | **Left:** Writing forearm/hand enclosed at $x_1=520$. **Top:** Hard hat at $y_1=195$. **Bottom:** Hands/countertop at $y_2=680$. | `AWAITING_APPROVAL` |
| **330** | 11.011 | `[642, 274, 228, 409]` | `[520, 202, 355, 485]` | 0.542 | **Left:** Writing arm enclosed at $x_1=520$. **Top:** Hard hat apex at $y_1=202$. **Bottom:** Occlusion edge at $y_2=687$. | `AWAITING_APPROVAL` |

---

## 4. Specific Verification Findings (Right-Side & Unrelated Objects)

1. **Overhead Lamp Fixture Exclusion ($y < 90$):**  
   In keyframes 0 through 150, a black overhead lamp fixture is mounted directly above the worker's workstation ($y \in [47, 124]$, $x \in [448, 627]$). In the reach keyframes (60, 90, 120, 150), the worker reaches overhead to touch or adjust the fixture. The proposed candidate boundary $y_1 \in [85, 90]$ accurately terminates at the knuckles/fingertips, strictly excluding the black overhead lamp armature above.
2. **Clipboard, Paper, and Workpiece Exclusion:**  
   In keyframes 180 through 330, the worker leans forward over the inspection bench to write on a clipboard. The bright white paper is located at $x \in [530, 600]$, $y \in [580, 670]$. The worker's hands hold the clipboard and pen. The proposed box encompasses the hands ($x_1=520, y_2=678\text{--}687$), while strictly excluding the separate table surface and extraneous tabletop items to the left and right.
3. **Right-Side Boundary Coverage ($x_2$):**  
   - In Frames 180–330, the worker leans forward and to the left; the right extremity is the worker's back and rear hips ($x_2 \in [820, 880]$). Beyond $x_2$, pixel values immediately transition to the uniform background wall (mean BGR `[185, 175, 170]`).
   - In Frame 0, the submitted box extended to $x_2=896$, enclosing 36px of empty background wall. The proposed box sets $x_2=860$. High-resolution pixel scan confirms the outermost sleeve contour extends to $x=868$. A refined candidate proposal `[560, 105, 308, 550]` ($x_2=868$) is recorded in the audit trail for human sign-off.

---

## 5. Structural Validation & Test Suite Status

The annotation schema validator ([`src/videx/evaluation/annotation_validator.py`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/src/videx/evaluation/annotation_validator.py)) and integrity tests ([`tests/unit/test_benchmark_integrity.py`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/tests/unit/test_benchmark_integrity.py)) were executed:
- **39/39 unit tests passed** in 1.06s.
- `corrections_applied_to_fixture == False` confirmed.
- `corrections_approved_by_reviewer == False` confirmed.
- All 12 frame statuses confirmed as `AWAITING_APPROVAL`.

---

## 6. Review Artifact Paths

- **Interactive Visual Review HTML:** [`outputs/annotation_correction_review/correction_review.html`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/annotation_correction_review/correction_review.html)
- **Machine-Readable Audit Trail:** [`outputs/annotation_correction_review/audit_trail.json`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/annotation_correction_review/audit_trail.json)
- **12 Review Images (3-Panel):** [`outputs/annotation_correction_review/review_f0000.png`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/annotation_correction_review/review_f0000.png) through [`review_f0330.png`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/annotation_correction_review/review_f0330.png)
- **Boundary Audits & Zoom Crops:** [`outputs/annotation_correction_review/boundary_audits/`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/annotation_correction_review/boundary_audits/) and [`boundary_zooms/`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/outputs/annotation_correction_review/boundary_zooms/)
- **Policy A Guidelines Specification:** [`docs/evaluation/annotation_guidelines_ppe1_v1.md`](file:///C:/Users/Swastik%20Pandey/Downloads/VIDEX/docs/evaluation/annotation_guidelines_ppe1_v1.md)
