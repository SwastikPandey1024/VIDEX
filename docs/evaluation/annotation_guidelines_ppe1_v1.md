# VIDEX PPE-1 Ground-Truth Annotation Guidelines & Protocol (Policy A)

> **Specification Identifier:** `annotation_guidelines_ppe1_v1`  
> **Target Video:** `Sample_Videos/ppe-1.mp4` (Duration: 11.445s, 343 frames, 29.97 fps, 1280x720)  
> **Annotation Policy:** **Policy A — Visible Person Only**  
> **Author:** VIDEX Computer Vision Evaluation Team  
> **Status:** ACTIVE (Prediction-Blind Independent Benchmark Gate)  

---

## 1. Objective & Scope

This specification establishes the strict bounding-box annotation standard for `Sample_Videos/ppe-1.mp4`. The objective is to produce an **independent, auditable, prediction-blind ground-truth benchmark** for evaluating detector localization, tracking, and PPE monitoring accuracy in the VIDEX perception pipeline.

---

## 2. Policy A — Bounding-Box Convention

### 2.1 Enclosed Extents
- **Visible Body Parts Only**: Annotate each visible person with exactly one axis-aligned bounding box enclosing all visually distinguishable body parts (head, torso, arms, hands, legs, feet).
- **Physical Boundary**: The box boundaries must fit snugly against the outer extremities of the visible body parts.

### 2.2 Excluded Objects (Strict Negative Constraints)
- **Separate Machinery & Tools**: Do **not** expand the box to include separate machinery, tools, workpieces, tables, or instruments, even when the person is actively holding, touching, or operating them.
- **Reach Envelope & Task Workspace**: Do **not** expand the bounding box to cover the person's functional task zone, reach envelope, or surrounding work bay.
- **Cast Shadows**: Do **not** include cast shadows on the floor or equipment.

### 2.3 Occlusion & Boundary Truncation
- **Partial Occlusion**: When body parts are partially occluded by machinery or foreground objects, box **only the visible portions** of the person. Do not extrapolate occluding objects into the bounding box.
- **Full Occlusion**: If a limb or body part is fully concealed behind machinery or another object, do not estimate or extend the box over the occluded region.
- **Image Frame Boundary Truncation**: When a person extends beyond the image boundaries ($x < 0$, $y < 0$, $x > 1280$, or $y > 720$), clip the bounding box strictly to the frame edge. Do not predict coordinates outside the image frame.

### 2.4 Ambiguity & Edge Cases
- If a hand or limb is difficult to distinguish from machinery or raw materials, annotate the best visually supported boundary, set `is_ambiguous: true`, and document the rationale in `annotation_rationale`.
- If no person is visible in a sampled frame, record an **explicit empty detection list** (`[]`) for that frame. Never omit a sampled frame from the benchmark.

### 2.5 Instance Identity
- When multiple frames capture the same individual across the video sequence, assign the consistent identity `object_identity: "worker_1"`.
- Identity must be based purely on visual continuity in the raw video, never inferred from detector tracking IDs.

---

## 3. Coordinate Convention & Technical Representation

| Attribute | Specification |
| :--- | :--- |
| **Image Resolution** | 1280 pixels width $\times$ 720 pixels height |
| **Coordinate Origin** | Top-left corner `(x=0, y=0)` |
| **Coordinate Axes** | X-axis increases rightward ($0 \to 1280$), Y-axis increases downward ($0 \to 720$) |
| **Box Representation** | `[x, y, width, height]` in pixel units (`coordinate_type: "pixel"`) |
| **Box Boundaries** | Top-left: `(x, y)`, Bottom-right: `(x + width, y + height)` |
| **Edge Clipping** | Coordinates must satisfy: $0 \le x < 1280$, $0 \le y < 720$, $x + \text{width} \le 1280$, $y + \text{height} \le 720$ |
| **Box Dimensions** | Width and height must be strictly positive finite numbers: $\text{width} > 0$, $\text{height} > 0$ |
| **Expected Class Label** | `person` (lowercase standard label) |
| **Rounding Tolerance** | Subpixel coordinates allowed; typical human manual entry precision is $\pm 0.5$ to $\pm 1.0$ pixel. |

---

## 4. Prediction-Blind Integrity Protocol

1. **Zero Model Exposure**: Annotators must inspect only the raw decoded frames (`outputs/evaluation/benchmark_gate_v2/frames/`). No detector overlays, prediction bounding boxes, confidence scores, or tracker IDs may be shown.
2. **Historical Fixture Isolation**: Historical fixtures `ppe1_annotations.json` (v1) and `ppe1_annotations_v2.json` (v2) must remain strictly quarantined in historical fixture paths and must not be used to initialize or prefill benchmark boxes.
3. **Audit Trail**: Every annotated frame must record annotator identity, verification timestamp, and validation status before freezing the benchmark dataset.
