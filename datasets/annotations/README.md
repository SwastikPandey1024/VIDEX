# VIDEX Gold-Standard Annotation Datasets

This directory contains human-verified ground-truth datasets for evaluating the VIDEX multimodal intelligence pipeline.

---

## Directory Organization

```text
datasets/annotations/
├── README.md                      ← Documentation and dataset catalog
├── schema_version.json            ← Machine-readable schema format specification
├── sample_corpus_ground_truth.json ← Verified annotations for sample videos (sample.mp4, ppe-1.mp4)
└── educational_lecture_slot.json  ← Documented specification for educational lecture evaluation
```

---

## Status Semantics

Every video modality explicitly declares its annotation state:

| Status Code | Definition | Scoring Behavior |
| :--- | :--- | :--- |
| `annotated` | Human ground truth is complete and verified. | Metrics computed with confidence intervals. |
| `not_applicable` | Modality does not exist in source (e.g. silent clip). | Metric marked `NOT_APPLICABLE` (omitted from penalty). |
| `not_annotated` | Ground truth has not yet been authored. | Metric marked `UNAVAILABLE` (never scored as 0.0). |
| `annotation_incomplete` | Partial ground truth exists. | Evaluated with explicit baseline limitation warning. |

---

## Initial Pilot Datasets

### 1. `sample.mp4` (Surveillance Regression Clip)
- **Source:** 320x240, 2.96s, 25 FPS, MPEG-4, Silent.
- **Annotations Provided:**
  - Object Detections: Vehicle bounding boxes.
  - OCR: `NOT_APPLICABLE` (verified zero on-screen text).
  - Audio: `NOT_APPLICABLE` (silent container, no audio track).
  - QA: Evaluates abstention on unanswerable speech queries.

### 2. `ppe-1.mp4` (Workplace Compliance Regression Clip)
- **Source:** 1280x720, 11.41s, 29.97 FPS, H.264, Silent.
- **Annotations Provided:**
  - Object Detections: Worker `person` bounding boxes across key timestamps.
  - Temporal Events: Worker appearance and motion.
  - OCR: `NOT_APPLICABLE` (no signage).
  - Audio: `NOT_APPLICABLE` (silent container).
  - QA: Worker presence and PPE status verification.

### 3. Educational Lecture Slot (`educational_lecture_slot.json`)
- **Status:** **BLOCKED PENDING SOURCE CLIP.**
- **Specification:** Slot allocated for a licensed, CC-BY or public domain university/technical lecture video (slide-heavy with mathematics/diagrams). Ground truth cannot be manufactured without the authoritative media asset.
