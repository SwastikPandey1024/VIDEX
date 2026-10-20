# VIDEX Ground-Truth Annotation Guidelines & Protocol

> **Specification Version:** 1.0.0  
> **Target Subsystems:** Detection · Tracking · OCR · ASR · Temporal Events · QA Abstention  
> **Author:** VIDEX Evaluation Team  

---

## 1. Principles of Honest Evaluation

1. **Human-Verified Ground Truth Only:** No model-generated labels may be stored as ground truth.
2. **Explicit Missing Semantics:**
   - `NOT_ANNOTATED`: Ground truth has not yet been supplied.
   - `NOT_APPLICABLE`: The modality does not apply to the video (e.g., ASR on a silent video container).
   - `ANNOTATED`: Ground truth exists, has been verified by a human annotator, and is ready for scoring.
   - `ANNOTATION_INCOMPLETE`: Ground truth is partial; evaluation must acknowledge this limitation.
3. **No Synthetic Zeroes or Perfect Scores:** Unannotated modalities must be reported as `UNAVAILABLE` rather than assigned a 0.0 or 1.0 score.
4. **Separation of Structural Integrity and Semantic Truth:**
   - Reference integrity verifies that a cited ID exists and belongs to the video.
   - Semantic truth requires verifying whether the cited image or text actually supports the factual claim made.

---

## 2. Modality Annotation Standards

### 2.1 Object Detection (`GroundTruthDetection`)
- **Coordinate Space:** Pixel coordinates `(x, y, width, height)` relative to native frame dimensions, with `(0, 0)` at the top-left corner.
- **Timestamping:** Every bounding box must reference an authoritative Presentation Timestamp (`timestamp_pts`) in seconds.
- **Tightness:** Bounding box should enclose the visible boundaries of the object. Partial occlusions (< 80%) are annotated; fully occluded objects are omitted.
- **Class Vocabulary:** Standardized lowercase labels (e.g., `person`, `car`, `bicycle`, `motorcycle`, `hard_hat`, `safety_vest`).

### 2.2 Multi-Object Tracking (`GroundTruthTrack`)
- **Trajectory Ordering:** The list of observations must be strictly ordered by non-decreasing timestamp ($t_0 \le t_1 \le \dots \le t_N$).
- **Identity Consistency:** If an object is temporarily occluded and reappears, it retains its original `track_id` unless visual identity is genuinely ambiguous.

### 2.3 On-Screen Text / OCR (`GroundTruthOCR`)
- **Verbatim Text:** Record on-screen text exactly as written, including Hindi/Devanagari characters, numbers, and punctuation.
- **Legibility Threshold:** Annotate only text that is clearly legible to a human observer at native resolution.
- **Language / Script:** Specify ISO code (e.g. `en`, `hi`, `code`).

### 2.4 Spoken Audio / ASR (`GroundTruthTranscript`)
- **Segment Bounds:** Mark start and end timestamps in seconds (`start_seconds`, `end_seconds`) with `end_seconds >= start_seconds`.
- **Verbatim Transcription:** Record spoken dialogue without editorial rewriting. Filler words and stuttering should be preserved where phonetically distinct.
- **Silent Containers:** If a video has no audio track or contains purely non-human ambient noise (traffic, wind), the modality status must be set to `NOT_APPLICABLE` with an explicit rationale string.

### 2.5 Temporal Events (`GroundTruthEvent`)
- **Interval Bounds:** Start and end timestamps defining the duration of the event.
- **Physical Grounding:** Each event must link to participant track IDs or evidence IDs that justify the event classification.
- **Event Types:** Standardized vocabulary (e.g., `object_appeared`, `object_stopped`, `object_turned`, `loitering`, `worker_interaction`, `slide_changed`).

### 2.6 Question Answering & Abstention (`GroundTruthQA`)
- **Answerability Contract (`is_answerable`):**
  - If `True`: The question is factually answerable from observable video evidence. A verified `reference_answer` must be provided, along with supporting `evidence_ids`.
  - If `False`: The question asks about information completely absent from the video (e.g. asking for a person's name when neither dialogue nor nametag reveals it). The evaluation harness tests whether the system correctly abstains (`did_abstain=True`) rather than hallucinating an answer. No fabricated answer string is required.

---

## 3. Educational & Lecture Annotation Specifications

When annotating educational, technical, or lecture content:
1. **Topic Boundaries:** Segment lecture clips by major slide changes or conceptual topic shifts.
2. **Formula / Equation Capture:** Record LaTeX representations in `equations_latex` when mathematical expressions are written on slides or whiteboards.
3. **Visual Exhibit Crops:** Record bounding boxes of diagrams, figures, or architecture charts that are necessary to understand the verbal explanation.
4. **Cross-Modal Grounding:** When an instructor refers to *"as shown on the left"*, record the simultaneous visual bounding box and verbal transcript timestamp.
