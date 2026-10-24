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
- **Clipping & Bounds:** All coordinates must lie within `[0, image_width]` and `[0, image_height]`. Boxes exceeding image bounds must be clipped to frame dimensions.

### 2.2 Multi-Object Tracking (`GroundTruthTrack`)
- **Trajectory Ordering:** The list of observations must be strictly ordered by non-decreasing timestamp ($t_0 \le t_1 \le \dots \le t_N$).
- **Identity Consistency:** If an object is temporarily occluded and reappears, it retains its original `track_id` unless visual identity is genuinely ambiguous.

### 2.3 On-Screen Text / OCR (`GroundTruthOCR`)
- **Verbatim Text:** Record on-screen text exactly as written, including Hindi/Devanagari characters, numbers, and punctuation.
- **Normalization Policy:** Evaluated under Unicode NFKC normalization and optional case-folding.
- **Legibility Threshold:** Annotate only text that is clearly legible to a human observer at native resolution.
- **Language / Script:** Specify ISO code (e.g. `en`, `hi`, `code`).

### 2.4 Spoken Audio / ASR (`GroundTruthTranscript`)
- **Segment Bounds:** Mark start and end timestamps in seconds (`start_seconds`, `end_seconds`) with `end_seconds >= start_seconds`.
- **Verbatim Transcription:** Record spoken dialogue without editorial rewriting. Filler words and stuttering should be preserved where phonetically distinct.
- **Silent Containers:** If a video has no audio track or contains purely non-human ambient noise (traffic, wind), the modality status must be set to `NOT_APPLICABLE` with an explicit rationale string.
- **Error Metrics:** Word Error Rate (WER) and Character Error Rate (CER) via Levenshtein edit distance. Notice: insertion errors can cause CER/WER to exceed 1.0 ($100\%$).

### 2.5 Temporal Events (`GroundTruthEvent`)
- **Interval Bounds:** Start and end timestamps in seconds defining the duration of the event ($t_{\text{start}} \le t_{\text{end}}$).
- **Physical Grounding:** Each event must link to participant track IDs or evidence IDs that justify the event classification.
- **Event Types:** Standardized vocabulary (e.g., `object_appeared`, `object_stopped`, `object_turned`, `loitering`, `worker_interaction`, `slide_changed`).
- **Temporal IoU (tIoU):** Defined as:
  $$\text{tIoU}(A, B) = \frac{\max(0, \min(A_{\text{end}}, B_{\text{end}}) - \max(A_{\text{start}}, B_{\text{start}}))}{\max(A_{\text{end}}, B_{\text{end}}) - \min(A_{\text{start}}, B_{\text{start}})}$$
  For zero-duration events ($A_{\text{start}} = A_{\text{end}}$ and $B_{\text{start}} = B_{\text{end}}$), $\text{tIoU} = 1.0$ if $|A_{\text{start}} - B_{\text{start}}| \le \text{tolerance}$, else $0.0$.
  Matching threshold default is $\text{tIoU} \ge 0.50$ matching the identical event type.

### 2.6 Question Answering & Abstention (`GroundTruthQA`)
- **Answerability Contract (`is_answerable`):**
  - If `True`: The question is factually answerable from observable video evidence. A verified `reference_answer` must be provided, along with supporting `evidence_ids`.
  - If `False`: The question asks about information completely absent from the video (e.g. asking for a person's name when neither dialogue nor nametag reveals it). The evaluation harness tests whether the system correctly abstains (`did_abstain=True` or returns `"insufficient_evidence"`) rather than hallucinating an answer.
- **Scoring Breakdown:**
  - True Positives (Abstain): Unanswerable question correctly abstained.
  - True Negatives (Answer): Answerable question answered (not abstained).
  - False Positives (False Refusal): Answerable question where system erroneously refused.
  - False Negatives (Hallucination): Unanswerable question where system hallucinated an answer.
  - Missing Predictions: Counted separately and never scored as valid abstentions.

---

## 3. Matching & Evaluation Algorithms

### 3.1 Object Detection Matching Policy
- **Optimal Bipartite Matching (Default):**
  Constructs a bipartite graph between ground-truth detections $G$ and candidate predictions $P$ filtered by exact class match and temporal tolerance $|t_P - t_G| \le \text{timestamp\_tolerance}$ (default $0.5\,\text{s}$). Edge weights represent spatial IoU:
  $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$
  Candidate edges exist only where $\text{IoU} \ge \text{iou\_threshold}$ (default $0.50$).
  The matching solves a maximum-weight bipartite matching via pure-Python Kuhn-Munkres (Hungarian) optimization, ensuring the maximum cardinality of valid matches with maximum aggregate IoU.
- **Greedy Matching Option:**
  Sorts candidate pairs by descending IoU and greedily claims unmatched detections. While faster, greedy assignment can yield suboptimal cardinality in dense or overlapping scenes.
- **Duplicate Penalization:**
  Each prediction and each ground-truth object is matched at most once. Surplus predictions for an already-matched ground-truth object are strictly counted as False Positives.
- **Wrong-Class Overlaps:**
  Predictions overlapping a ground-truth object with a discordant class label cannot match and are recorded as False Positives, while the ground truth remains a False Negative.

### 3.2 Metric Aggregation Standards
- **Micro-Averaged Precision, Recall, and F1:**
  Aggregates raw counts across all evaluated videos first:
  $$\text{Precision}_{\text{micro}} = \frac{\sum_v \text{TP}_v}{\sum_v \text{TP}_v + \sum_v \text{FP}_v}, \quad \text{Recall}_{\text{micro}} = \frac{\sum_v \text{TP}_v}{\sum_v \text{TP}_v + \sum_v \text{FN}_v}$$
  $$F1_{\text{micro}} = \frac{2 \cdot \text{Precision}_{\text{micro}} \cdot \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}$$
- **Macro-Averaged Metrics:**
  The unweighted arithmetic mean of per-video metrics across videos with defined scores. Must always be explicitly labeled as Macro (never confounded with Micro).
- **Zero Denominator Protection:**
  When $\text{TP} + \text{FP} = 0$ or $\text{TP} + \text{FN} = 0$, precision/recall/F1 are returned as `0.0` with explicit flags `zero_denominator=True` and a human-readable `undefined_reason`.

---

## 4. Annotation Audit & Provisional Status Policy

1. **Independent Physical Verification:**
   All ground truth must be verified by directly inspecting video frames (e.g. via OpenCV). No ground truth may be altered or synthesized to fit model performance.
2. **Provisional Marking:**
   If an existing annotation contains discrepancies with physical video reality (such as synthetic color scenes lacking real vehicles or unannotated legible text), the modality status must be updated to `annotation_incomplete` with an explanatory note in `audit_notes` and `rationale`.
3. **Educational / Lecture Benchmark Gate:**
   Surveillance and synthetic test clips must never be labeled or utilized as educational benchmarks. The educational baseline remains strictly blocked until permitted, high-quality lecture recordings with slide transcripts and formulas are verified.

---

## 5. Educational & Lecture Annotation Specifications

When annotating educational, technical, or lecture content:
1. **Topic Boundaries:** Segment lecture clips by major slide changes or conceptual topic shifts.
2. **Formula / Equation Capture:** Record LaTeX representations in `equations_latex` when mathematical expressions are written on slides or whiteboards.
3. **Visual Exhibit Crops:** Record bounding boxes of diagrams, figures, or architecture charts that are necessary to understand the verbal explanation.
4. **Cross-Modal Grounding:** When an instructor refers to *"as shown on the left"*, record the simultaneous visual bounding box and verbal transcript timestamp.
