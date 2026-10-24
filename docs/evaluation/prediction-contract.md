# VIDEX Evaluation — Prediction Ingestion Contract Specification

## 1. Executive Summary

In evaluation and benchmarking frameworks, confusing **a lack of evaluation input** with **poor model performance** is a critical failure. Running an evaluation CLI against ground-truth annotations without providing valid model predictions must never produce a zero-accuracy benchmark, nor may missing outputs masquerade as successful abstentions.

This document formalizes the VIDEX Prediction Ingestion Contract, specifying:
1. Strict Pydantic v2 schemas for model predictions across perception, OCR, ASR, temporal events, and QA.
2. The five distinct execution states distinguishing harness readiness from model accuracy.
3. Input validation, error handling, and routing rules.

---

## 2. Ingestion Schema Architecture

Model predictions are submitted as JSON documents conforming to `PredictionDataset`:

```python
class PredictionDataset(BaseModel):
    schema_version: str = "1.0.0"
    dataset_id: str
    model_info: dict[str, Any]
    videos: list[VideoPredictions]
```

### 2.1 Per-Video Prediction Schema (`VideoPredictions`)

Each entry in `videos` encapsulates model outputs for a specific video:

```python
class VideoPredictions(BaseModel):
    video_id: str
    detections: list[DetectionPrediction] | None = None
    events: list[EventPrediction] | None = None
    qa: list[QAPrediction] | None = None
    ocr: list[str] | None = None
    transcripts: list[str] | None = None
```

#### Modality Missing vs Empty Semantics:
- `None` (omitted or null): The model did **not** predict this modality (missing output). The evaluation harness marks the modality `MetricStatus.UNAVAILABLE` with an explicit reason (`"Modality '{name}' not supplied in predictions file"`).
- `[]` (empty list): The model explicitly predicted **zero** instances (e.g. model detected 0 objects). This is evaluated against ground truth as a valid observation of zero detections.

---

## 3. Modality-Specific Prediction Models

### 3.1 Object Detection (`DetectionPrediction`)

```python
class DetectionPrediction(BaseModel):
    timestamp_pts: float  # Presentation timestamp in seconds (>= 0.0)
    class_label: str      # Predicted object category (e.g. 'person', 'car')
    bbox: BoundingBox     # x, y, width, height (top-left origin, pixel space)
    confidence: float = 1.0  # Confidence score in [0.0, 1.0]
```

### 3.2 Temporal Events (`EventPrediction`)

```python
class EventPrediction(BaseModel):
    event_type: str       # Predicted canonical event type (e.g. 'loitering')
    start_seconds: float  # Event start timestamp (>= 0.0)
    end_seconds: float    # Event end timestamp (>= start_seconds)
    participants: list[str] = []
    confidence: float = 1.0
```

### 3.3 Question-Answering & Abstention (`QAPrediction`)

```python
class QAPrediction(BaseModel):
    qa_id: str            # Referenced GroundTruthQA qa_id
    predicted_answer: str | None = None  # None if model abstained
    did_abstain: bool     # True if model stated 'insufficient evidence'
    confidence: float = 1.0
```

### 3.4 OCR & ASR Predictions

- **OCR**: List of recognized strings (`list[str]`), or `list[OCRPrediction]`.
- **ASR**: List of transcribed utterances (`list[str]`), or `list[TranscriptPrediction]`.

---

## 4. The Five Execution States

The VIDEX evaluation runner mathematically distinguishes the following five states:

| State | Predictions Provided? | Ground Truth Status | Model Predictions Present? | Resulting Metric Status | Metric Value |
|---|---|---|---|---|---|
| **1. No Prediction File Supplied** | No (`None`) | `ANNOTATED` | N/A | `MetricStatus.UNAVAILABLE` | `None` (notice banner displayed) |
| **2. Explicitly Empty Predictions** | Yes | `ANNOTATED` | Yes (`[]`, 0 items) | `MetricStatus.AVAILABLE` | $0.0000$ (with `zero_denominator=True` on precision) |
| **3. Ground Truth Not Applicable** | Yes or No | `NOT_APPLICABLE` | Any | `MetricStatus.NOT_APPLICABLE` | `None` (annotated rationale recorded) |
| **4. Ground Truth Missing / Incomplete** | Yes or No | `NOT_ANNOTATED` or `ANNOTATION_INCOMPLETE` | Any | `MetricStatus.UNAVAILABLE` | `None` (GT baseline incomplete) |
| **5. Valid Predictions, Zero Matches** | Yes | `ANNOTATED` | Yes ($N > 0$ items, $0$ matched) | `MetricStatus.AVAILABLE` | $0.0000$ (non-zero denominators) |

---

## 5. Controlled Fixture Examples

Controlled test fixtures are provided in `tests/fixtures/evaluation/predictions/`:

1. `all_correct.json`: All ground truth items correctly matched.
2. `one_false_positive.json`: All ground truth items matched, plus 1 spurious prediction.
3. `one_false_negative.json`: 1 ground truth item omitted by model.
4. `duplicate_predictions.json`: 2 predictions targeting 1 GT object; exactly 1 matches, duplicate penalized as FP.
5. `wrong_class_overlapping.json`: Correct spatial box but wrong class label (`truck` vs `car`).
6. `cross_video_predictions.json`: Predictions belonging to another video; namespace isolation enforced.
7. `empty_predictions.json`: Explicitly empty prediction lists (`[]`).
8. `missing_modality_predictions.json`: Modality fields set to `null` / omitted.

---

## 6. CLI Execution

### Running an Annotation Readiness Audit (Without Predictions):
```bash
uv run python -m videx.evaluation.runner --annotations datasets/annotations/sample_corpus_ground_truth.json
```
*Output: Displays ground truth readiness and unavailable metric statuses; displays notice banner stating no accuracy benchmark is claimed.*

### Running an Accuracy Benchmark (With Predictions):
```bash
uv run python -m videx.evaluation.runner \
  --annotations datasets/annotations/sample_corpus_ground_truth.json \
  --predictions path/to/model_predictions.json \
  --output-json results/eval_report.json
```
