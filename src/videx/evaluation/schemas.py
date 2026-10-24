"""VIDEX Evaluation and Ground-Truth Domain Schemas.

Defines strict, typed Pydantic v2 models for:
- Gold-standard video annotations across modalities (detections, tracks, OCR, ASR, events, QA)
- Explicit annotation completeness and applicability semantics
- Evaluation results with statistical provenance and uncertainty reporting
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from videx.domain.schemas import BoundingBox


class AnnotationStatus(StrEnum):
    """Explicit annotation state for a video modality."""

    NOT_ANNOTATED = "not_annotated"
    """Ground truth has not yet been supplied for this modality."""

    NOT_APPLICABLE = "not_applicable"
    """The modality does not apply to this video (e.g. ASR on a silent video)."""

    ANNOTATED = "annotated"
    """Ground truth is complete, verified, and ready for evaluation."""

    ANNOTATION_INCOMPLETE = "annotation_incomplete"
    """Partial ground truth exists; evaluation must acknowledge incomplete baseline."""


class MetricStatus(StrEnum):
    """Execution status of an individual evaluation metric."""

    AVAILABLE = "available"
    """Metric was successfully computed against verified ground truth."""

    UNAVAILABLE = "unavailable"
    """Metric could not be computed (e.g., missing ground truth or predictions)."""

    NOT_APPLICABLE = "not_applicable"
    """Metric is intentionally excluded because the domain does not apply."""


T = TypeVar("T")


class ModalityAnnotation(BaseModel, Generic[T]):
    """Container for modality-specific ground-truth items with explicit status."""

    model_config = ConfigDict(frozen=True)

    status: AnnotationStatus = Field(
        default=AnnotationStatus.NOT_ANNOTATED,
        description="Annotation readiness status for this modality",
    )
    items: list[T] = Field(
        default_factory=list,
        description="List of ground-truth annotations for this modality",
    )
    rationale: str = Field(
        default="",
        description="Justification or context (e.g. why a modality is not applicable)",
    )

    @model_validator(mode="after")
    def validate_status_consistency(self) -> ModalityAnnotation[T]:
        if self.status in (AnnotationStatus.NOT_ANNOTATED, AnnotationStatus.NOT_APPLICABLE):
            if self.items:
                raise ValueError(f"Modality items must be empty when status is {self.status.value}")
        elif self.status == AnnotationStatus.ANNOTATED:
            # Note: For some videos, an annotated modality may have zero items
            # (e.g., verified zero speech, or verified zero objects). Rationale is encouraged.
            pass
        return self


# ── Ground-Truth Modality Schemas ─────────────────────────────────────────


class GroundTruthDetection(BaseModel):
    """Gold-standard bounding box observation at a specific timestamp or frame."""

    model_config = ConfigDict(frozen=True)

    detection_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the ground-truth detection",
    )
    timestamp_pts: float = Field(
        ...,
        ge=0.0,
        description="Authoritative presentation timestamp in seconds (must be non-negative)",
    )
    class_label: str = Field(
        ...,
        min_length=1,
        description="Exact annotated object class label (e.g., 'person', 'car')",
    )
    bbox: BoundingBox = Field(
        ...,
        description="Spatial coordinates of the object bounding box",
    )
    frame_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional decoded frame index corresponding to timestamp_pts",
    )
    object_identity: str | None = Field(
        default=None,
        description="Optional object identity or track reference (e.g. 'cyclist_1')",
    )


class GroundTruthTrack(BaseModel):
    """Gold-standard identity trajectory spanning multiple frame observations."""

    model_config = ConfigDict(frozen=True)

    track_id: str = Field(
        ...,
        min_length=1,
        description="Authoritative unique track identifier within the video",
    )
    object_identity: str = Field(
        ...,
        min_length=1,
        description="Human-readable entity name or identity code",
    )
    class_label: str = Field(
        ...,
        min_length=1,
        description="Object category for the track",
    )
    observations: list[GroundTruthDetection] = Field(
        ...,
        min_length=1,
        description="Chronologically ordered detections belonging to this track",
    )

    @model_validator(mode="after")
    def validate_temporal_ordering(self) -> GroundTruthTrack:
        for i in range(len(self.observations) - 1):
            curr_pts = self.observations[i].timestamp_pts
            next_pts = self.observations[i + 1].timestamp_pts
            if next_pts < curr_pts:
                raise ValueError(
                    f"Track observations must be monotonically ordered by timestamp. "
                    f"Observation at index {i + 1} ({next_pts}s) precedes index {i} ({curr_pts}s)."
                )
        return self


class GroundTruthOCR(BaseModel):
    """Gold-standard text observation visible on screen."""

    model_config = ConfigDict(frozen=True)

    ocr_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the ground-truth text observation",
    )
    timestamp_pts: float = Field(
        ...,
        ge=0.0,
        description="Timestamp in seconds when text is prominently legible",
    )
    expected_text: str = Field(
        ...,
        min_length=1,
        description="Exact expected text string (including script/punctuation as written)",
    )
    bbox: BoundingBox | None = Field(
        default=None,
        description="Optional bounding box locating the text region",
    )
    language: str = Field(
        default="en",
        description="Language code or script (e.g. 'en', 'hi', 'code')",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether matching should enforce strict casing",
    )


class GroundTruthTranscript(BaseModel):
    """Gold-standard spoken audio segment with verified start and end bounds."""

    model_config = ConfigDict(frozen=True)

    transcript_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the ground-truth transcript utterance",
    )
    start_seconds: float = Field(
        ...,
        ge=0.0,
        description="Spoken utterance start timestamp in seconds",
    )
    end_seconds: float = Field(
        ...,
        ge=0.0,
        description="Spoken utterance end timestamp in seconds",
    )
    reference_text: str = Field(
        ...,
        min_length=1,
        description="Verbatim spoken dialogue / reference text",
    )
    language: str = Field(
        default="en",
        description="Spoken language code (e.g. 'en', 'hi')",
    )

    @model_validator(mode="after")
    def validate_interval(self) -> GroundTruthTranscript:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"Transcript end_seconds ({self.end_seconds}s) cannot precede "
                f"start_seconds ({self.start_seconds}s)"
            )
        return self


class GroundTruthEvent(BaseModel):
    """Gold-standard temporal event interval with participants and provenance."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the ground-truth event",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Canonical event type (e.g. 'object_turned', 'loitering', 'slide_changed')",
    )
    start_seconds: float = Field(
        ...,
        ge=0.0,
        description="Event start timestamp in seconds",
    )
    end_seconds: float = Field(
        ...,
        ge=0.0,
        description="Event end timestamp in seconds",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="Referenced object identities or track IDs involved in event",
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of supporting detections, OCR, or transcript references",
    )
    evidence_description: str = Field(
        default="",
        description="Human-readable description of observable physical evidence",
    )

    @model_validator(mode="after")
    def validate_interval(self) -> GroundTruthEvent:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"Event end_seconds ({self.end_seconds}s) cannot precede "
                f"start_seconds ({self.start_seconds}s)"
            )
        return self


class GroundTruthQA(BaseModel):
    """Gold-standard question-answering evaluation pair.

    Explicitly models unanswerable queries: when is_answerable is False,
    no synthetic answer is required.
    """

    model_config = ConfigDict(frozen=True)

    qa_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the QA evaluation item",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Natural language question evaluated against the video",
    )
    is_answerable: bool = Field(
        ...,
        description="True if video contains sufficient evidence to answer; False otherwise",
    )
    reference_answer: str | None = Field(
        default=None,
        description="Authoritative reference answer string (required if is_answerable=True)",
    )
    acceptable_answers: list[str] = Field(
        default_factory=list,
        description="Synonymous or alternative acceptable answer phrases",
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs in the video that substantiate the answer",
    )
    annotation_rationale: str = Field(
        default="",
        description="Explanation of why this question is or is not answerable from evidence",
    )

    @model_validator(mode="after")
    def validate_answerability_contract(self) -> GroundTruthQA:
        if self.is_answerable:
            if not self.reference_answer and not self.acceptable_answers:
                raise ValueError(
                    "An answerable question (is_answerable=True) must provide a reference_answer "
                    "or at least one acceptable_answer."
                )
        else:
            # Unanswerable question: reference answer is not mandatory, but if provided,
            # it should document why evidence is insufficient.
            pass
        return self


# ── Video & Dataset Annotation Containers ─────────────────────────────────


class VideoAnnotation(BaseModel):
    """Complete ground-truth annotation record for a single video."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(
        ...,
        min_length=1,
        description="Unique video identifier matching canonical repository records",
    )
    source_identifier: str = Field(
        ...,
        min_length=1,
        description="Original source filename or URL (e.g. 'sample.mp4')",
    )
    duration_seconds: float = Field(
        ...,
        gt=0.0,
        description="Total duration of the video in seconds",
    )
    overall_status: AnnotationStatus = Field(
        default=AnnotationStatus.ANNOTATED,
        description="Overall annotation status for the video",
    )
    detections: ModalityAnnotation[GroundTruthDetection] = Field(
        default_factory=lambda: ModalityAnnotation[GroundTruthDetection](),
        description="Ground-truth object bounding boxes",
    )
    tracks: ModalityAnnotation[GroundTruthTrack] = Field(
        default_factory=lambda: ModalityAnnotation[GroundTruthTrack](),
        description="Ground-truth object identity tracks",
    )
    ocr: ModalityAnnotation[GroundTruthOCR] = Field(
        default_factory=lambda: ModalityAnnotation[GroundTruthOCR](),
        description="Ground-truth on-screen text observations",
    )
    transcripts: ModalityAnnotation[GroundTruthTranscript] = Field(
        default_factory=lambda: ModalityAnnotation[GroundTruthTranscript](),
        description="Ground-truth spoken utterances",
    )
    events: ModalityAnnotation[GroundTruthEvent] = Field(
        default_factory=lambda: ModalityAnnotation[GroundTruthEvent](),
        description="Ground-truth spatiotemporal events",
    )
    qa: ModalityAnnotation[GroundTruthQA] = Field(
        default_factory=lambda: ModalityAnnotation[GroundTruthQA](),
        description="Ground-truth question-answering evaluation pairs",
    )

    @model_validator(mode="after")
    def validate_timestamps_within_duration(self) -> VideoAnnotation:
        tolerance = 0.5  # 500ms container boundary tolerance
        max_allowed = self.duration_seconds + tolerance

        for det in self.detections.items:
            if det.timestamp_pts > max_allowed:
                raise ValueError(
                    f"Detection timestamp ({det.timestamp_pts}s) exceeds video duration "
                    f"({self.duration_seconds}s) + tolerance ({tolerance}s)"
                )
        for tr in self.transcripts.items:
            if tr.end_seconds > max_allowed:
                raise ValueError(
                    f"Transcript end timestamp ({tr.end_seconds}s) exceeds video duration "
                    f"({self.duration_seconds}s) + tolerance ({tolerance}s)"
                )
        for ev in self.events.items:
            if ev.end_seconds > max_allowed:
                raise ValueError(
                    f"Event end timestamp ({ev.end_seconds}s) exceeds video duration "
                    f"({self.duration_seconds}s) + tolerance ({tolerance}s)"
                )
        return self


class AnnotationDataset(BaseModel):
    """Root dataset container encompassing ground-truth annotations across multiple videos."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the annotation schema format",
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the annotation dataset release",
    )
    provenance: str = Field(
        ...,
        min_length=1,
        description="Dataset provenance, annotator attribution, and creation date",
    )
    annotation_guidelines: str = Field(
        default="",
        description="Summary or URI of the annotation specification rules used",
    )
    videos: list[VideoAnnotation] = Field(
        default_factory=list,
        description="Collection of video annotations in this dataset release",
    )

    @model_validator(mode="after")
    def validate_unique_video_ids(self) -> AnnotationDataset:
        seen_ids: set[str] = set()
        for v in self.videos:
            if v.video_id in seen_ids:
                raise ValueError(f"Duplicate video_id '{v.video_id}' found in dataset")
            seen_ids.add(v.video_id)
        return self


# ── Prediction Input Contract Schemas ─────────────────────────────────────


class DetectionPrediction(BaseModel):
    """Predicted bounding box observation at a specific timestamp."""

    model_config = ConfigDict(frozen=True)

    timestamp_pts: float = Field(..., ge=0.0, description="Presentation timestamp in seconds")
    class_label: str = Field(..., min_length=1, description="Predicted object class label")
    bbox: BoundingBox = Field(..., description="Predicted bounding box coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")


class EventPrediction(BaseModel):
    """Predicted temporal event interval."""

    model_config = ConfigDict(frozen=True)

    event_type: str = Field(..., min_length=1, description="Predicted event type")
    start_seconds: float = Field(..., ge=0.0, description="Predicted event start time in seconds")
    end_seconds: float = Field(..., ge=0.0, description="Predicted event end time in seconds")
    participants: list[str] = Field(
        default_factory=list, description="Predicted participating entities"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Event confidence score")

    @model_validator(mode="after")
    def validate_interval(self) -> EventPrediction:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"EventPrediction end_seconds ({self.end_seconds}s) cannot precede "
                f"start_seconds ({self.start_seconds}s)"
            )
        return self


class QAPrediction(BaseModel):
    """Predicted answer or explicit abstention for a question."""

    model_config = ConfigDict(frozen=True)

    qa_id: str = Field(..., min_length=1, description="Question ID being answered")
    predicted_answer: str | None = Field(
        default=None,
        description="Generated answer string; None if the model explicitly abstained",
    )
    did_abstain: bool = Field(
        ...,
        description="True if model reported 'insufficient evidence'; False if answered",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Prediction confidence score"
    )


class OCRPrediction(BaseModel):
    """Predicted text observation extracted via OCR."""

    model_config = ConfigDict(frozen=True)

    timestamp_pts: float = Field(
        default=0.0, ge=0.0, description="Observation timestamp in seconds"
    )
    text: str = Field(..., description="Recognized text string")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="OCR confidence score")


class TranscriptPrediction(BaseModel):
    """Predicted spoken utterance transcribed via ASR."""

    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(default=0.0, ge=0.0, description="Utterance start timestamp")
    end_seconds: float = Field(default=0.0, ge=0.0, description="Utterance end timestamp")
    text: str = Field(..., description="Transcribed utterance text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="ASR confidence score")


class VideoPredictions(BaseModel):
    """Container for modality-specific model predictions for a single video.

    A field set to None indicates that the model did not predict that modality (missing output).
    An explicitly empty list [] indicates that the model predicted zero instances.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., min_length=1, description="Target video identifier")
    detections: list[DetectionPrediction] | None = Field(
        default=None,
        description="Object detections; None if not predicted, [] if zero objects predicted",
    )
    events: list[EventPrediction] | None = Field(
        default=None,
        description="Temporal events; None if not predicted, [] if zero events predicted",
    )
    qa: list[QAPrediction] | None = Field(
        default=None,
        description="QA answers; None if not predicted, [] if zero answers predicted",
    )
    ocr: list[str] | None = Field(
        default=None,
        description="OCR text strings; None if not predicted, [] if zero text predicted",
    )
    transcripts: list[str] | None = Field(
        default=None,
        description="ASR transcripts; None if not predicted, [] if zero speech predicted",
    )


class PredictionDataset(BaseModel):
    """Strict prediction dataset container representing a tested model run."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default="1.0.0", description="Prediction schema version")
    dataset_id: str = Field(default="predictions_run", description="Prediction batch identifier")
    model_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata describing the model or pipeline that generated predictions",
    )
    videos: list[VideoPredictions] = Field(
        default_factory=list,
        description="Predictions grouped by video_id",
    )

    def get_video_predictions(self, video_id: str) -> VideoPredictions | None:
        """Finds predictions for a specific video_id, if present."""
        for v in self.videos:
            if v.video_id == video_id:
                return v
        return None


# ── Metric & Evaluation Output Models ─────────────────────────────────────


class MetricResult(BaseModel):
    """Measured metric value accompanied by statistical uncertainty and sample size."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Canonical metric identifier")
    status: MetricStatus = Field(
        default=MetricStatus.AVAILABLE,
        description="Metric calculation status",
    )
    value: float | None = Field(
        default=None,
        description="Observed point estimate (e.g. precision, recall, error rate)",
    )
    sample_size: int = Field(
        default=0,
        ge=0,
        description="Number of annotated sample units (N) evaluated",
    )
    confidence_interval: tuple[float, float] | None = Field(
        default=None,
        description="Documented interval bounds (e.g., Wilson 95% CI) [lower, upper]",
    )
    ci_method: str | None = Field(
        default=None,
        description="Method used for confidence interval (e.g. 'wilson_score', 'bootstrap')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic breakdown (e.g. TP, FP, FN counts, Levenshtein edit totals)",
    )
    reason_if_unavailable: str | None = Field(
        default=None,
        description="Documented rationale when status is UNAVAILABLE or NOT_APPLICABLE",
    )


class VideoEvaluationSummary(BaseModel):
    """Per-video evaluation breakdown ensuring strong videos do not mask failures."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="Unique video identifier")
    source_identifier: str = Field(..., description="Source filename or URI")
    metrics: dict[str, MetricResult] = Field(
        default_factory=dict,
        description="Mapping from metric name to MetricResult for this video",
    )
    unavailable_modalities: list[str] = Field(
        default_factory=list,
        description="Modalities with unsupplied or incomplete ground truth",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any processing errors or anomalies encountered during evaluation",
    )


class EvaluationResult(BaseModel):
    """Canonical machine-readable evaluation report."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this evaluation run",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC execution timestamp",
    )
    repo_commit: str | None = Field(
        default=None,
        description="Git commit hash of the tested repository state",
    )
    dataset_version: str = Field(
        default="1.0.0",
        description="Annotation dataset version evaluated against",
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation runner parameters and thresholds used",
    )
    provider_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Model versions and execution mode (real / mock / unavailable)",
    )
    aggregate_metrics: dict[str, MetricResult] = Field(
        default_factory=dict,
        description="Corpus-wide aggregate metrics with sample size and uncertainty",
    )
    per_video_results: list[VideoEvaluationSummary] = Field(
        default_factory=list,
        description="Per-video metric breakdown",
    )
    failures: list[str] = Field(
        default_factory=list,
        description="Corpus-level execution errors or validation failures",
    )
