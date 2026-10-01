"""VIDEX typed domain schemas for semantic routing, evidence bundles, and VLM outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from videx.domain.schemas import BoundingBox, FrameTimestamp
from videx.events.schemas import EventParticipant
from videx.semantic.types import (
    RoutingDecision,
    RoutingPriority,
    SemanticEventType,
    SemanticStatus,
)


class CropRegion(BaseModel):
    """A bounded spatial crop derived from a video frame for VLM inspection.

    Treated as a derived artifact referencing canonical evidence, not stored
    as permanent canonical evidence itself unless explicitly required.
    """

    model_config = ConfigDict(frozen=True)

    crop_id: UUID = Field(default_factory=uuid4, description="Unique crop identifier")
    frame_index: int = Field(..., ge=0, description="Source video frame sequence index")
    timestamp_seconds: float = Field(
        ..., ge=0.0, description="Authoritative Presentation Timestamp (PTS)"
    )
    original_bbox: BoundingBox = Field(
        ..., description="Original raw bounding box from detector/tracker"
    )
    clamped_bbox: BoundingBox = Field(
        ..., description="Expanded context box clamped to video frame boundaries"
    )
    context_margin: float = Field(
        default=0.15,
        ge=0.0,
        description="Fractional margin added to bbox dimensions for visual context",
    )
    source_evidence_id: UUID = Field(
        ..., description="UUID of the underlying canonical Evidence record"
    )
    image_bytes: bytes | None = Field(
        default=None,
        description="Optional encoded JPEG/PNG image bytes of the cropped patch",
        repr=False,
    )


class CandidateEvent(BaseModel):
    """A spatiotemporally bounded event candidate nominated for semantic reasoning.

    Formed deterministically from one or more supporting temporal events,
    incorporating saliency evaluation and optional query-relevance scoring.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: UUID = Field(default_factory=uuid4, description="Unique candidate ID")
    video_id: UUID = Field(..., description="Parent video identifier")
    source_event_ids: list[UUID] = Field(
        ...,
        min_length=1,
        description="Underlying temporal Event IDs that triggered this candidate",
    )
    start_timestamp: float = Field(
        ..., ge=0.0, description="Start presentation timestamp in seconds"
    )
    end_timestamp: float = Field(..., ge=0.0, description="End presentation timestamp in seconds")
    participant_ids: list[str] = Field(
        default_factory=list,
        description="String identifiers of participants involved (tracks, zones, speakers)",
    )
    saliency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Deterministic saliency score [0.0, 1.0]"
    )
    query_relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Deterministic query relevance score [0.0, 1.0]",
    )
    reason: str = Field(
        ..., description="Deterministic explainable explanation for why candidate was selected"
    )
    priority: RoutingPriority = Field(
        default=RoutingPriority.MEDIUM,
        description="Evaluation priority level",
    )
    evidence_ids: list[UUID] = Field(
        default_factory=list,
        description="Canonical evidence IDs underpinning the candidate events",
    )

    @property
    def duration_seconds(self) -> float:
        """Temporal duration of this candidate."""
        return max(0.0, self.end_timestamp - self.start_timestamp)


class KeyframeMetadata(BaseModel):
    """Temporal and index provenance for a selected representative video frame."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0)
    role: str = Field(..., description="'leading', 'midpoint', 'trailing', or 'instantaneous'")
    frame_timestamp: FrameTimestamp | None = Field(default=None)


class EvidenceBundle(BaseModel):
    """Self-contained, minimal evidence payload packaged for VLM reasoning.

    Contains only the bounded context needed for visual-linguistic reasoning,
    strictly preserving evidence provenance without duplicating the full video.
    """

    bundle_id: UUID = Field(default_factory=uuid4, description="Unique bundle identifier")
    candidate_id: UUID = Field(..., description="Associated CandidateEvent ID")
    video_id: UUID = Field(..., description="Parent video identifier")
    start_timestamp_seconds: float = Field(..., ge=0.0)
    end_timestamp_seconds: float = Field(..., ge=0.0)

    # Visual Evidence
    keyframes: list[KeyframeMetadata] = Field(
        default_factory=list,
        description="Selected keyframes providing temporal scene framing",
    )
    crops: list[CropRegion] = Field(
        default_factory=list,
        description="Targeted bounding-box crops of key participants",
    )

    # Non-Visual Context
    trajectory_summary: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Kinematic and displacement vector summaries of tracked entities",
    )
    ocr_observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Text strings recognized within the candidate time window",
    )
    transcript_segments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Spoken speech transcript segments active during the interval",
    )
    spatial_context: list[str] = Field(
        default_factory=list,
        description="Named spatial zones active or intersected",
    )

    # Provenance
    supporting_event_ids: list[UUID] = Field(
        default_factory=list,
        description="Temporal Event IDs included in this bundle",
    )
    evidence_ids: list[UUID] = Field(
        default_factory=list,
        description="All canonical Evidence IDs supplied in this bundle",
    )
    provenance_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Subsystem versions, detector configurations, and bundle metrics",
    )

    @property
    def total_visual_crops(self) -> int:
        """Total number of visual crops in this bundle."""
        return len(self.crops)


class SemanticEventPayload(BaseModel):
    """Structured semantic inference result produced by a VLM reasoning provider.

    Must explicitly cite supplied evidence_ids and support clean abstention
    when evidence is ambiguous or incomplete.
    """

    semantic_event_type: SemanticEventType = Field(
        default=SemanticEventType.GENERAL_ACTIVITY,
        description="High-level semantic category",
    )
    claim: str = Field(
        ...,
        description="Concise semantic factual claim (e.g. 'Security inspected vehicle')",
    )
    description: str = Field(
        ...,
        description="Detailed contextual narrative grounded in observed visual and audio cues",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0, 1.0] in the semantic interpretation",
    )
    uncertainty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Explicit uncertainty estimate [0.0, 1.0]",
    )
    status: SemanticStatus = Field(
        default=SemanticStatus.SUPPORTED,
        description="Confirmation or abstention status",
    )
    participants: list[EventParticipant] = Field(
        default_factory=list,
        description="Entities participating in this semantic event",
    )
    start_timestamp_seconds: float = Field(..., ge=0.0)
    end_timestamp_seconds: float = Field(..., ge=0.0)
    evidence_ids: list[UUID] = Field(
        default_factory=list,
        description="Canonical Evidence IDs explicitly cited to support this claim",
    )
    supporting_event_ids: list[UUID] = Field(
        default_factory=list,
        description="Temporal Event IDs corroborating this claim",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured semantic attributes (action, sentiment, intent, etc.)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Telemetry and model generation metadata",
    )

    @model_validator(mode="after")
    def _validate_interval(self) -> SemanticEventPayload:
        if self.end_timestamp_seconds < self.start_timestamp_seconds:
            raise ValueError(
                f"end_timestamp_seconds ({self.end_timestamp_seconds}) cannot be less than "
                f"start_timestamp_seconds ({self.start_timestamp_seconds})"
            )
        return self


class RoutingRequest(BaseModel):
    """Request submitted to the Semantic Router to evaluate a candidate."""

    candidate: CandidateEvent
    query: str | None = Field(
        default=None,
        description="Optional natural language search query to evaluate relevance against",
    )
    priority_override: RoutingPriority | None = Field(
        default=None,
        description="Optional explicit priority override",
    )
    force_refresh: bool = Field(
        default=False,
        description="Bypass cache and force fresh VLM inference if True",
    )


class RoutingDecisionResult(BaseModel):
    """The decision and telemetry outcome emitted by the Semantic Router."""

    decision: RoutingDecision = Field(..., description="Action taken by the router")
    candidate_id: UUID = Field(..., description="Candidate event evaluated")
    reason: str = Field(..., description="Detailed rationale for the routing decision")
    estimated_tokens: int = Field(default=0, ge=0, description="Estimated VLM prompt tokens")
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Routing decision latency in ms")
    payload: SemanticEventPayload | None = Field(
        default=None,
        description="Final semantic payload if reasoning was performed or served from cache",
    )
    telemetry: dict[str, Any] = Field(
        default_factory=dict,
        description="Observability metrics (provider, model, hash, crop count, validation)",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
