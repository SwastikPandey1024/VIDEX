"""VIDEX typed domain schemas for events and event evidence relationships."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from videx.domain.schemas import EvidenceType
from videx.events.types import EventSeverity, EventStatus, EventType


class EventParticipant(BaseModel):
    """An entity involved in an Event (e.g. object track, text observation, speaker)."""

    model_config = ConfigDict(frozen=True)

    participant_id: UUID | str = Field(..., description="Unique identifier of the entity")
    participant_type: str = Field(
        ...,
        description="Category of participant: 'track', 'text', 'speaker', 'zone', 'scene'",
    )
    role: str = Field(
        default="subject",
        description="Role in event: 'subject', 'target', 'source', 'zone', 'speaker'",
    )
    label: str | None = Field(
        default=None,
        description="Descriptive name or class label (e.g. 'car', 'UP32AB1234', 'speaker_0')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity context or attributes",
    )


class EventEvidence(BaseModel):
    """Explicit relationship mapping an Event to a concrete Evidence record."""

    model_config = ConfigDict(frozen=True)

    evidence_id: UUID = Field(
        default_factory=uuid4,
        description="ID of the underlying canonical Evidence record",
    )
    timestamp_seconds: float = Field(
        ...,
        ge=0.0,
        description="Authoritative video timestamp anchored to this evidence",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of supporting evidence (DETECTION, TRACK, OCR, AUDIO_TRANSCRIPT, etc.)",
    )
    role: str = Field(
        default="supporting",
        description=(
            "Functional role: 'trigger', 'supporting', 'appearance', 'disappearance', "
            "'pre_change', 'post_change', 'context'"
        ),
    )
    frame_id: UUID | None = Field(
        default=None,
        description="Frame ID if frame-anchored",
    )
    track_id: UUID | None = Field(
        default=None,
        description="Track ID if track-anchored",
    )
    ocr_observation_id: UUID | None = Field(
        default=None,
        description="OCR observation ID if OCR-anchored",
    )
    transcript_segment_id: UUID | None = Field(
        default=None,
        description="Audio transcript segment ID if audio-anchored",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Lightweight provenance metadata (frame index, source provider, etc.)",
    )

    @property
    def timestamp(self) -> float:
        """Convenience alias for timestamp_seconds."""
        return self.timestamp_seconds


class Event(BaseModel):
    """A detected temporal event in the video grounded in concrete evidence.

    Distinguishes:
    - Instantaneous events: ``end_timestamp_seconds is None`` or ``start == end``
    - Interval events: ``start_timestamp_seconds < end_timestamp_seconds``
    """

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event",
    )
    video_id: UUID = Field(..., description="Parent video UUID")
    event_type: EventType = Field(..., description="Taxonomy category of this event")

    # ── Temporal Bounds ───────────────────────────────────────────────────
    start_timestamp_seconds: float = Field(
        ...,
        ge=0.0,
        description="Authoritative video timestamp when the event began",
    )
    end_timestamp_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="When the event ended (None if instantaneous)",
    )

    # ── Quality & Lifecycle Status ────────────────────────────────────────
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated confidence score [0.0, 1.0]",
    )
    status: EventStatus = Field(
        default=EventStatus.CONFIRMED,
        description="Current confirmation state of the event",
    )
    severity: EventSeverity = Field(
        default=EventSeverity.INFO,
        description="Operational importance level",
    )

    # ── Grounded Provenance & Relations ───────────────────────────────────
    participants: list[EventParticipant] = Field(
        default_factory=list,
        description="Entities participating in this event",
    )
    evidence_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of all supporting canonical Evidence records",
    )
    event_evidence: list[EventEvidence] = Field(
        default_factory=list,
        description="Structured relationship records linking this Event to supporting Evidence",
    )

    # ── Context & Attributes ──────────────────────────────────────────────
    source_module: str = Field(
        default="event_engine",
        description="Engine or detector component that emitted this event",
    )
    description: str = Field(
        default="",
        description="Human-readable summary of the event",
    )
    zone_name: str | None = Field(
        default=None,
        description="Named spatial zone involved (if applicable)",
    )
    track_ids: list[UUID] = Field(
        default_factory=list,
        description="List of primary Track IDs involved (backward compatibility)",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured domain attributes (velocity, direction, text, etc.)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional detector metadata",
    )

    # ── Audit Datetimes ───────────────────────────────────────────────────
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this event record was created",
    )

    @model_validator(mode="after")
    def _validate_and_sync(self) -> Event:
        # Validate temporal interval
        if (
            self.end_timestamp_seconds is not None
            and self.end_timestamp_seconds < self.start_timestamp_seconds
        ):
            raise ValueError(
                f"end_timestamp_seconds ({self.end_timestamp_seconds}) cannot be less than "
                f"start_timestamp_seconds ({self.start_timestamp_seconds})"
            )

        # Synchronize track_ids with participants
        existing_part_ids = {str(p.participant_id) for p in self.participants}
        for trk_id in self.track_ids:
            if str(trk_id) not in existing_part_ids:
                self.participants.append(
                    EventParticipant(
                        participant_id=trk_id,
                        participant_type="track",
                        role="subject",
                    )
                )
                existing_part_ids.add(str(trk_id))

        # Synchronize evidence_ids from event_evidence
        ev_id_set = set(self.evidence_ids)
        for ee in self.event_evidence:
            if ee.evidence_id not in ev_id_set:
                self.evidence_ids.append(ee.evidence_id)
                ev_id_set.add(ee.evidence_id)

        # Ensure description is non-empty
        if not self.description:
            self.description = f"{self.event_type.value} at {self.start_timestamp_seconds:.2f}s"

        return self

    # ── Convenience Properties ────────────────────────────────────────────

    @property
    def start_timestamp(self) -> float:
        """Alias for start_timestamp_seconds."""
        return self.start_timestamp_seconds

    @property
    def end_timestamp(self) -> float:
        """Effective end timestamp (equals start_timestamp if instantaneous)."""
        if self.end_timestamp_seconds is not None:
            return self.end_timestamp_seconds
        return self.start_timestamp_seconds

    @property
    def duration(self) -> float:
        """Duration of the event in seconds (0.0 for instantaneous events)."""
        if self.end_timestamp_seconds is None:
            return 0.0
        return max(0.0, self.end_timestamp_seconds - self.start_timestamp_seconds)

    @property
    def duration_seconds(self) -> float:
        """Alias for duration."""
        return self.duration

    @property
    def is_instantaneous(self) -> bool:
        """True if the event marks a single point in time rather than a duration."""
        return (
            self.end_timestamp_seconds is None
            or self.end_timestamp_seconds == self.start_timestamp_seconds
        )

    @property
    def is_interval(self) -> bool:
        """True if the event spans a temporal duration."""
        return not self.is_instantaneous

    @property
    def created_at(self) -> datetime:
        """Alias for detected_at."""
        return self.detected_at


class EventEngineConfig(BaseModel):
    """Typed configuration thresholds for the Temporal Event Intelligence Engine.

    Every threshold is explicitly documented to prevent arbitrary heuristics.
    """

    model_config = ConfigDict(frozen=True)

    lifecycle_confirmation_threshold: int = Field(
        default=2,
        ge=1,
        description="Minimum confirmed frames before confirming appearance (filters noise)",
    )
    disappearance_threshold_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Temporal absence (s) before confirming disappearance vs occlusion",
    )
    movement_velocity_threshold_px_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Pixel velocity threshold distinguishing stationary dwelling from motion",
    )
    direction_change_degrees_threshold: float = Field(
        default=45.0,
        ge=0.0,
        le=180.0,
        description="Angular deflection threshold in degrees required for direction change",
    )
    spatial_boundary_tolerance_px: float = Field(
        default=5.0,
        ge=0.0,
        description="Pixel-space buffer around zone boundaries to absorb jitter",
    )
    spatial_debounce_interval_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum temporal gap (s) before re-emitting enter/exit events",
    )
    temporal_near_interval_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum temporal interval (s) between events to be NEAR_IN_TIME",
    )
    ocr_spatial_iou_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="IoU threshold between bounding boxes to associate text replacement events",
    )
    min_track_length: int = Field(
        default=2,
        ge=1,
        description="Minimum track length in frames to consider for event generation",
    )
