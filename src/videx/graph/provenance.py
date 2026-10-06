"""Provenance schemas for graph nodes and edges."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from videx.graph.types import DerivationType


class NodeProvenance(BaseModel):
    """Immutable audit and source provenance for an individual graph node."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., description="Canonical ID of the underlying domain entity")
    source_type: str = Field(..., description="Source entity class name (e.g. 'Event', 'Track')")
    video_id: str = Field(..., description="Canonical video ID scoping this entity")
    timestamp_start: float | None = Field(
        default=None,
        ge=0.0,
        description="Authoritative starting presentation timestamp (PTS in seconds)",
    )
    timestamp_end: float | None = Field(
        default=None,
        ge=0.0,
        description="Authoritative ending presentation timestamp (PTS in seconds)",
    )
    evidence_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Referenced canonical Evidence IDs grounding this entity",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional domain metadata",
    )

    @classmethod
    def create(
        cls,
        source_id: str | UUID,
        source_type: str,
        video_id: str | UUID,
        timestamp_start: float | None = None,
        timestamp_end: float | None = None,
        evidence_ids: Sequence[str | UUID] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NodeProvenance:
        """Helper to create NodeProvenance with string conversions."""
        raw_evidence = evidence_ids or ()
        normalized_ev = tuple(str(e) for e in raw_evidence)
        return cls(
            source_id=str(source_id),
            source_type=source_type,
            video_id=str(video_id),
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            evidence_ids=normalized_ev,
            metadata=dict(metadata or {}),
        )


class EdgeProvenance(BaseModel):
    """Audit and derivation metadata for a directed graph edge."""

    model_config = ConfigDict(frozen=True)

    derivation: DerivationType = Field(
        default=DerivationType.DETERMINISTIC,
        description="Derivation method behind relationship",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this relationship",
    )
    reason: str = Field(
        default="",
        description="Explicit rule or justification creating this edge",
    )
    threshold: float | None = Field(
        default=None,
        description="Applicable threshold value (e.g. near_threshold_seconds, iou_threshold)",
    )
    actual_value: float | None = Field(
        default=None,
        description="Measured value (e.g. actual_gap_seconds, measured_iou)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra derivation context",
    )
