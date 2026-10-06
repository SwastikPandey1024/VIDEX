"""Claim representation, sets, and rigorous epistemic validation."""

from __future__ import annotations

import logging
from typing import NamedTuple
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from videx.agent.types import ClaimStatus

logger = logging.getLogger(__name__)


class Claim(BaseModel):
    """An individual assertion produced during video investigation."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str = Field(..., description="Canonical video ID scoping this claim")
    claim_text: str = Field(..., description="Natural language statement being asserted")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence [0.0, 1.0]")
    epistemic_status: str = Field(
        ...,
        description="Explicit derivation classification (e.g. 'deterministically observed')",
    )
    status: ClaimStatus = Field(
        default=ClaimStatus.SUPPORTED,
        description="Verification state of this claim",
    )
    evidence_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Referenced canonical Evidence IDs grounding this assertion",
    )
    event_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Referenced Event or SemanticEvent IDs",
    )
    timestamps: tuple[float, float] | None = Field(
        default=None,
        description="Temporal interval [start_sec, end_sec] in PTS",
    )
    participant_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tracks, objects, or zones participating in this claim",
    )

    def with_status(self, status: ClaimStatus) -> Claim:
        """Returns a copy of this claim with updated status."""
        return self.model_copy(update={"status": status})

    @classmethod
    def create(
        cls,
        video_id: str | UUID,
        claim_text: str,
        confidence: float,
        epistemic_status: str,
        status: ClaimStatus = ClaimStatus.SUPPORTED,
        evidence_ids: list[str | UUID] | None = None,
        event_ids: list[str | UUID] | None = None,
        timestamps: tuple[float, float] | None = None,
        participant_ids: list[str | UUID] | None = None,
    ) -> Claim:
        """Helper to create Claim with normalized string tuples."""
        raw_ev = evidence_ids or []
        raw_events = event_ids or []
        raw_part = participant_ids or []
        return cls(
            video_id=str(video_id),
            claim_text=claim_text,
            confidence=confidence,
            epistemic_status=epistemic_status,
            status=status,
            evidence_ids=tuple(str(e) for e in raw_ev),
            event_ids=tuple(str(ev) for ev in raw_events),
            timestamps=timestamps,
            participant_ids=tuple(str(p) for p in raw_part),
        )


class ClaimSet(BaseModel):
    """Collection of claims forming the evidentiary basis of an answer."""

    model_config = ConfigDict(frozen=True)

    claims: list[Claim] = Field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.confidence for c in self.claims) / len(self.claims)

    @property
    def is_fully_supported(self) -> bool:
        return bool(self.claims) and all(c.status == ClaimStatus.SUPPORTED for c in self.claims)

    @property
    def has_insufficient_evidence(self) -> bool:
        return any(c.status == ClaimStatus.INSUFFICIENT_EVIDENCE for c in self.claims)


class ClaimValidationReport(NamedTuple):
    """Result of an evidence-grounding claim audit."""

    is_valid: bool
    errors: list[str]
    rejected_claims: list[Claim]
    validated_claims: list[Claim]


class ClaimValidator:
    """Validates claims against retrieved evidence context to enforce grounded citations
    and video isolation.

    Enforces that:
    1. Every SUPPORTED claim cites at least one valid evidence ID.
    2. All cited evidence IDs actually exist in the retrieved evidence repository.
    3. Video ID matches the requested video namespace.
    4. Timestamps are well-formed (non-negative, start <= end).
    5. Epistemic status is non-empty and correctly categorized.
    6. Confidence scores are valid probabilities in [0.0, 1.0].
    """

    ALLOWED_EPISTEMIC_STATUSES: frozenset[str] = frozenset({
        "deterministically observed",
        "heuristically associated",
        "VLM inferred",
        "insufficient evidence",
    })

    def validate(
        self,
        claims: list[Claim],
        expected_video_id: str | UUID,
        known_evidence_ids: set[str] | None = None,
        known_event_ids: set[str] | None = None,
        known_participant_ids: set[str] | None = None,
    ) -> ClaimValidationReport:
        errors: list[str] = []
        rejected: list[Claim] = []
        validated: list[Claim] = []
        exp_vid = str(expected_video_id)

        for c in claims:
            claim_errors: list[str] = []

            # 1. Video isolation
            if c.video_id != exp_vid:
                claim_errors.append(
                    f"Claim video_id '{c.video_id}' does not match expected '{exp_vid}'"
                )

            # 2. Confidence range
            if not (0.0 <= c.confidence <= 1.0):
                claim_errors.append(
                    f"Claim confidence {c.confidence} outside valid range [0.0, 1.0]"
                )

            # 3. Epistemic status presence & validity
            if not c.epistemic_status:
                claim_errors.append("Claim is missing epistemic_status metadata")
            elif c.epistemic_status not in self.ALLOWED_EPISTEMIC_STATUSES:
                claim_errors.append(
                    f"Invalid epistemic_status '{c.epistemic_status}'. "
                    f"Must be one of {sorted(self.ALLOWED_EPISTEMIC_STATUSES)}"
                )

            # 4. Timestamp interval validity
            if c.timestamps is not None:
                t_start, t_end = c.timestamps
                if t_start < 0.0:
                    claim_errors.append(f"Negative start timestamp: {t_start:.3f}s")
                if t_end < t_start:
                    claim_errors.append(
                        f"End timestamp {t_end:.3f}s occurs before start {t_start:.3f}s"
                    )

            # 5. Evidence grounding check
            if c.status == ClaimStatus.SUPPORTED:
                if not c.evidence_ids:
                    claim_errors.append(
                        f"Claim '{c.claim_text}' marked SUPPORTED but cites no evidence_ids"
                    )
                elif known_evidence_ids is not None:
                    missing_eids = set(c.evidence_ids) - known_evidence_ids
                    if missing_eids:
                        claim_errors.append(
                            f"Claim cites unknown/unretrieved evidence_ids: {sorted(missing_eids)}"
                        )

            # 6. Event verification if known events provided
            if known_event_ids is not None and c.event_ids:
                missing_events = set(c.event_ids) - known_event_ids
                if missing_events:
                    claim_errors.append(
                        f"Claim cites unknown event_ids: {sorted(missing_events)}"
                    )

            # 7. Participant verification if known participants provided
            if known_participant_ids is not None and c.participant_ids:
                missing_parts = set(c.participant_ids) - known_participant_ids
                if missing_parts:
                    claim_errors.append(
                        f"Claim cites unknown participant_ids: {sorted(missing_parts)}"
                    )

            if claim_errors:
                errors.extend(f"[Claim {c.claim_id}] {e}" for e in claim_errors)
                # Clone as REJECTED
                rejected.append(
                    Claim(
                        claim_id=c.claim_id,
                        video_id=c.video_id,
                        claim_text=c.claim_text,
                        confidence=0.0,
                        epistemic_status=c.epistemic_status or "insufficient evidence",
                        status=ClaimStatus.REJECTED,
                        evidence_ids=c.evidence_ids,
                        event_ids=c.event_ids,
                        timestamps=c.timestamps,
                        participant_ids=c.participant_ids,
                    )
                )
            else:
                validated.append(c)

        return ClaimValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            rejected_claims=rejected,
            validated_claims=validated,
        )
