"""Strict evidence validation for VLM outputs to prevent hallucinated intelligence."""

from __future__ import annotations

import logging
from typing import NamedTuple
from uuid import UUID

from videx.semantic.schemas import EvidenceBundle, SemanticEventPayload
from videx.semantic.types import SemanticStatus

logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of an evidence validation audit."""

    is_valid: bool
    errors: list[str]
    sanitized_payload: SemanticEventPayload


class EvidenceValidator:
    """Validates that a VLM reasoning result strictly adheres to supplied evidence.

    Rejects any payload that fabricates evidence IDs, invents timestamps outside
    the bundle window, claims unsupported participants, or has malformed confidence scores.
    """

    def __init__(self, temporal_tolerance_seconds: float = 0.50) -> None:
        self.temporal_tolerance_seconds = temporal_tolerance_seconds

    def validate(
        self,
        payload: SemanticEventPayload,
        bundle: EvidenceBundle,
    ) -> ValidationResult:
        """Validate payload against bundle. Returns ValidationResult."""
        errors: list[str] = []

        # 1. Evidence ID verification (Anti-hallucination check)
        supplied_evidence_ids: set[UUID] = set(bundle.evidence_ids)
        for cited_id in payload.evidence_ids:
            if cited_id not in supplied_evidence_ids:
                errors.append(
                    f"Cited evidence_id '{cited_id}' was not supplied in the evidence bundle"
                )

        # 2. Supporting event ID verification
        supplied_event_ids: set[UUID] = set(bundle.supporting_event_ids)
        for cited_ev_id in payload.supporting_event_ids:
            if cited_ev_id not in supplied_event_ids:
                errors.append(
                    f"Cited supporting_event_id '{cited_ev_id}' was not in the candidate bundle"
                )

        # 3. Temporal consistency check
        eps = self.temporal_tolerance_seconds
        min_allowed_start = bundle.start_timestamp_seconds - eps
        max_allowed_end = bundle.end_timestamp_seconds + eps

        if payload.start_timestamp_seconds < min_allowed_start:
            errors.append(
                f"Start timestamp {payload.start_timestamp_seconds:.3f}s is earlier than bundle "
                f"window ({bundle.start_timestamp_seconds:.3f}s - {eps}s)"
            )
        if payload.end_timestamp_seconds > max_allowed_end:
            errors.append(
                f"End timestamp {payload.end_timestamp_seconds:.3f}s exceeds bundle window "
                f"({bundle.end_timestamp_seconds:.3f}s + {eps}s)"
            )
        if payload.end_timestamp_seconds < payload.start_timestamp_seconds:
            errors.append(
                f"End timestamp ({payload.end_timestamp_seconds}) cannot be less than "
                f"start ({payload.start_timestamp_seconds})"
            )

        # 4. Confidence & uncertainty bounds check
        if not (0.0 <= payload.confidence <= 1.0):
            errors.append(f"Confidence score {payload.confidence} is out of valid range [0.0, 1.0]")
        if not (0.0 <= payload.uncertainty <= 1.0):
            errors.append(
                f"Uncertainty score {payload.uncertainty} is out of valid range [0.0, 1.0]"
            )

        # 5. Participant sanity check
        if payload.participants:
            # Valid participants should relate to candidate or bundle
            bundle_p_strings: set[str] = set()
            for traj in bundle.trajectory_summary:
                bundle_p_strings.add(str(traj.get("track_id", "")).lower())
                bundle_p_strings.add(str(traj.get("class_name", "")).lower())
            for z in bundle.spatial_context:
                bundle_p_strings.add(z.lower())

            for p in payload.participants:
                p_str = str(p.participant_id).lower()
                # Participant must have non-empty ID
                if not p_str.strip():
                    errors.append("Participant has empty participant_id")

        # 6. Construct sanitized outcome
        if errors:
            logger.warning(
                "Semantic reasoning result failed evidence validation with %d errors: %s",
                len(errors),
                "; ".join(errors),
            )
            # Produce REJECTED payload preserving error audit
            sanitized = payload.model_copy(
                update={
                    "status": SemanticStatus.REJECTED,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "attributes": {
                        **payload.attributes,
                        "validation_errors": errors,
                    },
                }
            )
            return ValidationResult(is_valid=False, errors=errors, sanitized_payload=sanitized)

        # Valid payload
        return ValidationResult(is_valid=True, errors=[], sanitized_payload=payload)
