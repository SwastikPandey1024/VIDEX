"""Evidence sufficiency gate for the VIDEX Agentic Investigator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from videx.agent.claims import Claim
from videx.agent.tools import AgentToolContext
from videx.agent.types import ClaimStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SufficiencyCheckResult:
    """Outcome of evaluating evidence sufficiency for a claim."""

    is_sufficient: bool
    status: ClaimStatus
    reason: str


class EvidenceSufficiencyGate:
    """Verifies evidence sufficiency and scoping before permitting factual claims.

    Guarantees:
    - Supporting evidence exists in canonical records
    - Evidence belongs to the target video (isolation)
    - Timestamps are temporally valid
    - Referenced events and participants exist
    - Epistemic status is non-empty and preserved
    """

    def __init__(self, context: AgentToolContext) -> None:
        self._ctx = context

    def check_claim(self, claim: Claim, target_video_id: str) -> SufficiencyCheckResult:
        """Evaluates whether the given claim has sufficient, valid grounding evidence."""
        # 1. Epistemic status check
        if not claim.epistemic_status:
            return SufficiencyCheckResult(
                is_sufficient=False,
                status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                reason="Claim is missing mandatory epistemic status",
            )

        # 2. Supporting evidence or events must be present
        if not claim.evidence_ids and not claim.event_ids:
            return SufficiencyCheckResult(
                is_sufficient=False,
                status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                reason="Claim has no supporting evidence IDs or event IDs",
            )

        # 3. Target video scoping and evidence existence
        for ev_id in claim.evidence_ids:
            canonical_ev = self._ctx.evidence_records.get(ev_id)
            if canonical_ev is not None:
                if str(canonical_ev.video_id) != target_video_id:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=(
                            f"Evidence '{ev_id}' video_id '{canonical_ev.video_id}' violates "
                            f"target isolation '{target_video_id}'"
                        ),
                    )
                continue

            if ev_id in self._ctx.ocr_observations:
                ocr_rec = self._ctx.ocr_observations[ev_id]
                if str(ocr_rec.video_id) != target_video_id:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=f"OCR observation '{ev_id}' violates target video isolation",
                    )
                continue

            if ev_id in self._ctx.transcript_segments:
                tr_rec = self._ctx.transcript_segments[ev_id]
                if str(tr_rec.video_id) != target_video_id:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=f"Transcript segment '{ev_id}' violates target video isolation",
                    )
                continue

            if ev_id in self._ctx.frames:
                fr_rec = self._ctx.frames[ev_id]
                if str(fr_rec.video_id) != target_video_id:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=f"Frame '{ev_id}' violates target video isolation",
                    )
                continue

            # Check graph node existence across possible types
            node = (
                self._ctx.graph_query_service.get_node(f"evidence:{ev_id}")
                or self._ctx.graph_query_service.get_node(f"ocr_obs:{ev_id}")
                or self._ctx.graph_query_service.get_node(f"transcript:{ev_id}")
                or self._ctx.graph_query_service.get_node(f"frame:{ev_id}")
                or self._ctx.graph_query_service.get_node(ev_id)
            )
            if node is None:
                return SufficiencyCheckResult(
                    is_sufficient=False,
                    status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                    reason=f"Referenced evidence ID '{ev_id}' does not exist in store",
                )
            if node.video_id != target_video_id:
                return SufficiencyCheckResult(
                    is_sufficient=False,
                    status=ClaimStatus.REJECTED,
                    reason=(
                        f"Evidence '{ev_id}' video_id '{node.video_id}' violates "
                        f"target isolation '{target_video_id}'"
                    ),
                )

        # 4. Event existence & video isolation
        for evnt_id in claim.event_ids:
            canonical_evnt = self._ctx.events.get(evnt_id)
            if canonical_evnt is None:
                node = self._ctx.graph_query_service.get_node(f"event:{evnt_id}")
                if node is None:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                        reason=f"Referenced event ID '{evnt_id}' does not exist in store",
                    )
                if node.video_id != target_video_id:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=f"Event '{evnt_id}' violates target video isolation",
                    )
            else:
                if str(canonical_evnt.video_id) != target_video_id:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=f"Event '{evnt_id}' violates target video isolation",
                    )

        # 5. Timestamp validation
        if claim.timestamps:
            for ts in claim.timestamps:
                if ts < 0:
                    return SufficiencyCheckResult(
                        is_sufficient=False,
                        status=ClaimStatus.REJECTED,
                        reason=f"Negative timestamp {ts} is invalid",
                    )
            if len(claim.timestamps) >= 2 and claim.timestamps[0] > claim.timestamps[1]:
                return SufficiencyCheckResult(
                    is_sufficient=False,
                    status=ClaimStatus.REJECTED,
                    reason=f"Inverted timestamp interval: {claim.timestamps}",
                )

        # 6. Participant validation
        for p_id in claim.participant_ids:
            if (
                p_id not in self._ctx.tracks
                and self._ctx.graph_query_service.get_node(f"track:{p_id}") is None
            ):
                return SufficiencyCheckResult(
                    is_sufficient=False,
                    status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                    reason=f"Referenced participant ID '{p_id}' does not exist in store",
                )

        return SufficiencyCheckResult(
            is_sufficient=True,
            status=ClaimStatus.SUPPORTED,
            reason="All grounding evidence, participant references, and isolation checks passed",
        )
