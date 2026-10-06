"""Structured investigation result models and answer rendering for VIDEX."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from videx.agent.claims import Claim
from videx.agent.schemas import EvidenceReference, InvestigationTrace
from videx.agent.types import InvestigationStatus


class InvestigationResult(BaseModel):
    """Auditable, evidence-grounded final result produced by VIDEX Investigator."""

    model_config = ConfigDict(frozen=True)

    investigation_id: str = Field(..., description="Unique investigation identifier")
    video_id: str = Field(..., description="Scope video identifier")
    question: str = Field(..., description="User query investigated")
    status: InvestigationStatus = Field(..., description="Overall investigation outcome status")
    answer: str = Field(..., description="Direct synthetic answer to the query")
    claims: list[Claim] = Field(
        default_factory=list,
        description="Discrete grounded factual claims supporting the answer",
    )
    evidence_references: list[EvidenceReference] = Field(
        default_factory=list,
        description="Direct references to supporting evidence",
    )
    events_used: list[str] = Field(
        default_factory=list,
        description="Event IDs used during investigation reasoning",
    )
    tools_executed: list[str] = Field(
        default_factory=list,
        description="Names of tools invoked during investigation",
    )
    trace: InvestigationTrace = Field(
        ..., description="Full auditable investigation execution trace"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metrics and latency breakdown",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary payload."""
        return self.model_dump()


def render_answer(result: InvestigationResult) -> str:
    """Renders human-readable markdown presentation from a structured InvestigationResult."""
    lines: list[str] = []
    lines.append(f"### Investigation Report [{result.investigation_id[:8]}]")
    lines.append(f"**Question:** {result.question}")
    lines.append(f"**Status:** `{result.status.value}` | **Video:** `{result.video_id}`")
    lines.append("")
    lines.append("#### Answer")
    lines.append(result.answer)
    lines.append("")

    if result.claims:
        lines.append("#### Grounded Claims")
        for idx, claim in enumerate(result.claims, 1):
            ts_str = ""
            if claim.timestamps is not None:
                t0, t1 = claim.timestamps
                ts_str = f" [{t0:.1f}s - {t1:.1f}s]"
            lines.append(
                f"{idx}. **[{claim.status.value.upper()}]** ({claim.epistemic_status}) "
                f"{claim.claim_text}{ts_str}"
            )
            ev_ids = ", ".join(claim.evidence_ids) if claim.evidence_ids else "None"
            evnt_ids = ", ".join(claim.event_ids) if claim.event_ids else "None"
            lines.append(f"   - *Confidence:* {claim.confidence:.2f}")
            lines.append(f"   - *Evidence IDs:* {ev_ids}")
            lines.append(f"   - *Event IDs:* {evnt_ids}")
        lines.append("")

    if result.evidence_references:
        lines.append("#### Supporting Canonical Evidence")
        for ref in result.evidence_references:
            conf_str = f" conf={ref.confidence:.2f}" if ref.confidence is not None else ""
            lines.append(
                f"- `{ref.evidence_id}` ({ref.evidence_type}) [{ref.source_module}]{conf_str}"
            )
        lines.append("")

    lines.append("#### Execution Summary")
    tools_str = ", ".join(result.tools_executed) if result.tools_executed else "none"
    duration = result.metrics.get("total_latency_ms", result.trace.total_latency_ms)
    lines.append(f"- **Steps:** {len(result.trace.steps)}")
    lines.append(f"- **Tools Called:** {tools_str}")
    lines.append(f"- **Total Latency:** {duration:.2f} ms")

    return "\n".join(lines)
