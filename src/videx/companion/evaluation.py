"""Evaluation metrics and quality gates for VIDEX YouTube Learning Companion.

Reuses and extends Phase 9 evaluation standards:
- Factual grounding and unsupported claim rate
- Key concept coverage compared to reference gold-standard
- LaTeX equation syntax fidelity
- Timestamp validity (bounded within duration, non-negative intervals)
- Epistemic uncertainty reporting and duplicate concept detection
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from videx.companion.schemas import StudyNote
from videx.companion.synthesizer import validate_latex_balance

logger = logging.getLogger(__name__)


class CompanionEvaluationResult(BaseModel):
    """Rigorous evaluation metrics for a synthesized study note."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="Target video ID")
    total_claims: int = Field(default=0, ge=0, description="Total substantive claims evaluated")
    grounded_claims: int = Field(default=0, ge=0, description="Claims with valid evidence links")
    unsupported_claims: int = Field(default=0, ge=0, description="Claims lacking evidence links")
    unsupported_claim_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Unsupported claims / total claims"
    )
    total_equations: int = Field(default=0, ge=0, description="Total mathematical equations")
    valid_equations: int = Field(default=0, ge=0, description="Syntactically valid LaTeX equations")
    latex_equation_fidelity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Valid equations / total equations"
    )
    total_timestamps: int = Field(
        default=0, ge=0, description="Total evidence timestamp references"
    )
    valid_timestamps: int = Field(
        default=0, ge=0, description="Timestamps within valid video duration interval"
    )
    timestamp_accuracy_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Valid timestamps / total timestamps"
    )
    concept_coverage: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Recall of key reference concepts"
    )
    duplicate_claim_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of duplicate/redundant claims"
    )
    uncertainty_count: int = Field(
        default=0, ge=0, description="Number of uncertainty flags raised"
    )
    is_acceptable: bool = Field(
        default=True, description="Whether the note meets Phase 9 acceptance threshold"
    )
    notes: tuple[str, ...] = Field(
        default_factory=tuple, description="Audit observations and evaluation notes"
    )


def evaluate_study_note(
    note: StudyNote,
    reference_concepts: list[str] | None = None,
) -> CompanionEvaluationResult:
    """Evaluate note against factual grounding, equation fidelity, and timestamp accuracy."""
    total_claims = 0
    grounded_claims = 0
    unsupported_claims = 0
    total_eqs = 0
    valid_eqs = 0
    total_ts = 0
    valid_ts = 0
    claim_texts: list[str] = []
    audit_notes: list[str] = []

    duration = note.duration_seconds

    for sec in note.sections:
        # Check definitions
        for d in sec.definitions:
            total_claims += 1
            claim_texts.append(d.term.lower())
            if d.evidence_refs:
                grounded_claims += 1
                for ref in d.evidence_refs:
                    total_ts += 1
                    if (
                        ref.start_seconds >= 0.0
                        and ref.end_seconds >= ref.start_seconds
                        and (duration == 0.0 or ref.end_seconds <= duration + 5.0)
                    ):
                        valid_ts += 1
                    else:
                        audit_notes.append(
                            f"Invalid timestamp interval: [{ref.start_seconds:.1f}, "
                            f"{ref.end_seconds:.1f}] vs dur {duration:.1f}"
                        )
            else:
                unsupported_claims += 1

        # Check theorems
        for t in sec.theorems:
            total_claims += 1
            claim_texts.append(t.name.lower())
            if t.evidence_refs:
                grounded_claims += 1
            else:
                unsupported_claims += 1

        # Check formulas
        for f in sec.formulas:
            total_claims += 1
            total_eqs += 1
            if validate_latex_balance(f.latex_expression):
                valid_eqs += 1
            else:
                audit_notes.append(f"Unbalanced LaTeX in formula: {f.latex_expression}")

            if f.evidence_refs:
                grounded_claims += 1
            else:
                unsupported_claims += 1

        # Check examples
        for ex in sec.worked_examples:
            total_claims += 1
            if ex.evidence_refs:
                grounded_claims += 1
            else:
                unsupported_claims += 1

        # Check traps
        for trap in sec.exam_traps:
            total_claims += 1
            if trap.evidence_refs:
                grounded_claims += 1
            else:
                unsupported_claims += 1

    # Check key takeaways
    for takeaway in note.key_takeaways:
        total_claims += 1
        if takeaway.evidence_refs:
            grounded_claims += 1
        else:
            unsupported_claims += 1

    unsupported_rate = round(unsupported_claims / total_claims, 3) if total_claims > 0 else 0.0
    latex_fidelity = round(valid_eqs / total_eqs, 3) if total_eqs > 0 else 1.0
    ts_accuracy = round(valid_ts / total_ts, 3) if total_ts > 0 else 1.0

    # Concept coverage against reference concepts
    coverage = 1.0
    if reference_concepts:
        found = 0
        note_text_corpus = (
            note.executive_summary.lower()
            + " "
            + " ".join(s.title.lower() + " " + s.overview.lower() for s in note.sections)
        )
        for c in reference_concepts:
            if c.lower() in note_text_corpus:
                found += 1
        coverage = round(found / len(reference_concepts), 3)

    # Duplicate claim rate
    unique_claims = set(claim_texts)
    duplicate_count = len(claim_texts) - len(unique_claims)
    duplicate_rate = round(duplicate_count / len(claim_texts), 3) if claim_texts else 0.0

    # Acceptance threshold
    is_acceptable = (
        unsupported_rate == 0.0
        and latex_fidelity >= 0.95
        and ts_accuracy >= 0.95
        and coverage >= 0.50
    )

    return CompanionEvaluationResult(
        video_id=note.video_id,
        total_claims=total_claims,
        grounded_claims=grounded_claims,
        unsupported_claims=unsupported_claims,
        unsupported_claim_rate=unsupported_rate,
        total_equations=total_eqs,
        valid_equations=valid_eqs,
        latex_equation_fidelity=latex_fidelity,
        total_timestamps=total_ts,
        valid_timestamps=valid_ts,
        timestamp_accuracy_rate=ts_accuracy,
        concept_coverage=coverage,
        duplicate_claim_rate=duplicate_rate,
        uncertainty_count=len(note.uncertainties),
        is_acceptable=is_acceptable,
        notes=tuple(audit_notes),
    )
