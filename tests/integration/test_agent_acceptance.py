"""End-to-end integration acceptance tests for the VIDEX Agentic Investigator."""

from __future__ import annotations

import pytest

from videx.agent.answer import render_answer
from videx.agent.mock import MockInvestigator, create_mock_context
from videx.agent.schemas import InvestigationRequest
from videx.agent.types import ClaimStatus, InvestigationStatus


@pytest.fixture
def investigator() -> MockInvestigator:
    return MockInvestigator()


def test_investigation_temporal_query(investigator: MockInvestigator):
    """1. Test question: What happened between 0 and 5 seconds?"""
    result = investigator.ask("What happened between 0 and 5 seconds?")

    assert result.status == InvestigationStatus.COMPLETED
    assert result.video_id == investigator.context.video_id
    assert len(result.claims) >= 1
    assert any(c.status == ClaimStatus.SUPPORTED for c in result.claims)
    assert any(c.epistemic_status == "deterministically observed" for c in result.claims)
    assert "search_events" in result.tools_executed
    assert result.trace.steps[0].result is not None
    assert result.trace.steps[0].result.success

    # Verify rendering
    rendered = render_answer(result)
    assert "Investigation Report" in rendered
    assert "SUPPORTED" in rendered


def test_investigation_track_query(investigator: MockInvestigator):
    """2. Test question: What happened to Track 7?"""
    result = investigator.ask("What happened to Track 7?")

    assert result.status == InvestigationStatus.COMPLETED
    assert len(result.claims) >= 1
    assert any("Track 7" in c.claim_text or "person" in c.claim_text for c in result.claims)
    assert any(c.status == ClaimStatus.SUPPORTED for c in result.claims)
    assert "get_track" in result.tools_executed


def test_investigation_ocr_query(investigator: MockInvestigator):
    """3. Test question: What OCR text appeared?"""
    result = investigator.ask("What OCR text appeared on screen?")

    assert result.status == InvestigationStatus.COMPLETED
    assert any("STOP" in c.claim_text for c in result.claims)
    assert any(c.status == ClaimStatus.SUPPORTED for c in result.claims)
    assert "get_ocr" in result.tools_executed


def test_investigation_speech_query(investigator: MockInvestigator):
    """4. Test question: What speech was detected?"""
    result = investigator.ask("What speech was detected in audio?")

    assert result.status == InvestigationStatus.COMPLETED
    assert any("Driver waiting" in c.claim_text for c in result.claims)
    assert any(c.status == ClaimStatus.SUPPORTED for c in result.claims)
    assert "get_transcript" in result.tools_executed


def test_investigation_relational_neighbors_query(investigator: MockInvestigator):
    """5. Test question: Which events were close in time?"""
    result = investigator.ask("Which events are related neighbors to event ev-1?")

    assert result.status == InvestigationStatus.COMPLETED
    assert "find_related_events" in result.tools_executed
    assert "find_temporal_neighbors" in result.tools_executed
    assert len(result.trace.steps) >= 2


def test_investigation_event_evidence_query(investigator: MockInvestigator):
    """6. Test question: What evidence supports Event X?"""
    result = investigator.ask("What evidence supports event ev-1?")

    assert result.status == InvestigationStatus.COMPLETED
    assert "get_event" in result.tools_executed
    assert "get_evidence" in result.tools_executed
    assert any(c.status == ClaimStatus.SUPPORTED for c in result.claims)
    assert len(result.evidence_references) >= 1


def test_investigation_semantic_interaction_supported(investigator: MockInvestigator):
    """7. Test question: Did Track A interact with Track B? (Supported case)"""
    result = investigator.ask("Did the person Track 7 interact with vehicle Track 12?")

    assert result.status == InvestigationStatus.COMPLETED
    assert "reason_semantic" in result.tools_executed
    # Verifies epistemic distinction
    assert any(c.epistemic_status == "VLM inferred" for c in result.claims)
    assert result.trace.semantic_calls >= 1


def test_investigation_semantic_abstention_insufficient_evidence():
    """Test question: Did the suspect draw a weapon? (Truthful abstention case)."""
    ctx = create_mock_context()
    # Mock runner returns inconclusive/insufficient evidence for weapon query
    def weapon_runner(query: str, cand_event_id: str | None = None):
        return {
            "answer": "No visual or auditory evidence of weapon drawn.",
            "confidence": 0.15,
            "evidence_ids": [],
            "event_ids": [],
        }

    ctx.semantic_runner = weapon_runner
    inv = MockInvestigator()
    inv.context = ctx

    req = InvestigationRequest.create(
        video_id=ctx.video_id,
        question="Did the person draw a weapon?",
    )
    result = inv.investigate(req)

    # Agent must not fabricate claims when evidence is insufficient
    assert result.status in (
        InvestigationStatus.INSUFFICIENT_EVIDENCE,
        InvestigationStatus.COMPLETED,
    )
    # The VLM candidate claim cites no evidence IDs so gate marks it rejected/insufficient
    assert any(
        c.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.REJECTED)
        for c in result.claims
    )
    assert not any(
        c.status == ClaimStatus.SUPPORTED and "weapon" in c.claim_text
        for c in result.claims
    )


def test_cross_video_isolation_rejection(investigator: MockInvestigator):
    """Verifies queries scoping an alien video_id return INSUFFICIENT_EVIDENCE."""
    alien_req = InvestigationRequest.create(
        video_id="99999999-9999-9999-9999-999999999999",
        question="What happened between 0 and 5 seconds?",
    )
    result = investigator.investigate(alien_req)

    assert result.status == InvestigationStatus.INSUFFICIENT_EVIDENCE
    assert all(
        c.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.REJECTED)
        for c in result.claims
    )
    assert "Insufficient" in result.answer


def test_investigation_trace_auditing(investigator: MockInvestigator):
    """Verifies comprehensive execution trace generation for Phase 9 UI explainability."""
    result = investigator.ask("What text appeared on screen?")

    trace = result.trace
    assert trace.investigation_id == result.investigation_id
    assert trace.video_id == result.video_id
    assert trace.user_question == result.question
    assert trace.plan is not None
    assert len(trace.steps) >= 1
    assert trace.total_latency_ms > 0.0

    step0 = trace.steps[0]
    assert step0.step_index == 0
    assert step0.action.tool_name == "get_ocr"
    assert step0.result is not None
    assert step0.result.success
    assert step0.duration_ms >= 0.0
