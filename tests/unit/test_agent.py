"""Unit tests for VIDEX Investigator planner, bounded executor, and evidence gate."""

from __future__ import annotations

import time

from videx.agent.claims import Claim
from videx.agent.evidence_gate import EvidenceSufficiencyGate
from videx.agent.executor import InvestigationExecutor
from videx.agent.mock import create_mock_context
from videx.agent.planner import InvestigationPlanner
from videx.agent.registry import ToolDefinition, ToolRegistry
from videx.agent.schemas import InvestigationPlan, InvestigationRequest, ToolCall, ToolResult
from videx.agent.types import ClaimStatus, InvestigationStatus, QuestionCategory


def test_planner_classification():
    planner = InvestigationPlanner()

    assert (
        planner.classify_question("What happened between 10 and 15 seconds?")
        == QuestionCategory.TEMPORAL
    )
    assert planner.classify_question("What happened to Track 7?") == QuestionCategory.TRACK
    assert planner.classify_question("What OCR text appeared on screen?") == QuestionCategory.OCR
    assert planner.classify_question("What speech was detected?") == QuestionCategory.AUDIO
    assert (
        planner.classify_question("What happened near the vehicle in the zone?")
        == QuestionCategory.SPATIAL
    )
    assert (
        planner.classify_question("Which events are related to Event 1?")
        == QuestionCategory.RELATIONAL
    )
    assert (
        planner.classify_question("Did the person interact with the car?")
        == QuestionCategory.SEMANTIC
    )
    assert planner.classify_question("What happened in event ev-1?") == QuestionCategory.EVENT


def test_planner_plan_and_tool_generation():
    planner = InvestigationPlanner()
    req = InvestigationRequest.create(
        video_id="00000000-0000-0000-0000-000000000001",
        question="What happened between 2.0 and 5.0 seconds?",
    )
    plan = planner.create_plan(req)
    assert plan.category == QuestionCategory.TEMPORAL
    assert "search_events" in plan.required_tools

    calls = planner.generate_initial_tool_calls(req, plan)
    assert len(calls) >= 1
    assert calls[0].tool_name == "search_events"
    assert calls[0].parameters.get("time_start") == 2.0
    assert calls[0].parameters.get("time_end") == 5.0


def test_executor_max_steps_limit():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="dummy_tool", description="Dummy test tool"),
        lambda _: ToolResult(call_id="", tool_name="dummy_tool", success=True, data={}),
    )
    executor = InvestigationExecutor(registry)
    plan = InvestigationPlan(
        category=QuestionCategory.GENERAL,
        objective="test limits",
        video_id="vid-1",
    )
    req = InvestigationRequest.create(
        video_id="vid-1",
        question="test",
        max_steps=2,
    )

    calls = [
        ToolCall.create("dummy_tool", {"step": i})
        for i in range(5)
    ]
    steps, results, status = executor.execute(req, plan, calls)
    assert len(steps) == 2
    assert status == InvestigationStatus.MAX_STEPS_EXCEEDED


def test_executor_duplicate_call_prevention():
    call_counts = {"count": 0}

    def counting_handler(_: dict) -> ToolResult:
        call_counts["count"] += 1
        return ToolResult(
            call_id="",
            tool_name="counting_tool",
            success=True,
            data={"val": call_counts["count"]},
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="counting_tool", description="Counting test tool"),
        counting_handler,
    )
    executor = InvestigationExecutor(registry)
    plan = InvestigationPlan(
        category=QuestionCategory.GENERAL,
        objective="test duplicates",
        video_id="vid-1",
    )
    req = InvestigationRequest.create(video_id="vid-1", question="test")

    duplicate_calls = [
        ToolCall.create("counting_tool", {"arg": "same"}),
        ToolCall.create("counting_tool", {"arg": "same"}),
    ]
    steps, results, status = executor.execute(req, plan, duplicate_calls)
    assert len(steps) == 2
    assert status == InvestigationStatus.COMPLETED
    # Handler must have run only once
    assert call_counts["count"] == 1
    assert "Duplicate call detected" in steps[1].thought


def test_executor_timeout_limit():
    def slow_handler(_: dict) -> ToolResult:
        time.sleep(0.05)
        return ToolResult(call_id="", tool_name="slow_tool", success=True, data={})

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="slow_tool", description="Slow test tool"),
        slow_handler,
    )
    executor = InvestigationExecutor(registry)
    plan = InvestigationPlan(
        category=QuestionCategory.GENERAL,
        objective="timeout",
        video_id="vid-1",
    )
    req = InvestigationRequest.create(
        video_id="vid-1",
        question="test",
        timeout_seconds=0.02,
        max_steps=10,
    )

    calls = [ToolCall.create("slow_tool", {"idx": i}) for i in range(5)]
    steps, results, status = executor.execute(req, plan, calls)
    assert status in (InvestigationStatus.TIMEOUT, InvestigationStatus.COMPLETED)
    assert len(steps) < 5


def test_evidence_sufficiency_gate():
    ctx = create_mock_context()
    gate = EvidenceSufficiencyGate(ctx)

    # 1. Valid claim
    valid_ev_id = list(ctx.evidence_records.keys())[0]
    c_valid = Claim.create(
        video_id=ctx.video_id,
        claim_text="Valid claim",
        confidence=0.9,
        epistemic_status="deterministically observed",
        evidence_ids=[valid_ev_id],
    )
    res_valid = gate.check_claim(c_valid, ctx.video_id)
    assert res_valid.is_sufficient
    assert res_valid.status == ClaimStatus.SUPPORTED

    # 2. Missing evidence and events
    c_empty = Claim.create(
        video_id=ctx.video_id,
        claim_text="No evidence claim",
        confidence=0.9,
        epistemic_status="deterministically observed",
        evidence_ids=[],
        event_ids=[],
    )
    res_empty = gate.check_claim(c_empty, ctx.video_id)
    assert not res_empty.is_sufficient
    assert res_empty.status == ClaimStatus.INSUFFICIENT_EVIDENCE

    # 3. Nonexistent evidence ID
    c_fake = Claim.create(
        video_id=ctx.video_id,
        claim_text="Fake evidence claim",
        confidence=0.9,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-nonexistent"],
    )
    res_fake = gate.check_claim(c_fake, ctx.video_id)
    assert not res_fake.is_sufficient
    assert res_fake.status == ClaimStatus.INSUFFICIENT_EVIDENCE

    # 4. Inverted timestamps
    c_inv = Claim.create(
        video_id=ctx.video_id,
        claim_text="Inverted time claim",
        confidence=0.9,
        epistemic_status="deterministically observed",
        evidence_ids=[valid_ev_id],
        timestamps=(5.0, 1.0),
    )
    res_inv = gate.check_claim(c_inv, ctx.video_id)
    assert not res_inv.is_sufficient
    assert res_inv.status == ClaimStatus.REJECTED
