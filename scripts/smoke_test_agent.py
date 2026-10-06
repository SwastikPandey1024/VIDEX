"""Phase 8 Smoke Test: Agentic Investigation, Query Reasoning, and Claim Auditing."""

from __future__ import annotations

import sys
import time

from videx.agent.claims import ClaimValidator
from videx.agent.mock import MockInvestigator
from videx.agent.types import ClaimStatus, InvestigationStatus


def run_smoke_test() -> None:
    print("=" * 70)
    print("VIDEX Phase 8 — Agentic Investigator Smoke Test")
    print("=" * 70)

    investigator = MockInvestigator()
    print(f"Initialized MockInvestigator for target video: {investigator.context.video_id}")

    test_queries = [
        ("1. Temporal Window", "What happened between 0 and 5 seconds?"),
        ("2. Track Kinematics", "What happened to Track 7?"),
        ("3. On-Screen Text / OCR", "What OCR text appeared on screen?"),
        ("4. Speech Audio / ASR", "What speech was detected in the video?"),
        ("5. Relational Neighbors", "Which events are related neighbors to event ev-1?"),
        ("6. Event Grounding", "What evidence supports event ev-1?"),
        (
            "7. Semantic Interaction (Supported)",
            "Did the person Track 7 interact with vehicle Track 12?",
        ),
        ("8. Truthful Abstention (Uncorroborated)", "Did the suspect draw a concealed weapon?"),
    ]

    total_start = time.perf_counter()
    results = []

    print("\nExecuting Representative Investigations:")
    print("-" * 70)

    for label, query in test_queries:
        t0 = time.perf_counter()
        result = investigator.ask(query)
        elapsed = (time.perf_counter() - t0) * 1000.0

        supported_claims = sum(1 for c in result.claims if c.status == ClaimStatus.SUPPORTED)
        epistemic_tags = {c.epistemic_status for c in result.claims}

        print(f"\n[{label}]")
        print(f"  Query:      \"{query}\"")
        print(f"  Status:     {result.status.value.upper()}")
        print(f"  Tools:      {', '.join(result.tools_executed)}")
        print(f"  Claims:     {len(result.claims)} total ({supported_claims} supported)")
        print(f"  Epistemic:  {sorted(epistemic_tags)}")
        ans_preview = f"{result.answer[:100]}..." if len(result.answer) > 100 else result.answer
        print(f"  Answer:     {ans_preview}")
        print(f"  Latency:    {result.trace.total_latency_ms:.2f} ms (wall: {elapsed:.2f} ms)")

        results.append((label, result))

    # Assertions on acceptance criteria
    # 1. Temporal
    assert results[0][1].status == InvestigationStatus.COMPLETED
    assert "search_events" in results[0][1].tools_executed

    # 2. Track
    assert results[1][1].status == InvestigationStatus.COMPLETED
    assert "get_track" in results[1][1].tools_executed

    # 3. OCR
    assert results[2][1].status == InvestigationStatus.COMPLETED
    assert "get_ocr" in results[2][1].tools_executed

    # 4. ASR
    assert results[3][1].status == InvestigationStatus.COMPLETED
    assert "get_transcript" in results[3][1].tools_executed

    # 5. Relational
    assert results[4][1].status == InvestigationStatus.COMPLETED
    assert "find_related_events" in results[4][1].tools_executed

    # 6. Event evidence
    assert results[5][1].status == InvestigationStatus.COMPLETED
    assert "get_evidence" in results[5][1].tools_executed

    # 7. Semantic supported
    assert results[6][1].status == InvestigationStatus.COMPLETED
    assert "reason_semantic" in results[6][1].tools_executed
    assert any(c.epistemic_status == "VLM inferred" for c in results[6][1].claims)

    # 8. Truthful abstention on weapon
    abstention_res = results[7][1]
    assert abstention_res.status == InvestigationStatus.INSUFFICIENT_EVIDENCE
    assert not any(c.status == ClaimStatus.SUPPORTED for c in abstention_res.claims)
    print("\n[PASS] Truthful abstention verified: Returned INSUFFICIENT_EVIDENCE.")

    # 9. Cross-video isolation test
    foreign_vid = "00000000-0000-0000-0000-000000000999"
    validator = ClaimValidator()
    foreign_report = validator.validate(
        claims=results[0][1].claims,
        expected_video_id=foreign_vid,
    )
    assert not foreign_report.is_valid
    assert len(foreign_report.rejected_claims) >= 1
    print("[PASS] Cross-video isolation verified: Foreign video claims rejected by ClaimValidator.")

    total_elapsed = (time.perf_counter() - total_start) * 1000.0

    print("\n" + "=" * 70)
    print("PERFORMANCE & LATENCY SUMMARY")
    print("=" * 70)
    print(f"Total investigations executed: {len(test_queries)}")
    print(f"Total end-to-end wall latency: {total_elapsed:.2f} ms")

    avg_planning = sum(r[1].metrics.get("planning_latency_ms", 0.0) for r in results) / len(results)
    avg_tool = sum(r[1].metrics.get("tool_latency_ms", 0.0) for r in results) / len(results)
    avg_val = sum(r[1].metrics.get("validation_latency_ms", 0.0) for r in results) / len(results)
    avg_total = sum(r[1].metrics.get("total_latency_ms", 0.0) for r in results) / len(results)

    print(f"Average Planning Latency:      {avg_planning:.3f} ms")
    print(f"Average Tool Execution Latency:{avg_tool:.3f} ms")
    print(f"Average Claim Validation:      {avg_val:.3f} ms")
    print(f"Average Total Investigation:   {avg_total:.3f} ms")

    print("\n[PASS] ALL PHASE 8 AGENTIC INVESTIGATION INTEGRITY CHECKS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:
        print(f"\n❌ Phase 8 Smoke Test failed: {exc}", file=sys.stderr)
        raise
