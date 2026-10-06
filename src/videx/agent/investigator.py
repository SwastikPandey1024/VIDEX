"""Core VIDEX Investigator tying together planning, execution, and validation."""

from __future__ import annotations

import logging
import time
from typing import Any

from videx.agent.answer import InvestigationResult
from videx.agent.claims import Claim, ClaimValidator
from videx.agent.evidence_gate import EvidenceSufficiencyGate
from videx.agent.executor import InvestigationExecutor
from videx.agent.planner import InvestigationPlanner
from videx.agent.registry import ToolRegistry
from videx.agent.schemas import (
    EvidenceReference,
    InvestigationPlan,
    InvestigationRequest,
    InvestigationTrace,
    ToolResult,
)
from videx.agent.telemetry import InvestigationTelemetry
from videx.agent.tools import AgentToolContext, register_investigation_tools
from videx.agent.types import ClaimStatus, InvestigationStatus, QuestionCategory
from videx.graph.schemas import NodeFilter
from videx.graph.types import GraphNodeType

logger = logging.getLogger(__name__)


class Investigator:
    """Evidence-grounded video investigator.

    Orchestrates:
    User Question -> Intent Understanding -> Plan -> Typed Tool Calls
    -> Evidence Retrieval -> Sufficiency Gate -> Optional Semantic Reasoning
    -> Claim Validation -> Auditable Answer
    """

    def __init__(
        self,
        context: AgentToolContext,
        registry: ToolRegistry | None = None,
        planner: InvestigationPlanner | None = None,
        executor: InvestigationExecutor | None = None,
    ) -> None:
        self.context = context
        if registry is None:
            self.registry = ToolRegistry()
            register_investigation_tools(self.registry, self.context)
        else:
            self.registry = registry

        self.planner = planner or InvestigationPlanner()
        self.executor = executor or InvestigationExecutor(self.registry)
        self.sufficiency_gate = EvidenceSufficiencyGate(self.context)
        self.claim_validator = ClaimValidator()

    def investigate(self, request: InvestigationRequest) -> InvestigationResult:
        """Executes a full, evidence-grounded investigation for the given request."""
        telemetry = InvestigationTelemetry(
            investigation_id=request.investigation_id,
            video_id=request.video_id,
        )

        # 1. Planning
        t_plan_start = time.perf_counter()
        plan = self.planner.create_plan(request)
        initial_calls = self.planner.generate_initial_tool_calls(request, plan)
        plan_ms = (time.perf_counter() - t_plan_start) * 1000.0
        telemetry.record_planning_time(plan_ms)

        # 2. Bounded Execution
        steps, tool_results, exec_status = self.executor.execute(
            request=request,
            plan=plan,
            tool_calls=initial_calls,
            telemetry=telemetry,
        )

        # 3. Evidence Extraction & Candidate Claim Formulation
        candidate_claims, ev_refs, events_used = self._formulate_candidate_claims(
            request=request,
            plan=plan,
            results=tool_results,
        )

        # 4. Evidence Sufficiency Gate & Claim Validation
        t_val_start = time.perf_counter()
        passed_sufficiency: list[Claim] = []
        all_passed = True

        for candidate in candidate_claims:
            suff = self.sufficiency_gate.check_claim(candidate, request.video_id)
            if not suff.is_sufficient:
                candidate = candidate.with_status(suff.status)
                all_passed = False
            passed_sufficiency.append(candidate)

        all_graph_nodes = self.context.graph_query_service.query_nodes(
            NodeFilter(video_id=request.video_id)
        )
        known_eids = (
            set(self.context.evidence_records.keys())
            | {n.source_id for n in all_graph_nodes if n.node_type == GraphNodeType.EVIDENCE}
            | {eid for n in all_graph_nodes for eid in n.evidence_ids}
            | set(self.context.ocr_observations.keys())
            | {
                n.source_id
                for n in all_graph_nodes
                if n.node_type == GraphNodeType.OCR_OBSERVATION
            }
            | set(self.context.transcript_segments.keys())
            | {
                n.source_id
                for n in all_graph_nodes
                if n.node_type == GraphNodeType.TRANSCRIPT_SEGMENT
            }
            | set(self.context.frames.keys())
            | {n.source_id for n in all_graph_nodes if n.node_type == GraphNodeType.FRAME}
        )
        known_events = set(self.context.events.keys()) | {
            n.source_id for n in all_graph_nodes if n.node_type == GraphNodeType.EVENT
        }
        known_parts = set(self.context.tracks.keys()) | {
            n.source_id for n in all_graph_nodes if n.node_type == GraphNodeType.TRACK
        }

        val_report = self.claim_validator.validate(
            claims=passed_sufficiency,
            expected_video_id=request.video_id,
            known_evidence_ids=known_eids,
            known_event_ids=known_events,
            known_participant_ids=known_parts,
        )
        if not val_report.is_valid:
            all_passed = False

        validated_claims = val_report.validated_claims + val_report.rejected_claims

        val_ms = (time.perf_counter() - t_val_start) * 1000.0
        telemetry.record_validation_time(val_ms)

        # 5. Synthesize Final Answer & Status
        final_status = exec_status
        if not validated_claims or any(
            c.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.REJECTED)
            for c in validated_claims
        ):
            if any(c.status == ClaimStatus.SUPPORTED for c in validated_claims):
                final_status = InvestigationStatus.COMPLETED
            else:
                final_status = InvestigationStatus.INSUFFICIENT_EVIDENCE

        answer = self._synthesize_answer(request.question, validated_claims, final_status)
        metrics = telemetry.finalize(len(steps))

        tools_executed = [s.action.tool_name for s in steps]
        ev_ids_all = sorted({ref.evidence_id for ref in ev_refs})

        trace = InvestigationTrace(
            investigation_id=request.investigation_id,
            video_id=request.video_id,
            user_question=request.question,
            plan=plan,
            steps=steps,
            evidence_ids=ev_ids_all,
            events_used=sorted(events_used),
            semantic_calls=sum(1 for s in steps if s.action.tool_name == "reason_semantic"),
            total_latency_ms=metrics.total_latency_ms,
            validation_passed=all_passed,
        )

        return InvestigationResult(
            investigation_id=request.investigation_id,
            video_id=request.video_id,
            question=request.question,
            status=final_status,
            answer=answer,
            claims=validated_claims,
            evidence_references=ev_refs,
            events_used=sorted(events_used),
            tools_executed=tools_executed,
            trace=trace,
            metrics={
                "planning_latency_ms": metrics.planning_latency_ms,
                "tool_latency_ms": metrics.tool_latency_ms,
                "graph_latency_ms": metrics.graph_latency_ms,
                "evidence_latency_ms": metrics.evidence_latency_ms,
                "validation_latency_ms": metrics.validation_latency_ms,
                "total_latency_ms": metrics.total_latency_ms,
                "step_count": metrics.step_count,
                "tool_call_count": metrics.tool_call_count,
            },
        )

    def _formulate_candidate_claims(
        self,
        request: InvestigationRequest,
        plan: InvestigationPlan,
        results: list[ToolResult],
    ) -> tuple[list[Claim], list[EvidenceReference], set[str]]:
        """Formulates candidate claims from tool outputs preserving exact epistemic provenance."""
        claims: list[Claim] = []
        ev_refs: list[EvidenceReference] = []
        events_used: set[str] = set()

        for res in results:
            if not res.success or not res.data:
                continue

            # Extract provenance records as evidence references
            for prov in res.provenance:
                ev_list = prov.get("evidence_ids", [])
                for eid in ev_list:
                    ev_refs.append(
                        EvidenceReference(
                            evidence_id=str(eid),
                            evidence_type=str(prov.get("source_type", "evidence")),
                            video_id=request.video_id,
                            timestamp_seconds=prov.get("timestamp_start"),
                            confidence=prov.get("confidence", 1.0),
                            description=f"Provenance from {res.tool_name}",
                        )
                    )

            # 1. search_events
            if res.tool_name == "search_events":
                events_data = res.data.get("events", [])
                for ev in events_data:
                    ev_id = str(ev.get("event_id", ""))
                    events_used.add(ev_id)
                    if plan.category == QuestionCategory.SEMANTIC:
                        # For semantic investigations, search_events provides background context
                        # to the reasoner rather than direct factual answers to the semantic query.
                        continue
                    t0 = ev.get("time_start", 0.0)
                    t1 = ev.get("time_end", t0)
                    desc = ev.get("description") or f"Event {ev.get('event_type')}"
                    claims.append(
                        Claim.create(
                            video_id=request.video_id,
                            claim_text=f"Observed event '{desc}' at {t0:.1f}s - {t1:.1f}s",
                            confidence=ev.get("confidence", 0.95),
                            epistemic_status="deterministically observed",
                            evidence_ids=ev.get("evidence_ids", []),
                            event_ids=[ev_id] if ev_id else [],
                            timestamps=(t0, t1),
                            participant_ids=[str(p) for p in ev.get("participant_ids", [])],
                        )
                    )

            # 2. get_event
            elif res.tool_name == "get_event":
                ev_info = res.data.get("event", {})
                if ev_info:
                    ev_id = str(ev_info.get("event_id", ""))
                    events_used.add(ev_id)
                    t0 = ev_info.get("time_start", 0.0)
                    t1 = ev_info.get("time_end", t0)
                    claims.append(
                        Claim.create(
                            video_id=request.video_id,
                            claim_text=(
                                f"Event {ev_id} ({ev_info.get('event_type')}): "
                                f"{ev_info.get('description', '')}"
                            ),
                            confidence=ev_info.get("confidence", 0.95),
                            epistemic_status="deterministically observed",
                            evidence_ids=ev_info.get("evidence_ids", []),
                            event_ids=[ev_id],
                            timestamps=(t0, t1),
                            participant_ids=[str(p) for p in ev_info.get("participant_ids", [])],
                        )
                    )

            # 3. get_track
            elif res.tool_name == "get_track":
                track_info = res.data.get("track", {})
                if track_info:
                    tr_id = str(track_info.get("track_id", ""))
                    t0 = track_info.get("start_pts", 0.0)
                    t1 = track_info.get("end_pts", t0)
                    claims.append(
                        Claim.create(
                            video_id=request.video_id,
                            claim_text=(
                                f"Track {tr_id} ({track_info.get('primary_class')}) tracked "
                                f"for {track_info.get('duration_seconds', 0.0):.2f}s"
                            ),
                            confidence=0.95,
                            epistemic_status="deterministically observed",
                            evidence_ids=track_info.get("evidence_ids", []),
                            timestamps=(t0, t1),
                            participant_ids=[tr_id],
                        )
                    )

            # 4. get_ocr
            elif res.tool_name == "get_ocr":
                ocr_obs = res.data.get("observations", []) or res.data.get("ocr_observations", [])
                for ocr in ocr_obs:
                    t = ocr.get("pts_seconds", 0.0)
                    raw_oid = ocr.get("observation_id")
                    eids = ocr.get("evidence_ids") or ([raw_oid] if raw_oid else [])
                    claims.append(
                        Claim.create(
                            video_id=request.video_id,
                            claim_text=f"Detected on-screen text: \"{ocr.get('text')}\"",
                            confidence=ocr.get("confidence", 0.9),
                            epistemic_status="deterministically observed",
                            evidence_ids=eids,
                            timestamps=(t, t),
                        )
                    )

            # 5. get_transcript
            elif res.tool_name == "get_transcript":
                segs = res.data.get("segments", []) or res.data.get("transcript_segments", [])
                for seg in segs:
                    t0 = seg.get("start_pts", 0.0)
                    t1 = seg.get("end_pts", t0)
                    spk = seg.get("speaker") or seg.get("speaker_id")
                    spk_str = f" by {spk}" if spk else ""
                    raw_sid = seg.get("segment_id")
                    eids = seg.get("evidence_ids") or ([raw_sid] if raw_sid else [])
                    claims.append(
                        Claim.create(
                            video_id=request.video_id,
                            claim_text=f"Spoken text detected{spk_str}: \"{seg.get('text')}\"",
                            confidence=seg.get("confidence", 0.9),
                            epistemic_status="deterministically observed",
                            evidence_ids=eids,
                            timestamps=(t0, t1),
                        )
                    )

            # 6. find_related_events / temporal_neighbors / spatial_neighbors
            elif res.tool_name in (
                "find_related_events",
                "find_temporal_neighbors",
                "find_spatial_neighbors",
            ):
                items: list[dict[str, Any]] = (
                    res.data.get("related_events")
                    or res.data.get("temporal_neighbors")
                    or res.data.get("spatial_neighbors")
                    or []
                )
                for item in items:
                    target_id = (
                        item.get("event_id")
                        or item.get("neighbor_id")
                        or item.get("source_id")
                    )
                    ev_used = str(target_id)
                    events_used.add(ev_used)
                    eids = item.get("evidence_ids", [])
                    if not eids:
                        ev_nodes = self.context.graph_query_service.evidence_for_event(ev_used)
                        eids = [n.source_id for n in ev_nodes]
                    t0 = item.get("start_pts", 0.0)
                    t1 = item.get("end_pts", t0)
                    lbl = item.get("label", ev_used)
                    claims.append(
                        Claim.create(
                            video_id=request.video_id,
                            claim_text=f"Graph relation identified for {ev_used}: {lbl}",
                            confidence=0.88,
                            epistemic_status="heuristically associated",
                            evidence_ids=eids,
                            event_ids=[ev_used],
                            timestamps=(t0, t1),
                        )
                    )

            # 7. reason_semantic
            elif res.tool_name == "reason_semantic":
                sem = res.data
                sem_desc = sem.get("explanation") or sem.get("answer") or str(sem)
                eids = sem.get("evidence_ids", [])
                st = ClaimStatus.SUPPORTED if eids else ClaimStatus.INSUFFICIENT_EVIDENCE
                claims.append(
                    Claim.create(
                        video_id=request.video_id,
                        claim_text=f"VLM semantic analysis: {sem_desc}",
                        confidence=sem.get("confidence", 0.82),
                        epistemic_status="VLM inferred",
                        evidence_ids=eids,
                        event_ids=sem.get("event_ids", []),
                        status=st,
                    )
                )

        if not claims:
            # Truthful abstention: No evidence found
            claims.append(
                Claim.create(
                    video_id=request.video_id,
                    claim_text=(
                        f"No canonical evidence or events found in video '{request.video_id}' "
                        f"satisfying query '{request.question}'"
                    ),
                    confidence=0.0,
                    epistemic_status="unsupported assertion",
                    status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                )
            )

        return claims, ev_refs, events_used

    @staticmethod
    def _synthesize_answer(
        question: str,
        claims: list[Claim],
        status: InvestigationStatus,
    ) -> str:
        """Synthesizes human-readable answer grounded purely in validated claims."""
        if status == InvestigationStatus.INSUFFICIENT_EVIDENCE:
            return (
                f"Insufficient canonical evidence to answer question: '{question}'. "
                "No validated events or observations match the requested criteria."
            )

        supported = [c for c in claims if c.status == ClaimStatus.SUPPORTED]
        if not supported:
            return "Unable to corroborate factual claims from canonical evidence."

        answer_parts: list[str] = []
        for c in supported:
            answer_parts.append(f"{c.claim_text} ({c.epistemic_status}, conf={c.confidence:.2f}).")

        return " ".join(answer_parts)
