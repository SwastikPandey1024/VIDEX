"""Master Semantic Router coordinating candidate selection, caching, VLM reasoning,
and validation.
"""

from __future__ import annotations

import logging
import time

from videx.audio.pipeline import AudioPipelineResult
from videx.domain.schemas import EvidenceType
from videx.events.engine import EventTimeline
from videx.events.schemas import Event, EventEvidence
from videx.events.types import EventSeverity, EventStatus, EventType
from videx.ingestion.base import VideoReader
from videx.ocr.pipeline import OCRPipelineResult
from videx.perception.pipeline import PerceptionResult
from videx.semantic.cache import SemanticCache
from videx.semantic.candidates import CandidateSelector
from videx.semantic.evidence_bundle import EvidenceBundleBuilder
from videx.semantic.mock import MockVLMProvider
from videx.semantic.policy import RoutingPolicy
from videx.semantic.providers import VLMProvider
from videx.semantic.schemas import (
    CandidateEvent,
    EvidenceBundle,
    RoutingDecisionResult,
    RoutingRequest,
    SemanticEventPayload,
)
from videx.semantic.types import RoutingDecision, RoutingPriority, SemanticStatus
from videx.semantic.validator import EvidenceValidator

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Orchestrates candidate selection, gating, evidence bundling, VLM reasoning,
    and strict evidence validation to produce auditable semantic events.
    """

    def __init__(
        self,
        candidate_selector: CandidateSelector | None = None,
        bundle_builder: EvidenceBundleBuilder | None = None,
        policy: RoutingPolicy | None = None,
        cache: SemanticCache | None = None,
        provider: VLMProvider | None = None,
        validator: EvidenceValidator | None = None,
    ) -> None:
        self.candidate_selector = candidate_selector or CandidateSelector()
        self.bundle_builder = bundle_builder or EvidenceBundleBuilder()
        self.policy = policy or RoutingPolicy()
        self.cache = cache or SemanticCache()
        self.provider = provider or MockVLMProvider()
        self.validator = validator or EvidenceValidator()

    def route_timeline(
        self,
        timeline: EventTimeline,
        query: str | None = None,
        video_reader: VideoReader | None = None,
        perception_result: PerceptionResult | None = None,
        ocr_result: OCRPipelineResult | None = None,
        audio_result: AudioPipelineResult | None = None,
    ) -> list[tuple[RoutingDecisionResult, Event | None]]:
        """Evaluate an entire EventTimeline, extracting and routing candidate events.

        Returns:
            List of tuples (RoutingDecisionResult, SemanticEvent | None).
        """
        # 1. Deterministic Candidate Selection
        candidates = self.candidate_selector.select_candidates(
            timeline_or_events=timeline,
            query=query,
            video_id=timeline.video_id,
        )

        results: list[tuple[RoutingDecisionResult, Event | None]] = []
        for cand in candidates:
            res, ev = self.route_candidate(
                candidate=cand,
                timeline=timeline,
                query=query,
                video_reader=video_reader,
                perception_result=perception_result,
                ocr_result=ocr_result,
                audio_result=audio_result,
            )
            results.append((res, ev))

        return results

    def route_candidate(
        self,
        candidate: CandidateEvent,
        timeline: EventTimeline | list[Event],
        query: str | None = None,
        video_reader: VideoReader | None = None,
        perception_result: PerceptionResult | None = None,
        ocr_result: OCRPipelineResult | None = None,
        audio_result: AudioPipelineResult | None = None,
        priority_override: RoutingPriority | None = None,
        force_refresh: bool = False,
    ) -> tuple[RoutingDecisionResult, Event | None]:
        """Route a single candidate event through policy gating, cache, VLM, and validation."""
        start_time = time.monotonic()
        req = RoutingRequest(
            candidate=candidate,
            query=query,
            priority_override=priority_override,
            force_refresh=force_refresh,
        )

        # 1. Gating Check (Saliency & Query relevance)
        gating_decision, gating_reason = self.policy.evaluate_gating(
            candidate=candidate,
            query=query,
            priority_override=priority_override,
        )
        if gating_decision != RoutingDecision.DISPATCH_VLM:
            latency = (time.monotonic() - start_time) * 1000.0
            decision_res = RoutingDecisionResult(
                decision=gating_decision,
                candidate_id=candidate.candidate_id,
                reason=gating_reason,
                latency_ms=round(latency, 2),
                telemetry={
                    "candidate_id": str(candidate.candidate_id),
                    "saliency": candidate.saliency_score,
                    "query_score": candidate.query_relevance_score,
                    "gating_status": gating_decision.value,
                },
            )
            return decision_res, None

        # 2. Construct Minimal Evidence Bundle
        bundle = self.bundle_builder.build_bundle(
            candidate=candidate,
            timeline=timeline,
            video_reader=video_reader,
            perception_result=perception_result,
            ocr_result=ocr_result,
            audio_result=audio_result,
        )

        # 3. Request Hash & Cache Lookup
        req_hash = self.cache.compute_request_hash(
            bundle=bundle,
            query=query,
            provider_name=self.provider.provider_name,
        )

        if not force_refresh:
            cached_payload = self.cache.get(req_hash)
            if cached_payload is not None:
                latency = (time.monotonic() - start_time) * 1000.0
                decision_res = RoutingDecisionResult(
                    decision=RoutingDecision.SERVE_FROM_CACHE,
                    candidate_id=candidate.candidate_id,
                    reason="Serving identical candidate context from semantic cache",
                    cache_hit=True,
                    latency_ms=round(latency, 2),
                    payload=cached_payload,
                    telemetry={
                        "candidate_id": str(candidate.candidate_id),
                        "request_hash": req_hash,
                        "cache_hit": True,
                        "provider": self.provider.provider_name,
                    },
                )
                cached_ev = self._create_semantic_event(cached_payload, candidate, bundle)
                return decision_res, cached_ev

        # 4. Token Estimation & Budget Enforcement
        est_tokens = self.policy.estimate_tokens(bundle)
        if not self.policy.check_budget(est_tokens):
            latency = (time.monotonic() - start_time) * 1000.0
            decision_res = RoutingDecisionResult(
                decision=RoutingDecision.SUPPRESS_BUDGET_EXCEEDED,
                candidate_id=candidate.candidate_id,
                reason=f"Sliding token budget exceeded (estimated: {est_tokens} tokens)",
                estimated_tokens=est_tokens,
                latency_ms=round(latency, 2),
                telemetry={
                    "candidate_id": str(candidate.candidate_id),
                    "estimated_tokens": est_tokens,
                    "budget": self.policy.get_budget_snapshot()._asdict(),
                },
            )
            return decision_res, None

        # 5. Invoke VLM Reasoning Provider
        raw_payload = self.provider.analyze(bundle, req)
        self.policy.record_usage(est_tokens)

        # 6. Validate Evidence Provenance
        val_result = self.validator.validate(raw_payload, bundle)
        final_payload = val_result.sanitized_payload

        # 7. Update Cache if Valid
        if val_result.is_valid and final_payload.status != SemanticStatus.REJECTED:
            self.cache.set(req_hash, final_payload)

        # 8. Construct Canonical Semantic Event (if not rejected)
        semantic_ev: Event | None = None
        if final_payload.status != SemanticStatus.REJECTED:
            semantic_ev = self._create_semantic_event(final_payload, candidate, bundle)

        latency = (time.monotonic() - start_time) * 1000.0
        decision_res = RoutingDecisionResult(
            decision=RoutingDecision.DISPATCH_VLM,
            candidate_id=candidate.candidate_id,
            reason=f"Successfully reasoned with VLM provider '{self.provider.provider_name}'",
            estimated_tokens=est_tokens,
            cache_hit=False,
            latency_ms=round(latency, 2),
            payload=final_payload,
            telemetry={
                "candidate_id": str(candidate.candidate_id),
                "provider": self.provider.provider_name,
                "model": final_payload.metadata.get("model", "unknown"),
                "request_hash": req_hash,
                "latency_ms": round(latency, 2),
                "input_frame_count": len(bundle.keyframes),
                "crop_count": len(bundle.crops),
                "token_budget": est_tokens,
                "cache_hit": False,
                "validation_status": "passed" if val_result.is_valid else "rejected",
                "validation_errors": val_result.errors,
            },
        )
        return decision_res, semantic_ev

    def _create_semantic_event(
        self,
        payload: SemanticEventPayload,
        candidate: CandidateEvent,
        bundle: EvidenceBundle,
    ) -> Event:
        """Convert validated SemanticEventPayload into an evidence-backed Event record."""
        # Structured EventEvidence descriptors linking cited evidence IDs
        event_evidences: list[EventEvidence] = []
        for eid in payload.evidence_ids:
            event_evidences.append(
                EventEvidence(
                    evidence_id=eid,
                    timestamp_seconds=payload.start_timestamp_seconds,
                    evidence_type=EvidenceType.EVENT,
                    role="semantic_grounding",
                    provenance={"semantic_type": payload.semantic_event_type.value},
                )
            )

        # Status mapping
        event_status = EventStatus.CONFIRMED
        if payload.status in (SemanticStatus.UNCERTAIN, SemanticStatus.INSUFFICIENT_EVIDENCE):
            event_status = EventStatus.DETECTED
        elif payload.status == SemanticStatus.REJECTED:
            event_status = EventStatus.REJECTED

        # Severity mapping
        severity = EventSeverity.INFO
        if candidate.priority == RoutingPriority.CRITICAL:
            severity = EventSeverity.CRITICAL
        elif candidate.priority == RoutingPriority.HIGH:
            severity = EventSeverity.HIGH
        elif candidate.priority == RoutingPriority.MEDIUM:
            severity = EventSeverity.MEDIUM

        return Event(
            video_id=candidate.video_id,
            event_type=EventType.CUSTOM,
            start_timestamp_seconds=payload.start_timestamp_seconds,
            end_timestamp_seconds=payload.end_timestamp_seconds,
            confidence=payload.confidence,
            status=event_status,
            severity=severity,
            participants=payload.participants,
            evidence_ids=payload.evidence_ids,
            event_evidence=event_evidences,
            source_module=f"semantic_router:{self.provider.provider_name}",
            description=f"{payload.claim} — {payload.description}",
            attributes={
                "semantic_event_type": payload.semantic_event_type.value,
                "claim": payload.claim,
                "uncertainty": payload.uncertainty,
                "semantic_status": payload.status.value,
                "candidate_id": str(candidate.candidate_id),
                "supporting_event_ids": [str(eid) for eid in payload.supporting_event_ids],
                **payload.attributes,
            },
            metadata=payload.metadata,
        )
