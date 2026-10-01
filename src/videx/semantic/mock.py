"""Deterministic MockVLMProvider for offline, CPU-only unit and integration testing."""

from __future__ import annotations

import time
from collections.abc import Callable
from uuid import uuid4

from videx.events.schemas import EventParticipant
from videx.semantic.schemas import EvidenceBundle, RoutingRequest, SemanticEventPayload
from videx.semantic.types import SemanticEventType, SemanticStatus


class MockVLMProvider:
    """Deterministic, configurable mock implementation of VLMProvider.

    Allows unit testing the entire Semantic Router pipeline without GPU dependencies
    or network connectivity, while testing edge cases and negative scenarios.
    """

    def __init__(
        self,
        default_status: SemanticStatus = SemanticStatus.SUPPORTED,
        default_confidence: float = 0.90,
        simulated_latency_ms: float = 5.0,
        custom_handler: Callable[[EvidenceBundle, RoutingRequest], SemanticEventPayload]
        | None = None,
        should_hallucinate_evidence_id: bool = False,
        should_produce_timestamp_mismatch: bool = False,
        should_fail_confidence: bool = False,
    ) -> None:
        self.default_status = default_status
        self.default_confidence = default_confidence
        self.simulated_latency_ms = simulated_latency_ms
        self.custom_handler = custom_handler
        self.should_hallucinate_evidence_id = should_hallucinate_evidence_id
        self.should_produce_timestamp_mismatch = should_produce_timestamp_mismatch
        self.should_fail_confidence = should_fail_confidence
        self.invocations: list[tuple[EvidenceBundle, RoutingRequest]] = []

    @property
    def provider_name(self) -> str:
        return "mock_vlm"

    def warmup(self) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def analyze(
        self,
        bundle: EvidenceBundle,
        request: RoutingRequest,
    ) -> SemanticEventPayload:
        """Generate a deterministic SemanticEventPayload reflecting the bundle."""
        self.invocations.append((bundle, request))

        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        if self.custom_handler is not None:
            return self.custom_handler(bundle, request)

        # 1. Determine cited evidence IDs
        cited_evidence_ids = list(bundle.evidence_ids)
        if self.should_hallucinate_evidence_id:
            cited_evidence_ids.append(uuid4())  # Phantom ID not present in bundle!

        # 2. Determine timestamps
        start_ts = bundle.start_timestamp_seconds
        end_ts = bundle.end_timestamp_seconds
        if self.should_produce_timestamp_mismatch:
            start_ts = end_ts + 10.0  # Outside bundle interval!

        # 3. Determine confidence
        confidence = self.default_confidence
        if self.should_fail_confidence:
            confidence = 1.85  # Invalid score > 1.0!

        # 4. Handle abstention on insufficient context
        status = self.default_status
        if (
            len(bundle.crops) == 0
            and len(bundle.transcript_segments) == 0
            and len(bundle.ocr_observations) == 0
        ):
            status = SemanticStatus.INSUFFICIENT_EVIDENCE
            claim = (
                "Insufficient sensory evidence in bundle to synthesize high-level semantic claim"
            )
            description = (
                "Observed: Empty visual crops and audio segments. "
                "Interpretation: Abstaining from semantic inference to prevent speculation."
            )
            confidence = 0.0
            uncertainty = 1.0
        else:
            claim = self._synthesize_claim(bundle, request)
            crops_n = len(bundle.crops)
            asrs_n = len(bundle.transcript_segments)
            ocrs_n = len(bundle.ocr_observations)
            description = (
                f"Observed: {crops_n} crops, {asrs_n} speech segments, {ocrs_n} OCR observations. "
                f"Interpretation: {claim} occurred during {start_ts:.2f}s - {end_ts:.2f}s."
            )
            uncertainty = round(1.0 - confidence, 3)

        participants = [
            EventParticipant(
                participant_id=p_id,
                participant_type="track" if "-" in str(p_id) else "entity",
                role="subject",
                label=str(p_id),
            )
            for p_id in request.candidate.participant_ids[:4]
        ]

        # Determine semantic category
        sem_type = SemanticEventType.GENERAL_ACTIVITY
        if bundle.spatial_context:
            sem_type = (
                SemanticEventType.SECURITY_INCIDENT
                if "checkpoint" in str(bundle.spatial_context).lower()
                else SemanticEventType.VEHICLE_ACTIVITY
            )
        elif bundle.transcript_segments:
            sem_type = SemanticEventType.INTERACTION

        return SemanticEventPayload(
            semantic_event_type=sem_type,
            claim=claim,
            description=description,
            confidence=confidence,
            uncertainty=uncertainty,
            status=status,
            participants=participants,
            start_timestamp_seconds=start_ts,
            end_timestamp_seconds=end_ts,
            evidence_ids=cited_evidence_ids,
            supporting_event_ids=list(bundle.supporting_event_ids),
            attributes={
                "provider": self.provider_name,
                "keyframes_analyzed": len(bundle.keyframes),
                "crops_analyzed": len(bundle.crops),
                "query": request.query,
            },
            metadata={
                "latency_ms": self.simulated_latency_ms,
                "model": "mock-reasoner-v1",
            },
        )

    def _synthesize_claim(self, bundle: EvidenceBundle, request: RoutingRequest) -> str:
        """Create a sensible grounded claim based on bundle attributes."""
        parts: list[str] = []
        if bundle.spatial_context:
            parts.append(f"activity observed at {', '.join(bundle.spatial_context)}")
        if bundle.transcript_segments:
            first_txt = bundle.transcript_segments[0].get("text", "")
            parts.append(f"accompanied by utterance '{first_txt}'")
        if bundle.ocr_observations:
            first_ocr = bundle.ocr_observations[0].get("text", "")
            parts.append(f"displaying text '{first_ocr}'")

        if parts:
            return "Subject " + " and ".join(parts)
        return "Coordinated entity activity verified across temporal interval"
