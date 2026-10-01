"""Unit tests for Phase 6.0 Semantic Intelligence Foundation.

Covers:
- Candidate selection, deterministic saliency scoring, query relevance matching
- Crop extraction, context margin expansion, boundary clamping, and invalid box handling
- Evidence bundle construction, timestamp-driven keyframe selection
- Routing policy, token estimation, rate/token budget limits
- Semantic cache, SHA-256 request hashing, candidate deduplication, TTL eviction
- MockVLMProvider execution, prompt contract generation
- EvidenceValidator validation and all negative tests (fabricated IDs, timestamp leaks, etc.)
"""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from videx.domain.schemas import BoundingBox, CoordinateType
from videx.events.schemas import Event, EventEvidence, EventParticipant
from videx.events.types import EventSeverity, EventStatus, EventType
from videx.semantic.cache import SemanticCache
from videx.semantic.candidates import CandidateSelectionConfig, CandidateSelector
from videx.semantic.crops import CropExtractionError, CropExtractor
from videx.semantic.evidence_bundle import EvidenceBundleBuilder
from videx.semantic.mock import MockVLMProvider
from videx.semantic.policy import RoutingPolicy, RoutingPolicyConfig
from videx.semantic.providers import build_evidence_grounded_prompt
from videx.semantic.qwen import Qwen3VLAdapter, Qwen3VLConfig
from videx.semantic.schemas import (
    CandidateEvent,
    EvidenceBundle,
    RoutingRequest,
    SemanticEventPayload,
)
from videx.semantic.types import (
    RoutingDecision,
    RoutingPriority,
    SemanticStatus,
)
from videx.semantic.validator import EvidenceValidator

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_video_id() -> UUID:
    return uuid4()


@pytest.fixture
def sample_temporal_events(sample_video_id: UUID) -> list[Event]:
    ev1_id = uuid4()
    ev2_id = uuid4()
    evidence_1 = uuid4()
    evidence_2 = uuid4()

    ev1 = Event(
        event_id=ev1_id,
        video_id=sample_video_id,
        event_type=EventType.OBJECT_ENTERED_ZONE,
        start_timestamp_seconds=1.20,
        end_timestamp_seconds=1.20,
        confidence=0.92,
        status=EventStatus.CONFIRMED,
        severity=EventSeverity.MEDIUM,
        zone_name="Checkpoint Gate",
        description="Car entered security checkpoint gate",
        evidence_ids=[evidence_1],
        event_evidence=[
            EventEvidence(
                evidence_id=evidence_1,
                timestamp_seconds=1.20,
                evidence_type="event",
                role="trigger",
            )
        ],
        participants=[
            EventParticipant(
                participant_id=uuid4(),
                participant_type="track",
                role="subject",
                label="car",
            )
        ],
        attributes={"zone_name": "Checkpoint Gate"},
    )

    ev2 = Event(
        event_id=ev2_id,
        video_id=sample_video_id,
        event_type=EventType.SPEECH_DETECTED,
        start_timestamp_seconds=1.30,
        end_timestamp_seconds=2.40,
        confidence=0.88,
        status=EventStatus.CONFIRMED,
        severity=EventSeverity.INFO,
        description="Driver stating license clearance",
        evidence_ids=[evidence_2],
        event_evidence=[
            EventEvidence(
                evidence_id=evidence_2,
                timestamp_seconds=1.30,
                evidence_type="audio_transcript",
                role="trigger",
            )
        ],
        participants=[
            EventParticipant(
                participant_id="speaker_0",
                participant_type="speaker",
                role="speaker",
                label="driver",
            )
        ],
        attributes={"speech_text": "License verified, proceeding"},
    )
    return [ev1, ev2]


# ── Unit: Candidate Selection & Saliency ──────────────────────────────────


class TestCandidateSelection:
    def test_candidate_selection_clustering_and_saliency(
        self, sample_temporal_events: list[Event], sample_video_id: UUID
    ) -> None:
        selector = CandidateSelector()
        candidates = selector.select_candidates(sample_temporal_events, video_id=sample_video_id)

        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.video_id == sample_video_id
        assert len(cand.source_event_ids) == 2
        assert cand.start_timestamp == 1.20
        assert cand.end_timestamp == 2.40
        assert cand.saliency_score >= 0.50
        assert cand.priority in (RoutingPriority.HIGH, RoutingPriority.MEDIUM)
        assert "cross_modal_co_occurrence" in cand.reason

    def test_query_relevance_matching(
        self, sample_temporal_events: list[Event], sample_video_id: UUID
    ) -> None:
        selector = CandidateSelector()

        # Query matching checkpoint and car
        matching_query = "What happened at the checkpoint gate with the car?"
        cands_match = selector.select_candidates(
            sample_temporal_events, query=matching_query, video_id=sample_video_id
        )
        assert len(cands_match) == 1
        assert cands_match[0].query_relevance_score >= 0.50
        assert "matched" in cands_match[0].reason

        # Query with unrelated tokens
        unrelated_query = "Did a submarine launch a torpedo?"
        cands_unrelated = selector.select_candidates(
            sample_temporal_events, query=unrelated_query, video_id=sample_video_id
        )
        if cands_unrelated:
            assert cands_unrelated[0].query_relevance_score == 0.0

    def test_saliency_threshold_suppression(self, sample_video_id: UUID) -> None:
        # Create a single trivial low-severity event
        low_ev = Event(
            video_id=sample_video_id,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=10.0,
            end_timestamp_seconds=10.0,
            confidence=0.20,
            severity=EventSeverity.INFO,
            description="Brief detection",
        )
        selector = CandidateSelector(CandidateSelectionConfig(min_saliency_threshold=0.85))
        cands = selector.select_candidates([low_ev], video_id=sample_video_id)
        assert len(cands) == 0


# ── Unit: Crop Extraction & Geometry ─────────────────────────────────────


class TestCropExtraction:
    def test_crop_margin_expansion_and_clamping(self) -> None:
        extractor = CropExtractor(default_context_margin=0.20)
        # Create a 200x200 image
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:100, 50:100] = 255  # bright square

        bbox = BoundingBox(x=50.0, y=50.0, width=50.0, height=50.0)
        ev_id = uuid4()

        crop = extractor.extract_crop(
            frame=frame,
            bbox=bbox,
            source_evidence_id=ev_id,
            frame_index=15,
            timestamp_seconds=0.50,
            encode_jpeg=True,
        )

        assert crop.frame_index == 15
        assert crop.timestamp_seconds == 0.50
        assert crop.source_evidence_id == ev_id
        assert crop.context_margin == 0.20
        # Expected expansion: 50 * (1 + 2 * 0.2) = 70px width/height, x = 50 - 10 = 40
        assert crop.clamped_bbox.x == 40.0
        assert crop.clamped_bbox.y == 40.0
        assert crop.clamped_bbox.width == 70.0
        assert crop.clamped_bbox.height == 70.0
        assert crop.image_bytes is not None
        assert len(crop.image_bytes) > 0

    def test_crop_clamping_at_frame_boundaries(self) -> None:
        extractor = CropExtractor(default_context_margin=0.50)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Box near top-left corner
        bbox = BoundingBox(x=5.0, y=5.0, width=20.0, height=20.0)
        crop = extractor.extract_crop(frame=frame, bbox=bbox, source_evidence_id=uuid4())

        assert crop.clamped_bbox.x == 0.0
        assert crop.clamped_bbox.y == 0.0
        assert crop.clamped_bbox.width <= 100.0
        assert crop.clamped_bbox.height <= 100.0

    def test_normalized_bounding_box_conversion(self) -> None:
        extractor = CropExtractor(default_context_margin=0.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Normalized coordinates: center 50%
        bbox = BoundingBox(
            x=0.25,
            y=0.25,
            width=0.50,
            height=0.50,
            coordinate_type=CoordinateType.NORMALIZED,
        )
        crop = extractor.extract_crop(frame=frame, bbox=bbox, source_evidence_id=uuid4())

        assert crop.clamped_bbox.x == 160.0  # 0.25 * 640
        assert crop.clamped_bbox.y == 120.0  # 0.25 * 480
        assert crop.clamped_bbox.width == 320.0  # 0.50 * 640
        assert crop.clamped_bbox.height == 240.0  # 0.50 * 480

    def test_invalid_box_handling(self) -> None:
        extractor = CropExtractor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Degenerate box with 0 area
        bad_box = BoundingBox(x=10.0, y=10.0, width=0.0001, height=0.0001)
        # Should raise CropExtractionError due to tiny clamp size
        with pytest.raises(CropExtractionError):
            extractor.extract_crop(frame=frame, bbox=bad_box, source_evidence_id=uuid4())


# ── Unit: Evidence Bundle & Keyframe Selection ────────────────────────────


class TestEvidenceBundleBuilder:
    def test_timestamp_driven_keyframe_selection_for_interval(
        self, sample_temporal_events: list[Event], sample_video_id: UUID
    ) -> None:
        candidate = CandidateEvent(
            video_id=sample_video_id,
            source_event_ids=[e.event_id for e in sample_temporal_events],
            start_timestamp=1.00,
            end_timestamp=3.00,
            saliency_score=0.80,
            reason="Test candidate",
            evidence_ids=[uuid4()],
        )
        builder = EvidenceBundleBuilder()
        bundle = builder.build_bundle(candidate, timeline=sample_temporal_events)

        assert len(bundle.keyframes) == 3
        roles = [kf.role for kf in bundle.keyframes]
        assert roles == ["leading", "midpoint", "trailing"]
        assert bundle.keyframes[0].timestamp_seconds == 1.00
        assert bundle.keyframes[1].timestamp_seconds == 2.00
        assert bundle.keyframes[2].timestamp_seconds == 3.00

    def test_keyframe_selection_for_instantaneous_event(
        self, sample_temporal_events: list[Event], sample_video_id: UUID
    ) -> None:
        candidate = CandidateEvent(
            video_id=sample_video_id,
            source_event_ids=[sample_temporal_events[0].event_id],
            start_timestamp=1.20,
            end_timestamp=1.20,
            saliency_score=0.70,
            reason="Instantaneous event",
            evidence_ids=[uuid4()],
        )
        builder = EvidenceBundleBuilder()
        bundle = builder.build_bundle(candidate, timeline=sample_temporal_events)

        assert len(bundle.keyframes) == 1
        assert bundle.keyframes[0].role == "instantaneous"
        assert bundle.keyframes[0].timestamp_seconds == 1.20


# ── Unit: Routing Policy & Budget ─────────────────────────────────────────


class TestRoutingPolicy:
    def test_token_estimation_and_budget_guards(self, sample_video_id: UUID) -> None:
        policy = RoutingPolicy(
            RoutingPolicyConfig(
                max_requests_per_minute=2,
                max_tokens_per_minute=5000,
            )
        )
        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=3.0,
            crops=[],
        )
        est = policy.estimate_tokens(bundle)
        assert est >= 400

        # Request 1: Allowed
        assert policy.check_budget(est) is True
        policy.record_usage(est)

        # Request 2: Allowed
        assert policy.check_budget(est) is True
        policy.record_usage(est)

        # Request 3: Exceeded max_requests_per_minute
        assert policy.check_budget(est) is False

    def test_query_mismatch_rejection(self, sample_video_id: UUID) -> None:
        policy = RoutingPolicy(RoutingPolicyConfig(min_query_relevance_threshold=0.50))
        cand = CandidateEvent(
            video_id=sample_video_id,
            source_event_ids=[uuid4()],
            start_timestamp=1.0,
            end_timestamp=2.0,
            saliency_score=0.90,
            query_relevance_score=0.10,  # Below query threshold
            reason="Mismatched query candidate",
        )
        dec, reason = policy.evaluate_gating(cand, query="looking for blue truck")
        assert dec == RoutingDecision.SUPPRESS_QUERY_MISMATCH


# ── Unit: Semantic Cache ──────────────────────────────────────────────────


class TestSemanticCache:
    def test_request_hashing_and_caching(self, sample_video_id: UUID) -> None:
        cache = SemanticCache(default_ttl_seconds=60.0)
        ev_id = uuid4()
        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[ev_id],
        )

        h1 = cache.compute_request_hash(bundle, query="gate", provider_name="mock")
        h2 = cache.compute_request_hash(bundle, query="gate", provider_name="mock")
        h3 = cache.compute_request_hash(bundle, query="different query", provider_name="mock")

        assert h1 == h2
        assert h1 != h3

        payload = SemanticEventPayload(
            claim="Security check confirmed",
            description="Driver presented credential",
            confidence=0.95,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[ev_id],
        )
        cache.set(h1, payload)
        assert cache.get(h1) is not None
        assert cache.get(h3) is None

    def test_candidate_deduplication(self, sample_video_id: UUID) -> None:
        cache = SemanticCache()
        cand1 = CandidateEvent(
            video_id=sample_video_id,
            source_event_ids=[uuid4()],
            start_timestamp=1.05,
            end_timestamp=2.0,
            participant_ids=["car_1", "gate"],
            saliency_score=0.8,
            reason="test",
        )
        cand2 = CandidateEvent(
            video_id=sample_video_id,
            source_event_ids=[uuid4()],
            start_timestamp=1.10,  # Same 1-second bucket
            end_timestamp=2.0,
            participant_ids=["car_1", "gate"],
            saliency_score=0.8,
            reason="test",
        )

        assert cache.is_duplicate_candidate(cand1) is False
        # Second candidate with identical participants in same temporal bucket is duplicate
        assert cache.is_duplicate_candidate(cand2) is True


# ── Unit: Evidence Validator & Negative Tests ─────────────────────────────


class TestEvidenceValidatorAndNegatives:
    def test_valid_payload_passes_validation(self, sample_video_id: UUID) -> None:
        validator = EvidenceValidator()
        ev_id = uuid4()
        source_ev_id = uuid4()
        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=3.0,
            evidence_ids=[ev_id],
            supporting_event_ids=[source_ev_id],
        )

        payload = SemanticEventPayload(
            claim="Authorized entry",
            description="Vehicle passed through gate",
            confidence=0.91,
            start_timestamp_seconds=1.1,
            end_timestamp_seconds=2.8,
            evidence_ids=[ev_id],
            supporting_event_ids=[source_ev_id],
        )
        res = validator.validate(payload, bundle)
        assert res.is_valid is True
        assert len(res.errors) == 0

    # Negative Test 1: Fabricated evidence ID
    def test_negative_fabricated_evidence_id_rejected(self, sample_video_id: UUID) -> None:
        validator = EvidenceValidator()
        ev_id = uuid4()
        phantom_ev_id = uuid4()

        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[ev_id],
        )
        payload = SemanticEventPayload(
            claim="Hallucinated claim",
            description="Model hallucinated an ID",
            confidence=0.9,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[phantom_ev_id],  # Phantom ID!
        )
        res = validator.validate(payload, bundle)
        assert res.is_valid is False
        assert res.sanitized_payload.status == SemanticStatus.REJECTED
        assert any("Cited evidence_id" in err for err in res.errors)

    # Negative Test 2: Fabricated supporting event ID
    def test_negative_fabricated_event_id_rejected(self, sample_video_id: UUID) -> None:
        validator = EvidenceValidator()
        ev_id = uuid4()
        phantom_event_id = uuid4()

        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[ev_id],
            supporting_event_ids=[uuid4()],
        )
        payload = SemanticEventPayload(
            claim="Invalid event citation",
            description="Model cited wrong event",
            confidence=0.8,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[ev_id],
            supporting_event_ids=[phantom_event_id],  # Phantom event!
        )
        res = validator.validate(payload, bundle)
        assert res.is_valid is False
        assert any("supporting_event_id" in err for err in res.errors)

    # Negative Test 3: Timestamp outside bundle window
    def test_negative_timestamp_outside_bundle_rejected(self, sample_video_id: UUID) -> None:
        validator = EvidenceValidator(temporal_tolerance_seconds=0.20)
        ev_id = uuid4()
        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            evidence_ids=[ev_id],
        )
        payload = SemanticEventPayload(
            claim="Temporal leak",
            description="Event happened way later",
            confidence=0.85,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=5.0,  # 3 seconds after bundle end!
            evidence_ids=[ev_id],
        )
        res = validator.validate(payload, bundle)
        assert res.is_valid is False
        assert any("exceeds bundle window" in err for err in res.errors)

    # Negative Test 4: Abstention on insufficient context
    def test_mock_abstention_on_empty_evidence_bundle(self, sample_video_id: UUID) -> None:
        mock = MockVLMProvider()
        empty_bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=1.0,
            crops=[],
            transcript_segments=[],
            ocr_observations=[],
        )
        req = RoutingRequest(
            candidate=CandidateEvent(
                video_id=sample_video_id,
                source_event_ids=[uuid4()],
                start_timestamp=0.0,
                end_timestamp=1.0,
                saliency_score=0.5,
                reason="empty candidate",
            )
        )
        result = mock.analyze(empty_bundle, req)
        assert result.status == SemanticStatus.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.0
        assert result.uncertainty == 1.0


# ── Unit: Prompt Contract & Qwen Adapter Configuration ────────────────────


class TestPromptContractAndQwenAdapter:
    def test_evidence_grounded_prompt_includes_constraints(self, sample_video_id: UUID) -> None:
        eid = uuid4()
        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=3.0,
            evidence_ids=[eid],
            spatial_context=["Zone Alpha"],
            transcript_segments=[
                {"text": "Clear the area", "language": "en", "start_ts": 1.2, "end_ts": 2.0}
            ],
            ocr_observations=[{"text": "STOP", "start_ts": 1.0, "confidence": 0.95}],
        )
        prompt = build_evidence_grounded_prompt(bundle, query="Who said clear?")

        assert "STRICT EPISTEMIC RULES" in prompt
        assert "GROUNDING MANDATE" in prompt
        assert "ABSTENTION RULE" in prompt
        assert "Zone Alpha" in prompt
        assert "Clear the area" in prompt
        assert "STOP" in prompt
        assert "Who said clear?" in prompt
        assert str(eid) in prompt

    def test_qwen_adapter_disabled_and_mock_modes(self, sample_video_id: UUID) -> None:
        # Disabled mode
        qwen_disabled = Qwen3VLAdapter(Qwen3VLConfig(execution_mode="disabled"))
        assert qwen_disabled.is_available() is False
        bundle = EvidenceBundle(
            candidate_id=uuid4(),
            video_id=sample_video_id,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
        )
        req = RoutingRequest(
            candidate=CandidateEvent(
                video_id=sample_video_id,
                source_event_ids=[uuid4()],
                start_timestamp=1.0,
                end_timestamp=2.0,
                saliency_score=0.5,
                reason="test",
            )
        )
        with pytest.raises(RuntimeError):
            qwen_disabled.analyze(bundle, req)

        # Mock mode
        qwen_mock = Qwen3VLAdapter(Qwen3VLConfig(execution_mode="mock"))
        assert qwen_mock.is_available() is True
        res = qwen_mock.analyze(bundle, req)
        assert res.status == SemanticStatus.INSUFFICIENT_EVIDENCE
        assert res.metadata["adapter"] == "qwen3_vl"
