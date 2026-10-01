"""Integration acceptance tests for VIDEX Phase 6.0 Semantic Intelligence Foundation.

Validates the full architectural sequence:
EventTimeline
    ↓
Candidate Semantic Event Selection
    ↓
Evidence Bundle Construction
    ↓
Semantic Router
    ↓
VLMProvider (MockVLMProvider)
    ↓
Structured Semantic Result
    ↓
Evidence Validation
    ↓
Evidence-backed Semantic Event
"""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from videx.domain.schemas import BoundingBox, CoordinateType, Evidence, EvidenceType, FrameTimestamp
from videx.events.engine import EventTimeline
from videx.events.schemas import Event, EventEvidence, EventParticipant
from videx.events.types import EventSeverity, EventStatus, EventType
from videx.semantic.cache import SemanticCache
from videx.semantic.candidates import CandidateSelectionConfig, CandidateSelector
from videx.semantic.mock import MockVLMProvider
from videx.semantic.policy import RoutingPolicy, RoutingPolicyConfig
from videx.semantic.router import SemanticRouter
from videx.semantic.types import RoutingDecision, RoutingPriority, SemanticStatus
from videx.semantic.validator import EvidenceValidator


@pytest.fixture
def test_video_id() -> UUID:
    return uuid4()


@pytest.fixture
def synthetic_frame_store() -> dict[int, tuple[np.ndarray, float]]:
    """Synthetic frame store for video reader mocking: frame_idx -> (BGR image, pts_seconds)."""
    frames: dict[int, tuple[np.ndarray, float]] = {}
    fps = 10.0
    for idx in range(30):
        ts = idx / fps
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Put a synthetic pattern
        img[100:200, 150:250] = 200
        frames[idx] = (img, ts)
    return frames


class MockVideoReader:
    """Mock video reader conforming to VideoReader frame sampling."""

    def __init__(self, frame_store: dict[int, tuple[np.ndarray, float]]) -> None:
        self.frame_store = frame_store

    def read_frame_at_timestamp(self, ts_sec: float) -> tuple[object, bytes]:
        # Find closest frame
        closest_idx = min(
            self.frame_store.keys(),
            key=lambda i: abs(self.frame_store[i][1] - ts_sec),
        )
        img, actual_ts = self.frame_store[closest_idx]

        class _MockFrame:
            frame_number = closest_idx
            frame_timestamp = FrameTimestamp(
                pts=int(actual_ts * 1000),
                time_base="1/1000",
                seconds=actual_ts,
            )

        return _MockFrame(), b""


@pytest.fixture
def mock_timeline_with_events(test_video_id: UUID) -> tuple[EventTimeline, list[Evidence]]:
    ev1_id = uuid4()
    ev2_id = uuid4()
    evidence_visual = uuid4()
    evidence_audio = uuid4()

    canonical_evidence = [
        Evidence(
            evidence_id=evidence_visual,
            evidence_type=EvidenceType.TRACK,
            source_module="yolo_tracker",
            video_id=test_video_id,
            confidence=0.95,
            description="Vehicle detection in gate area",
            timestamp_seconds=1.0,
            bbox=BoundingBox(
                x=150.0, y=100.0, width=100.0, height=100.0, coordinate_type=CoordinateType.PIXEL
            ),
        ),
        Evidence(
            evidence_id=evidence_audio,
            evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
            source_module="whisper_asr",
            video_id=test_video_id,
            confidence=0.90,
            description="Audio utterance clearance",
            timestamp_seconds=1.2,
        ),
    ]

    event_1 = Event(
        event_id=ev1_id,
        video_id=test_video_id,
        event_type=EventType.OBJECT_ENTERED_ZONE,
        start_timestamp_seconds=0.8,
        end_timestamp_seconds=2.0,
        confidence=0.92,
        status=EventStatus.CONFIRMED,
        severity=EventSeverity.HIGH,
        description="Vehicle entered inspection gate",
        zone_name="Inspection Gate",
        evidence_ids=[evidence_visual],
        event_evidence=[
            EventEvidence(
                evidence_id=evidence_visual,
                timestamp_seconds=1.0,
                evidence_type=EvidenceType.TRACK,
                role="spatial_entry",
            )
        ],
        participants=[
            EventParticipant(
                participant_id=uuid4(),
                participant_type="track",
                role="vehicle",
                label="sedan",
            )
        ],
    )

    event_2 = Event(
        event_id=ev2_id,
        video_id=test_video_id,
        event_type=EventType.SPEECH_DETECTED,
        start_timestamp_seconds=1.1,
        end_timestamp_seconds=1.9,
        confidence=0.89,
        status=EventStatus.CONFIRMED,
        severity=EventSeverity.INFO,
        description="Audio statement: Clearance confirmed",
        evidence_ids=[evidence_audio],
        event_evidence=[
            EventEvidence(
                evidence_id=evidence_audio,
                timestamp_seconds=1.2,
                evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
                role="utterance",
            )
        ],
        participants=[
            EventParticipant(
                participant_id="speaker_guard",
                participant_type="speaker",
                role="guard",
                label="security guard",
            )
        ],
        attributes={"speech_text": "Clearance confirmed by guard"},
    )

    timeline = EventTimeline(video_id=test_video_id, events=[event_1, event_2])
    return timeline, canonical_evidence


class TestSemanticPipelineAcceptance:
    """End-to-end integration acceptance tests for Layer 4 & Layer 5 architecture."""

    def test_full_pipeline_to_evidence_backed_semantic_event(
        self,
        test_video_id: UUID,
        mock_timeline_with_events: tuple[EventTimeline, list[Evidence]],
        synthetic_frame_store: dict[int, tuple[np.ndarray, float]],
    ) -> None:
        """Verify complete path: Timeline -> Candidate -> Bundle -> VLM -> Validation -> Event."""
        timeline, canonical_evidence = mock_timeline_with_events

        # 1. Initialize complete router stack
        vlm_provider = MockVLMProvider(
            default_status=SemanticStatus.SUPPORTED,
            default_confidence=0.94,
            simulated_latency_ms=2.0,
        )
        validator = EvidenceValidator()
        cache = SemanticCache()
        policy = RoutingPolicy(RoutingPolicyConfig(min_saliency_threshold=0.30))
        candidate_selector = CandidateSelector(
            CandidateSelectionConfig(min_saliency_threshold=0.30)
        )

        router = SemanticRouter(
            candidate_selector=candidate_selector,
            policy=policy,
            cache=cache,
            provider=vlm_provider,
            validator=validator,
        )

        query = "Did the sedan enter the inspection gate?"

        # 2. Process timeline through semantic router
        results = router.route_timeline(
            timeline=timeline,
            query=query,
        )

        # 3. Verify routing decisions
        assert len(results) == 1
        decision_res, semantic_ev = results[0]
        assert decision_res.decision == RoutingDecision.DISPATCH_VLM
        assert decision_res.cache_hit is False
        assert decision_res.payload is not None
        assert decision_res.payload.status == SemanticStatus.SUPPORTED
        assert decision_res.telemetry["validation_status"] == "passed"

        # 4. Verify candidate explainability
        cand = router.candidate_selector.select_candidates(
            timeline, query=query, video_id=test_video_id
        )[0]
        assert cand.video_id == test_video_id
        assert cand.priority in (RoutingPriority.CRITICAL, RoutingPriority.HIGH)
        assert len(cand.source_event_ids) == 2
        assert "cross_modal_co_occurrence" in cand.reason

        # 5. Verify the generated canonical semantic event
        assert semantic_ev is not None
        assert semantic_ev.video_id == test_video_id
        assert semantic_ev.event_type == EventType.CUSTOM
        assert semantic_ev.attributes["semantic_status"] == SemanticStatus.SUPPORTED.value
        assert semantic_ev.confidence == decision_res.payload.confidence
        # Citing valid evidence IDs from source events
        for eid in semantic_ev.evidence_ids:
            assert eid in cand.evidence_ids

    def test_abstention_on_insufficient_evidence(
        self,
        test_video_id: UUID,
    ) -> None:
        """Ensure model abstains and does not create confirmed event when evidence is lacking."""
        # Timeline with a single instantaneous event lacking visual crops or OCR/audio details
        lone_event = Event(
            video_id=test_video_id,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=5.0,
            end_timestamp_seconds=5.0,
            confidence=0.50,
            severity=EventSeverity.INFO,
            description="Faint object flicker",
            evidence_ids=[],
        )
        timeline = EventTimeline(video_id=test_video_id, events=[lone_event])

        # Configure VLM to abstain when no evidence IDs are provided
        vlm_provider = MockVLMProvider(
            default_status=SemanticStatus.INSUFFICIENT_EVIDENCE,
            default_confidence=0.0,
        )
        router = SemanticRouter(
            candidate_selector=CandidateSelector(
                CandidateSelectionConfig(min_saliency_threshold=0.10)
            ),
            policy=RoutingPolicy(RoutingPolicyConfig(min_saliency_threshold=0.10)),
            provider=vlm_provider,
        )

        results = router.route_timeline(timeline=timeline)

        assert len(results) == 1
        decision_res, semantic_ev = results[0]
        assert decision_res.payload is not None
        assert decision_res.payload.status == SemanticStatus.INSUFFICIENT_EVIDENCE
        # Invariant 5: When status is INSUFFICIENT_EVIDENCE, semantic event is marked detected
        assert semantic_ev is not None
        assert semantic_ev.status == EventStatus.DETECTED
        assert semantic_ev.confidence == 0.0

    def test_semantic_cache_hit_prevents_duplicate_vlm_invocation(
        self,
        test_video_id: UUID,
        mock_timeline_with_events: tuple[EventTimeline, list[Evidence]],
    ) -> None:
        """Verify that identical candidate/bundle queries hit the local cache."""
        timeline, canonical_evidence = mock_timeline_with_events
        cache = SemanticCache()

        router = SemanticRouter(
            candidate_selector=CandidateSelector(CandidateSelectionConfig(min_saliency_threshold=0.30)),
            policy=RoutingPolicy(RoutingPolicyConfig(min_saliency_threshold=0.30)),
            cache=cache,
            provider=MockVLMProvider(),
        )

        query = "Did the sedan enter inspection gate?"

        # First run: cache miss
        res1 = router.route_timeline(timeline=timeline, query=query)
        assert res1[0][0].cache_hit is False

        # Second run with exact same request and timeline: cache hit
        res2 = router.route_timeline(timeline=timeline, query=query)
        assert res2[0][0].cache_hit is True
        assert res2[0][0].decision == RoutingDecision.SERVE_FROM_CACHE
        assert res2[0][0].payload == res1[0][0].payload
        assert cache.size() >= 1

    def test_budget_exhaustion_suppression(
        self,
        test_video_id: UUID,
        mock_timeline_with_events: tuple[EventTimeline, list[Evidence]],
    ) -> None:
        """Verify that token / rate limit budget policy cleanly suppresses candidates."""
        timeline, canonical_evidence = mock_timeline_with_events

        # Policy with strict token budget limit (1 token per min)
        policy = RoutingPolicy(
            RoutingPolicyConfig(
                min_saliency_threshold=0.30,
                max_tokens_per_minute=1,  # Any bundle will exceed 1 token
            )
        )

        router = SemanticRouter(
            candidate_selector=CandidateSelector(CandidateSelectionConfig(min_saliency_threshold=0.30)),
            policy=policy,
            provider=MockVLMProvider(),
        )

        results = router.route_timeline(timeline=timeline)

        assert len(results) == 1
        decision_res, semantic_ev = results[0]
        assert decision_res.decision == RoutingDecision.SUPPRESS_BUDGET_EXCEEDED
        assert "Sliding token budget exceeded" in decision_res.reason
        assert decision_res.payload is None
        assert semantic_ev is None
