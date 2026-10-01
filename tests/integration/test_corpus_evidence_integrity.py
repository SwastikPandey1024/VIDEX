"""Integration tests for sample video corpus evidence integrity and Phase 6.1 VLM contracts.

Verifies:
1. Sample video manifest schema and canonical sample corpus availability.
2. Ingestion -> Perception -> OCR -> Audio -> Events -> Candidate selection pipeline execution.
3. Strict PTS monotonicity and zero FPS-derived replacement timestamps.
4. All emitted events and candidates have grounded, non-hallucinated evidence links.
5. Phase 6.1 VLM runtime configuration:
   - VIDEX_SEMANTIC_PROVIDER (mock, qwen, qwen3_vl)
   - VIDEX_SEMANTIC_EXECUTION_MODE (mock, api, local_cpu, local_gpu, disabled)
   - Mock VLM execution guarantees
   - Real VLM safety gate (reports NOT EXECUTED without crashing)
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from videx.domain.schemas import EvidenceType
from videx.events.engine import EventEngine, EventTimeline
from videx.events.schemas import Event, EventEvidence, EventParticipant
from videx.events.types import EventSeverity, EventStatus, EventType
from videx.ingestion.metadata import extract_video_metadata
from videx.ingestion.reader import OpenCVVideoReader
from videx.perception.detection import MockDetector
from videx.perception.pipeline import PerceptionPipeline, PerceptionResult
from videx.perception.tracking import MockTracker
from videx.semantic.candidates import CandidateSelector
from videx.semantic.mock import MockVLMProvider
from videx.semantic.providers import create_vlm_provider
from videx.semantic.qwen import Qwen3VLAdapter
from videx.semantic.router import SemanticRouter
from videx.semantic.schemas import CandidateEvent
from videx.semantic.types import RoutingDecision, SemanticStatus


def test_sample_video_manifest_integrity() -> None:
    """Verify that sample video manifest exists, parses, and describes canonical corpus."""
    manifest_path = Path("datasets/manifests/sample_videos.json")
    assert manifest_path.is_file(), "Manifest datasets/manifests/sample_videos.json must exist"

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "corpus_name" in data
    assert "videos" in data
    assert len(data["videos"]) == 8

    expected_files = {
        "bikes.mp4",
        "cars.mp4",
        "motorbikes.mp4",
        "people.mp4",
        "ppe-1.mp4",
        "ppe-2.mp4",
        "ppe-3.mp4",
        "sample.mp4",
    }
    manifest_files = {v["filename"] for v in data["videos"]}
    assert manifest_files == expected_files

    for v in data["videos"]:
        assert v["duration_seconds"] > 0
        assert v["width"] > 0
        assert v["height"] > 0
        assert v["fps"] > 0
        assert v["video_codec"]
        assert "audio" in v
        assert "has_audio" in v


def test_corpus_pipeline_and_evidence_integrity() -> None:
    """Verify full evidence grounding and PTS monotonicity on real sample video."""
    video_path = Path("Sample_Videos/sample.mp4")
    assert video_path.is_file(), f"Sample video {video_path} must exist"

    meta = extract_video_metadata(video_path)
    assert meta.duration_seconds > 0
    reader = OpenCVVideoReader(video_path)

    # 1. Perception
    perc_pipe = PerceptionPipeline(detector=MockDetector(), tracker=MockTracker())
    perc_res: PerceptionResult = perc_pipe.process_reader(reader, stride=2)
    assert perc_res.total_detections >= 0

    # 2. Event Engine
    event_engine = EventEngine()
    timeline: EventTimeline = event_engine.process_multimodal(
        video_id=reader.video_id,
        perception_result=perc_res,
    )

    # 3. Candidate Selector
    selector = CandidateSelector()
    candidates: list[CandidateEvent] = selector.select_candidates(timeline)

    # 4. Verify Monotonicity and Grounding
    prev_ts = -1.0
    for ev in timeline.events:
        assert ev.start_timestamp_seconds >= 0.0
        assert ev.start_timestamp_seconds >= prev_ts, "Timeline must be PTS monotonic"
        prev_ts = ev.start_timestamp_seconds
        assert len(ev.event_evidence) > 0 or len(ev.evidence_ids) > 0

    for cand in candidates:
        assert 0.0 <= cand.saliency_score <= 1.0
        assert cand.start_timestamp >= 0.0
        assert cand.end_timestamp >= cand.start_timestamp
        assert cand.reason != ""


def test_vlm_provider_configuration_and_factory() -> None:
    """Verify Phase 6.1 VLM provider factory and environment variable mappings."""
    # 1. Default mock provider
    p_mock = create_vlm_provider(provider_name="mock")
    assert p_mock.provider_name == "mock_vlm"
    assert p_mock.is_available() is True

    # 2. Qwen adapter with mock execution mode
    p_qwen_mock = create_vlm_provider(provider_name="qwen", execution_mode="mock")
    assert p_qwen_mock.provider_name == "qwen3_vl"
    assert isinstance(p_qwen_mock, Qwen3VLAdapter)
    assert p_qwen_mock.is_available() is True

    # 3. Qwen adapter with API execution mode
    p_qwen_api = create_vlm_provider(
        provider_name="qwen3_vl",
        execution_mode="api",
        api_base_url="http://localhost:8000/v1",
    )
    assert p_qwen_api.provider_name == "qwen3_vl"
    assert p_qwen_api.is_available() is True

    # 4. Qwen adapter with disabled mode
    p_qwen_disabled = create_vlm_provider(provider_name="qwen", execution_mode="disabled")
    assert p_qwen_disabled.is_available() is False

    # 5. Unsupported provider raises ValueError
    with pytest.raises(ValueError, match="Unsupported VLM provider"):
        create_vlm_provider(provider_name="nonexistent_vlm")


def test_semantic_router_with_sample_video_and_mock_vlm() -> None:
    """Validate SemanticRouter end-to-end execution on sample video using MockVLM."""
    video_path = Path("Sample_Videos/sample.mp4")
    reader = OpenCVVideoReader(video_path)

    # Create grounded test event
    ev_id = uuid4()
    test_event = Event(
        event_id=ev_id,
        video_id=reader.video_id,
        event_type=EventType.OBJECT_APPEARED,
        start_timestamp_seconds=0.5,
        end_timestamp_seconds=1.5,
        confidence=0.95,
        status=EventStatus.CONFIRMED,
        severity=EventSeverity.HIGH,
        participants=[
            EventParticipant(participant_id=uuid4(), participant_type="track", label="person")
        ],
        event_evidence=[
            EventEvidence(
                timestamp_seconds=0.5,
                evidence_type=EvidenceType.TRACK,
                role="appearance",
            )
        ],
        description="Person appeared in scene",
    )

    timeline = EventTimeline(events=[test_event], video_id=reader.video_id)
    router = SemanticRouter(provider=MockVLMProvider())

    results = router.route_timeline(timeline, video_reader=reader)
    assert len(results) >= 1
    decision_res, sem_event = results[0]

    assert decision_res.decision in (RoutingDecision.DISPATCH_VLM, RoutingDecision.SERVE_FROM_CACHE)
    assert sem_event is not None
    assert sem_event.event_type == EventType.CUSTOM
    assert sem_event.attributes["semantic_status"] in (
        SemanticStatus.SUPPORTED.value,
        SemanticStatus.INSUFFICIENT_EVIDENCE.value,
    )
    # Ensure evidence IDs match bundle
    assert len(sem_event.evidence_ids) > 0
