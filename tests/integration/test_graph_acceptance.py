"""Integration acceptance tests for Evidence Graph projection, query contracts, and integrity."""

from __future__ import annotations

from uuid import uuid4

from videx.domain.schemas import (
    BoundingBox,
    CoordinateType,
    Detection,
    Evidence,
    EvidenceType,
    Frame,
    OCRObservation,
    Scene,
    Track,
    TranscriptSegment,
    Video,
)
from videx.events.schemas import Event, EventParticipant
from videx.events.spatial import SpatialZone
from videx.events.types import EventType
from videx.graph.projection import GraphProjection, GraphProjectionConfig
from videx.graph.query import GraphQueryService
from videx.graph.schemas import NodeFilter
from videx.graph.types import GraphNodeType
from videx.graph.validator import GraphIntegrityValidator
from videx.semantic.schemas import SemanticEventPayload
from videx.semantic.types import SemanticEventType, SemanticStatus


def test_full_pipeline_graph_projection_and_query_acceptance() -> None:
    """Acceptance test: projects all multimodal entities and validates queries & integrity."""
    vid = uuid4()
    video = Video(
        video_id=vid,
        source_path="Sample_Videos/sample.mp4",
        duration_seconds=12.0,
        fps=25.0,
        width=1280,
        height=720,
    )

    sc1 = Scene(
        scene_id=uuid4(),
        video_id=vid,
        scene_index=0,
        start_frame_number=0,
        end_frame_number=150,
        start_timestamp_seconds=0.0,
        end_timestamp_seconds=6.0,
    )
    sc2 = Scene(
        scene_id=uuid4(),
        video_id=vid,
        scene_index=1,
        start_frame_number=151,
        end_frame_number=300,
        start_timestamp_seconds=6.0,
        end_timestamp_seconds=12.0,
    )

    f1 = Frame(
        frame_id=uuid4(),
        video_id=vid,
        scene_id=sc1.scene_id,
        frame_number=25,
        timestamp_seconds=1.0,
        width=1280,
        height=720,
    )
    f2 = Frame(
        frame_id=uuid4(),
        video_id=vid,
        scene_id=sc1.scene_id,
        frame_number=50,
        timestamp_seconds=2.0,
        width=1280,
        height=720,
    )

    b1 = BoundingBox(x=100, y=100, width=80, height=180, coordinate_type=CoordinateType.PIXEL)
    b2 = BoundingBox(x=120, y=110, width=70, height=170, coordinate_type=CoordinateType.PIXEL)

    trk1_id = uuid4()
    det1 = Detection(
        detection_id=uuid4(),
        frame_id=f1.frame_id,
        video_id=vid,
        frame_number=25,
        timestamp_seconds=1.0,
        class_name="person",
        class_id=0,
        confidence=0.92,
        bbox=b1,
        provider="yolo26",
    )
    det2 = Detection(
        detection_id=uuid4(),
        frame_id=f2.frame_id,
        video_id=vid,
        frame_number=50,
        timestamp_seconds=2.0,
        class_name="person",
        class_id=0,
        confidence=0.91,
        bbox=b2,
        provider="yolo26",
    )

    trk1 = Track(
        track_id=trk1_id,
        video_id=vid,
        class_name="person",
        first_seen_frame_number=25,
        last_seen_frame_number=50,
        first_seen_timestamp_seconds=1.0,
        last_seen_timestamp_seconds=2.0,
        start_bbox=b1,
        end_bbox=b2,
        detection_ids=[det1.detection_id, det2.detection_id],
        provider="bytetrack",
    )

    zone = SpatialZone.from_rectangle("dock_zone", "Loading Dock", 50, 50, 400, 400)

    ocr = OCRObservation(
        observation_id=uuid4(),
        frame_id=f1.frame_id,
        video_id=vid,
        frame_number=25,
        timestamp_seconds=1.0,
        text="SECURITY",
        normalized_text="security",
        confidence=0.97,
        bbox=BoundingBox(x=50, y=50, width=100, height=30, coordinate_type=CoordinateType.PIXEL),
        provider="paddleocr",
    )

    asr = TranscriptSegment(
        segment_id=uuid4(),
        video_id=vid,
        start_timestamp_seconds=1.0,
        end_timestamp_seconds=3.0,
        raw_text="authorized personnel only",
        normalized_text="authorized personnel only",
        language="en",
        confidence=0.95,
        provider="faster_whisper",
    )

    evid1 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.DETECTION,
        source_module="yolo26",
        video_id=vid,
        frame_id=f1.frame_id,
        timestamp_seconds=1.0,
        confidence=0.92,
        description="Person detection evidence",
    )

    ev1_id = uuid4()
    ev1 = Event(
        event_id=ev1_id,
        video_id=vid,
        event_type=EventType.OBJECT_ENTERED_ZONE,
        start_timestamp_seconds=1.0,
        end_timestamp_seconds=2.0,
        confidence=0.95,
        description="Person entered loading dock",
        participants=[
            EventParticipant(
                participant_id=trk1_id,
                participant_type="track",
                role="subject",
                label="person",
            ),
            EventParticipant(
                participant_id="dock_zone",
                participant_type="zone",
                role="zone",
                label="Loading Dock",
            ),
        ],
        evidence_ids=[evid1.evidence_id],
    )

    sem_payload = SemanticEventPayload(
        semantic_event_type=SemanticEventType.GENERAL_ACTIVITY,
        claim="Authorized personnel inspected dock zone",
        description="Person in uniform entered loading dock with badge matching security.",
        confidence=0.93,
        status=SemanticStatus.SUPPORTED,
        start_timestamp_seconds=1.0,
        end_timestamp_seconds=2.5,
        evidence_ids=[evid1.evidence_id],
        supporting_event_ids=[ev1_id],
    )

    # 1. Execute deterministic projection
    projection = GraphProjection(
        GraphProjectionConfig(
            near_threshold_seconds=2.0,
            iou_overlap_threshold=0.1,
            enable_temporal_relations=True,
            enable_spatial_relations=True,
            enable_zone_containment=True,
        )
    )

    store = projection.project(
        video=video,
        scenes=[sc1, sc2],
        frames=[f1, f2],
        detections=[det1, det2],
        tracks=[trk1],
        ocr_observations=[ocr],
        transcript_segments=[asr],
        zones=[zone],
        events=[ev1],
        semantic_events=[sem_payload],
        evidence=[evid1],
    )

    # 2. Validate Graph Integrity
    validator = GraphIntegrityValidator()
    report = validator.validate(store, video_id=vid)
    assert report.is_valid, f"Graph integrity errors: {report.errors}"
    assert report.total_nodes >= 10
    assert report.total_edges >= 10
    assert len(report.errors) == 0

    # 3. Query Service Contracts
    query = GraphQueryService(store)

    # Query events between 0.5s and 3.0s
    events_in_range = query.events_between(vid, 0.5, 3.0)
    assert len(events_in_range) >= 2  # Event + SemanticEvent

    # Query events involving track
    trk_events = query.events_involving_track(trk1_id)
    assert len(trk_events) == 1
    assert trk_events[0].source_id == str(ev1_id)

    # Query evidence grounding Event
    grounding = query.evidence_for_event(ev1_id)
    assert len(grounding) == 1
    assert grounding[0].source_id == str(evid1.evidence_id)

    # Trace complete evidence chain from Semantic Event
    sem_nodes = store.query_nodes(
        NodeFilter(video_id=vid, node_types=(GraphNodeType.SEMANTIC_EVENT,))
    )
    assert len(sem_nodes) == 1
    chain = query.trace_evidence_chain(sem_nodes[0].source_id)
    chain_types = [n.node_type for n in chain]
    assert GraphNodeType.SEMANTIC_EVENT in chain_types
    assert GraphNodeType.EVENT in chain_types
    assert GraphNodeType.EVIDENCE in chain_types
    assert GraphNodeType.FRAME in chain_types
