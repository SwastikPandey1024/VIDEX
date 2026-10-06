"""Mock investigator and fixture factory for deterministic testing and validation."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from videx.agent.answer import InvestigationResult
from videx.agent.investigator import Investigator
from videx.agent.schemas import InvestigationRequest
from videx.agent.tools import AgentToolContext
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
from videx.graph.projection import GraphProjection
from videx.graph.query import GraphQueryService
from videx.graph.store import InMemoryGraphStore


def create_mock_context(
    video_id: str | UUID = "00000000-0000-0000-0000-000000000001",
) -> AgentToolContext:
    """Creates a rich, canonical multimodal fixture context for agent testing."""
    vid = UUID(str(video_id))
    video = Video(
        video_id=vid,
        source_path="mock_surveillance.mp4",
        filename="mock_surveillance.mp4",
        duration_seconds=20.0,
        fps=30.0,
        width=1280,
        height=720,
    )

    # 1. Scenes
    sc1 = Scene(
        scene_id=uuid4(),
        video_id=vid,
        scene_index=0,
        start_frame_number=0,
        end_frame_number=150,
        start_timestamp_seconds=0.0,
        end_timestamp_seconds=5.0,
    )
    sc2 = Scene(
        scene_id=uuid4(),
        video_id=vid,
        scene_index=1,
        start_frame_number=151,
        end_frame_number=300,
        start_timestamp_seconds=5.0,
        end_timestamp_seconds=10.0,
    )
    scenes_dict = {str(s.scene_id): s for s in [sc1, sc2]}

    # 2. Frames
    f1 = Frame(
        frame_id=uuid4(),
        video_id=vid,
        scene_id=sc1.scene_id,
        frame_number=30,
        timestamp_seconds=1.0,
        width=1280,
        height=720,
    )
    f2 = Frame(
        frame_id=uuid4(),
        video_id=vid,
        scene_id=sc1.scene_id,
        frame_number=60,
        timestamp_seconds=2.0,
        width=1280,
        height=720,
    )
    f3 = Frame(
        frame_id=uuid4(),
        video_id=vid,
        scene_id=sc1.scene_id,
        frame_number=90,
        timestamp_seconds=3.0,
        width=1280,
        height=720,
    )
    frames_dict = {str(f.frame_id): f for f in [f1, f2, f3]}

    # 3. Bounding Boxes and Detections
    bbox_person = BoundingBox(
        x=100, y=100, width=80, height=180, coordinate_type=CoordinateType.PIXEL
    )
    bbox_car = BoundingBox(
        x=500, y=400, width=300, height=200, coordinate_type=CoordinateType.PIXEL
    )

    det1 = Detection(
        detection_id=uuid4(),
        frame_id=f1.frame_id,
        video_id=vid,
        frame_number=30,
        timestamp_seconds=1.0,
        class_name="person",
        class_id=0,
        confidence=0.96,
        bbox=bbox_person,
        provider="yolo26",
    )
    det2 = Detection(
        detection_id=uuid4(),
        frame_id=f2.frame_id,
        video_id=vid,
        frame_number=60,
        timestamp_seconds=2.0,
        class_name="person",
        class_id=0,
        confidence=0.94,
        bbox=bbox_person,
        provider="yolo26",
    )
    det3 = Detection(
        detection_id=uuid4(),
        frame_id=f2.frame_id,
        video_id=vid,
        frame_number=60,
        timestamp_seconds=2.0,
        class_name="car",
        class_id=1,
        confidence=0.98,
        bbox=bbox_car,
        provider="yolo26",
    )

    # 4. Tracks
    trk7_id = uuid4()
    trk12_id = uuid4()

    track7 = Track(
        track_id=trk7_id,
        video_id=vid,
        class_name="person",
        first_seen_frame_number=30,
        last_seen_frame_number=90,
        first_seen_timestamp_seconds=1.0,
        last_seen_timestamp_seconds=3.5,
        start_bbox=bbox_person,
        end_bbox=bbox_person,
        detection_ids=[det1.detection_id, det2.detection_id],
        provider="bytetrack",
    )
    track12 = Track(
        track_id=trk12_id,
        video_id=vid,
        class_name="car",
        first_seen_frame_number=60,
        last_seen_frame_number=150,
        first_seen_timestamp_seconds=2.0,
        last_seen_timestamp_seconds=6.0,
        start_bbox=bbox_car,
        end_bbox=bbox_car,
        detection_ids=[det3.detection_id],
        provider="bytetrack",
    )
    # Also support string lookup by numeric ID "7" and "12"
    tracks_dict = {
        str(track7.track_id): track7,
        "7": track7,
        str(track12.track_id): track12,
        "12": track12,
    }

    # 5. Evidence
    ev_det1 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.DETECTION,
        source_module="yolo26",
        video_id=vid,
        frame_id=f1.frame_id,
        timestamp_seconds=1.0,
        confidence=0.96,
        description="Person detection evidence",
    )
    ev_trk7 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.TRACK,
        source_module="bytetrack",
        video_id=vid,
        timestamp_seconds=1.0,
        confidence=0.95,
        description="Track 7 trajectory evidence",
    )
    ev_trk12 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.TRACK,
        source_module="bytetrack",
        video_id=vid,
        timestamp_seconds=2.0,
        confidence=0.98,
        description="Track 12 trajectory evidence",
    )
    ev_ocr1 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.OCR,
        source_module="paddleocr",
        video_id=vid,
        frame_id=f2.frame_id,
        timestamp_seconds=2.0,
        confidence=0.92,
        description="OCR observation: STOP",
    )
    ev_asr1 = Evidence(
        evidence_id=uuid4(),
        evidence_type=EvidenceType.AUDIO_TRANSCRIPT,
        source_module="whisper",
        video_id=vid,
        timestamp_seconds=1.5,
        confidence=0.89,
        description="Audio transcript evidence",
    )
    evidence_dict = {
        str(ev.evidence_id): ev
        for ev in [ev_det1, ev_trk7, ev_trk12, ev_ocr1, ev_asr1]
    }

    # 6. OCR Observation
    bbox_ocr = BoundingBox(x=50, y=50, width=100, height=30, coordinate_type=CoordinateType.PIXEL)
    ocr1 = OCRObservation(
        observation_id=uuid4(),
        frame_id=f2.frame_id,
        video_id=vid,
        frame_number=60,
        timestamp_seconds=2.0,
        text="STOP",
        normalized_text="stop",
        confidence=0.92,
        bbox=bbox_ocr,
        provider="paddleocr",
    )
    ocr_dict = {str(ocr1.observation_id): ocr1}

    # 7. Transcript Segment
    seg1 = TranscriptSegment(
        segment_id=uuid4(),
        video_id=vid,
        start_timestamp_seconds=1.5,
        end_timestamp_seconds=3.2,
        raw_text="Driver waiting at checkpoint",
        normalized_text="driver waiting at checkpoint",
        language="en",
        confidence=0.89,
        speaker_id="speaker_guard",
        provider="whisper",
    )
    transcript_dict = {str(seg1.segment_id): seg1}

    # 8. Events
    zone = SpatialZone.from_rectangle("checkpoint_zone", "Checkpoint Area", 50, 50, 600, 600)
    ev1_id = uuid4()
    ev2_id = uuid4()
    ev3_id = uuid4()

    event1 = Event(
        event_id=ev1_id,
        video_id=vid,
        event_type=EventType.OBJECT_ENTERED_ZONE,
        start_timestamp_seconds=1.0,
        end_timestamp_seconds=3.5,
        confidence=0.95,
        description="Track 7 entered checkpoint zone",
        participants=[
            EventParticipant(
                participant_id=trk7_id,
                participant_type="track",
                role="subject",
                label="person",
            )
        ],
        evidence_ids=[ev_det1.evidence_id, ev_trk7.evidence_id],
    )
    event2 = Event(
        event_id=ev2_id,
        video_id=vid,
        event_type=EventType.OBJECT_STATIONARY,
        start_timestamp_seconds=2.0,
        end_timestamp_seconds=6.0,
        confidence=0.97,
        description="Track 12 stopped at checkpoint",
        participants=[
            EventParticipant(
                participant_id=trk12_id,
                participant_type="track",
                role="subject",
                label="car",
            )
        ],
        evidence_ids=[ev_trk12.evidence_id],
    )
    event3 = Event(
        event_id=ev3_id,
        video_id=vid,
        event_type=EventType.CROWD_FORMATION,
        start_timestamp_seconds=2.0,
        end_timestamp_seconds=3.2,
        confidence=0.91,
        description="Track 7 approached Track 12",
        participants=[
            EventParticipant(
                participant_id=trk7_id,
                participant_type="track",
                role="subject",
                label="person",
            ),
            EventParticipant(
                participant_id=trk12_id,
                participant_type="track",
                role="object",
                label="car",
            ),
        ],
        evidence_ids=[ev_trk7.evidence_id, ev_trk12.evidence_id],
    )

    events_dict = {
        str(event1.event_id): event1,
        "ev-1": event1,
        str(event2.event_id): event2,
        "ev-2": event2,
        str(event3.event_id): event3,
        "ev-3": event3,
    }

    # 9. Graph Projection
    graph_store = InMemoryGraphStore()
    projection = GraphProjection()
    projection.project(
        video=video,
        scenes=[sc1, sc2],
        frames=[f1, f2, f3],
        detections=[det1, det2, det3],
        tracks=[track7, track12],
        ocr_observations=[ocr1],
        transcript_segments=[seg1],
        zones=[zone],
        events=[event1, event2, event3],
        evidence=[ev_det1, ev_trk7, ev_trk12, ev_ocr1, ev_asr1],
        store=graph_store,
    )
    graph_query_service = GraphQueryService(graph_store)

    # 10. Semantic Runner mock
    def mock_semantic_runner(query: str, cand_event_id: str | None = None) -> dict[str, Any]:
        q_lower = query.lower()
        if "interact" in q_lower and any(
            w in q_lower for w in ("person", "track", "vehicle", "7", "12")
        ):
            return {
                "answer": "Track 7 approached and stopped near Track 12 between 2.0s and 3.2s.",
                "confidence": 0.88,
                "evidence_ids": [str(ev_trk7.evidence_id), str(ev_trk12.evidence_id)],
                "event_ids": [str(ev3_id)],
            }
        return {
            "answer": "Inconclusive evidence: no interaction corroborated.",
            "confidence": 0.20,
            "evidence_ids": [],
            "event_ids": [],
        }

    return AgentToolContext(
        video_id=str(vid),
        graph_query_service=graph_query_service,
        evidence_records=evidence_dict,
        events=events_dict,
        tracks=tracks_dict,
        scenes=scenes_dict,
        frames=frames_dict,
        ocr_observations=ocr_dict,
        transcript_segments=transcript_dict,
        semantic_runner=mock_semantic_runner,
    )


class MockInvestigator(Investigator):
    """Investigator pre-configured with deterministic mock data for representative evaluations."""

    def __init__(self, video_id: str = "00000000-0000-0000-0000-000000000001") -> None:
        context = create_mock_context(video_id)
        super().__init__(context=context)

    def ask(self, question: str) -> InvestigationResult:
        """Convenience method to run an investigation for a natural language query."""
        req = InvestigationRequest.create(
            video_id=self.context.video_id,
            question=question,
        )
        return self.investigate(req)
