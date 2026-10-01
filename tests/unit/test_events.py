"""Unit tests for VIDEX Temporal Event Intelligence Engine (Phase 5.0).

Tests domain schemas, deterministic detectors, spatial zones, temporal relations,
timeline ordering, and evidence provenance chains.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from videx.domain.schemas import BoundingBox, EvidenceType
from videx.events.audio import AudioEventDetector
from videx.events.engine import EventTimeline
from videx.events.evidence import EvidenceLinker
from videx.events.lifecycle import LifecycleEventDetector
from videx.events.mock import (
    create_mock_text_observation,
    create_mock_track,
    create_mock_trajectory,
    create_mock_transcript_segment,
    create_mock_zone,
)
from videx.events.movement import MovementEventDetector
from videx.events.ocr import OCREventDetector
from videx.events.schemas import (
    Event,
    EventEngineConfig,
    EventEvidence,
)
from videx.events.spatial import SpatialEventDetector
from videx.events.temporal import TemporalRelationEngine
from videx.events.types import EventType


class TestEventSchemas:
    """Test Event, EventParticipant, EventEvidence, and EventEngineConfig schemas."""

    def test_instantaneous_event(self) -> None:
        vid_id = uuid4()
        ev = Event(
            video_id=vid_id,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=5.0,
            end_timestamp_seconds=5.0,
            confidence=0.95,
        )
        assert ev.is_instantaneous is True
        assert ev.is_interval is False
        assert ev.duration == 0.0
        assert ev.start_timestamp == 5.0
        assert ev.end_timestamp == 5.0

    def test_interval_event(self) -> None:
        vid_id = uuid4()
        ev = Event(
            video_id=vid_id,
            event_type=EventType.OBJECT_PRESENT,
            start_timestamp_seconds=2.0,
            end_timestamp_seconds=7.5,
            confidence=0.88,
        )
        assert ev.is_instantaneous is False
        assert ev.is_interval is True
        assert ev.duration == pytest.approx(5.5)

    def test_none_end_timestamp_is_instantaneous(self) -> None:
        vid_id = uuid4()
        ev = Event(
            video_id=vid_id,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=3.2,
            end_timestamp_seconds=None,
            confidence=0.9,
        )
        assert ev.is_instantaneous is True
        assert ev.end_timestamp == 3.2
        assert ev.duration == 0.0

    def test_invalid_interval_rejected(self) -> None:
        vid_id = uuid4()
        with pytest.raises(ValidationError):
            Event(
                video_id=vid_id,
                event_type=EventType.OBJECT_PRESENT,
                start_timestamp_seconds=10.0,
                end_timestamp_seconds=5.0,  # Invalid: end < start
                confidence=0.8,
            )

    def test_track_id_sync_with_participants(self) -> None:
        vid_id = uuid4()
        trk_id = uuid4()
        ev = Event(
            video_id=vid_id,
            event_type=EventType.OBJECT_STARTED_MOVING,
            start_timestamp_seconds=1.0,
            confidence=0.9,
            track_ids=[trk_id],
        )
        assert len(ev.participants) == 1
        assert ev.participants[0].participant_id == trk_id
        assert ev.participants[0].participant_type == "track"

    def test_event_evidence_linking(self) -> None:
        vid_id = uuid4()
        trk_id = uuid4()
        ee = EventEvidence(
            timestamp_seconds=1.5,
            evidence_type=EvidenceType.TRACK,
            role="appearance",
            track_id=trk_id,
            provenance={"frame": 37},
        )
        ev = Event(
            video_id=vid_id,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=1.5,
            confidence=0.92,
            event_evidence=[ee],
        )
        assert len(ev.evidence_ids) == 1
        assert ev.evidence_ids[0] == ee.evidence_id
        assert ev.event_evidence[0].timestamp == 1.5

    def test_config_defaults_and_validation(self) -> None:
        cfg = EventEngineConfig()
        assert cfg.lifecycle_confirmation_threshold == 2
        assert cfg.disappearance_threshold_seconds == 1.0
        assert cfg.movement_velocity_threshold_px_s == 10.0
        assert cfg.direction_change_degrees_threshold == 45.0
        assert cfg.spatial_boundary_tolerance_px == 5.0
        assert cfg.spatial_debounce_interval_seconds == 0.5
        assert cfg.temporal_near_interval_seconds == 2.0


class TestLifecycleEventDetector:
    """Test object appearance and disappearance detection with noise filtering."""

    def test_continuous_track_emits_appeared_present_disappeared(self) -> None:
        detector = LifecycleEventDetector()
        trk = create_mock_track(start_frame=0, end_frame=24, start_time=0.0, end_time=1.0)
        events = detector.detect_events([trk])

        types = [e.event_type for e in events]
        assert EventType.OBJECT_APPEARED in types
        assert EventType.OBJECT_PRESENT in types
        assert EventType.OBJECT_DISAPPEARED in types

        # Check appearance
        app = next(e for e in events if e.event_type == EventType.OBJECT_APPEARED)
        assert app.start_timestamp_seconds == 0.0
        assert app.is_instantaneous is True
        assert app.track_ids == [trk.track_id]

        # Check disappearance
        dis = next(e for e in events if e.event_type == EventType.OBJECT_DISAPPEARED)
        assert dis.start_timestamp_seconds == 1.0
        assert dis.is_instantaneous is True

        # Check presence
        pres = next(e for e in events if e.event_type == EventType.OBJECT_PRESENT)
        assert pres.start_timestamp_seconds == 0.0
        assert pres.end_timestamp_seconds == 1.0
        assert pres.duration == pytest.approx(1.0)

    def test_single_frame_noise_filtered(self) -> None:
        detector = LifecycleEventDetector(EventEngineConfig(lifecycle_confirmation_threshold=2))
        noisy_trk = create_mock_track(start_frame=5, end_frame=5, start_time=0.2, end_time=0.2)
        events = detector.detect_events([noisy_trk])
        assert len(events) == 0

    def test_multiple_tracks_produce_discrete_events(self) -> None:
        detector = LifecycleEventDetector()
        trk1 = create_mock_track(class_name="car", start_time=0.0, end_time=1.0)
        trk2 = create_mock_track(class_name="person", start_time=2.0, end_time=4.0)
        events = detector.detect_events([trk1, trk2])

        car_events = [e for e in events if e.track_ids == [trk1.track_id]]
        person_events = [e for e in events if e.track_ids == [trk2.track_id]]
        assert len(car_events) == 3
        assert len(person_events) == 3


class TestMovementEventDetector:
    """Test movement transitions and direction changes in pixel space."""

    def test_started_and_stopped_moving(self) -> None:
        detector = MovementEventDetector(EventEngineConfig(movement_velocity_threshold_px_s=15.0))
        trk = create_mock_track(start_time=0.0, end_time=2.0)

        # Build trajectory: stationary (0-0.5s), moving fast (0.5-1.0s), stationary (1.0-1.5s)
        pts = []
        fps = 10.0
        # Phase 1: stationary at (100, 100) for 5 frames (0.0 to 0.4s)
        for i in range(5):
            pts.append(
                create_mock_trajectory(
                    trk.track_id, start_x=100.0, start_y=100.0, end_x=100.0, end_y=100.0, frames=1
                )[0]
            )
            pts[-1] = pts[-1].model_copy(update={"frame_number": i, "timestamp_seconds": i / fps})

        # Phase 2: fast movement (100, 100) to (200, 100) over 5 frames (dx=20px = 200px/s)
        for i in range(5, 10):
            x = 100.0 + (i - 4) * 20.0
            pts.append(
                create_mock_trajectory(
                    trk.track_id, start_x=x, start_y=100.0, end_x=x, end_y=100.0, frames=1
                )[0]
            )
            pts[-1] = pts[-1].model_copy(update={"frame_number": i, "timestamp_seconds": i / fps})

        # Phase 3: stationary at (200, 100) for 5 frames
        for i in range(10, 15):
            pts.append(
                create_mock_trajectory(
                    trk.track_id, start_x=200.0, start_y=100.0, end_x=200.0, end_y=100.0, frames=1
                )[0]
            )
            pts[-1] = pts[-1].model_copy(update={"frame_number": i, "timestamp_seconds": i / fps})

        events = detector.detect_events([trk], {trk.track_id: pts})
        types = [e.event_type for e in events]

        assert EventType.OBJECT_STARTED_MOVING in types
        assert EventType.OBJECT_STOPPED_MOVING in types

    def test_direction_change_detection(self) -> None:
        detector = MovementEventDetector(
            EventEngineConfig(
                movement_velocity_threshold_px_s=5.0,
                direction_change_degrees_threshold=40.0,
            )
        )
        trk = create_mock_track()
        pts = []
        fps = 10.0

        # Moving East: (0, 0) -> (100, 0) in 5 frames
        for i in range(5):
            x = i * 20.0
            p = create_mock_trajectory(
                trk.track_id, start_x=x, start_y=0.0, end_x=x, end_y=0.0, frames=1
            )[0]
            pts.append(p.model_copy(update={"frame_number": i, "timestamp_seconds": i / fps}))

        # Turn South 90°: (100, 0) -> (100, 100) in 5 frames
        for i in range(5, 10):
            y = (i - 4) * 20.0
            p = create_mock_trajectory(
                trk.track_id, start_x=100.0, start_y=y, end_x=100.0, end_y=y, frames=1
            )[0]
            pts.append(p.model_copy(update={"frame_number": i, "timestamp_seconds": i / fps}))

        events = detector.detect_events([trk], {trk.track_id: pts})
        dir_events = [e for e in events if e.event_type == EventType.OBJECT_CHANGED_DIRECTION]
        assert len(dir_events) >= 1
        ev = dir_events[0]
        assert "direction_cardinal" in ev.attributes
        assert ev.attributes["angular_deflection_degrees"] >= 40.0


class TestSpatialEventDetector:
    """Test spatial zones and boundary crossing events."""

    def test_zone_entry_and_exit(self) -> None:
        zone = create_mock_zone("gate_1", "Gate 1", x_min=100, y_min=100, x_max=200, y_max=200)
        detector = SpatialEventDetector(
            zones=[zone],
            config=EventEngineConfig(spatial_debounce_interval_seconds=0.2),
        )
        trk = create_mock_track()

        # Path enters zone at frame 5, leaves at frame 15
        pts = []
        for i in range(20):
            x = 50.0 + i * 10.0  # from 50 to 240 (crosses 100 at i=5, crosses 200 at i=15)
            y = 150.0
            p = create_mock_trajectory(
                trk.track_id, start_x=x, start_y=y, end_x=x, end_y=y, frames=1
            )[0]
            pts.append(p.model_copy(update={"frame_number": i, "timestamp_seconds": i * 0.1}))

        events = detector.detect_events([trk], {trk.track_id: pts})
        types = [e.event_type for e in events]

        assert EventType.OBJECT_ENTERED_ZONE in types
        assert EventType.OBJECT_EXITED_ZONE in types

        entry = next(e for e in events if e.event_type == EventType.OBJECT_ENTERED_ZONE)
        exit_ev = next(e for e in events if e.event_type == EventType.OBJECT_EXITED_ZONE)

        assert entry.zone_name == "Gate 1"
        assert exit_ev.zone_name == "Gate 1"
        assert exit_ev.start_timestamp_seconds > entry.start_timestamp_seconds


class TestOCREventDetector:
    """Test text appeared, disappeared, and text changed detection."""

    def test_text_appeared_and_disappeared(self) -> None:
        detector = OCREventDetector()
        obs = create_mock_text_observation(text="SPEED 40", start_time=1.0, end_time=2.5)
        events = detector.detect_events([obs])

        types = [e.event_type for e in events]
        assert EventType.TEXT_APPEARED in types
        assert EventType.TEXT_DISAPPEARED in types

        app = next(e for e in events if e.event_type == EventType.TEXT_APPEARED)
        assert app.attributes["text"] == "SPEED 40"
        assert app.start_timestamp_seconds == 1.0

        dis = next(e for e in events if e.event_type == EventType.TEXT_DISAPPEARED)
        assert dis.start_timestamp_seconds == 2.5

    def test_text_changed_event(self) -> None:
        detector = OCREventDetector()
        box = BoundingBox(x=100, y=100, width=80, height=20)
        obs1 = create_mock_text_observation(
            text="UP32AB1234", start_time=1.0, end_time=2.0, bbox=box
        )
        obs2 = create_mock_text_observation(
            text="UP32AB1235", start_time=2.1, end_time=3.0, bbox=box
        )

        events = detector.detect_events([obs1, obs2])
        changed_events = [e for e in events if e.event_type == EventType.TEXT_CHANGED]
        assert len(changed_events) == 1

        chg = changed_events[0]
        assert chg.attributes["previous_text"] == "UP32AB1234"
        assert chg.attributes["new_text"] == "UP32AB1235"
        assert len(chg.event_evidence) == 2  # links both observations


class TestAudioEventDetector:
    """Test speech start, end, and detected interval events."""

    def test_speech_detection_events(self) -> None:
        detector = AudioEventDetector()
        seg = create_mock_transcript_segment(text="Caution road work", start_time=3.0, end_time=5.5)
        events = detector.detect_events([seg])

        types = [e.event_type for e in events]
        assert EventType.SPEECH_STARTED in types
        assert EventType.SPEECH_DETECTED in types
        assert EventType.SPEECH_ENDED in types

        started = next(e for e in events if e.event_type == EventType.SPEECH_STARTED)
        assert started.start_timestamp_seconds == 3.0
        assert started.is_instantaneous is True

        detected = next(e for e in events if e.event_type == EventType.SPEECH_DETECTED)
        assert detected.start_timestamp_seconds == 3.0
        assert detected.end_timestamp_seconds == 5.5
        assert detected.duration == pytest.approx(2.5)
        assert detected.attributes["raw_text"] == "Caution road work"


class TestTemporalRelationEngine:
    """Test qualitative temporal relationships between events."""

    def test_before_and_after(self) -> None:
        vid = uuid4()
        e1 = Event(
            video_id=vid,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=2.0,
            confidence=0.9,
        )
        e2 = Event(
            video_id=vid,
            event_type=EventType.TEXT_APPEARED,
            start_timestamp_seconds=3.0,
            end_timestamp_seconds=4.0,
            confidence=0.9,
        )

        engine = TemporalRelationEngine()
        assert engine.is_before(e1, e2) is True
        assert engine.is_after(e2, e1) is True
        assert engine.is_overlaps(e1, e2) is False

    def test_overlaps_and_contains(self) -> None:
        vid = uuid4()
        e_container = Event(
            video_id=vid,
            event_type=EventType.OBJECT_PRESENT,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=10.0,
            confidence=0.9,
        )
        e_contained = Event(
            video_id=vid,
            event_type=EventType.SPEECH_DETECTED,
            start_timestamp_seconds=3.0,
            end_timestamp_seconds=5.0,
            confidence=0.9,
        )

        engine = TemporalRelationEngine()
        assert engine.is_contains(e_container, e_contained) is True
        assert engine.is_during(e_contained, e_container) is True
        assert engine.is_overlaps(e_container, e_contained) is True

    def test_near_in_time(self) -> None:
        vid = uuid4()
        e1 = Event(
            video_id=vid,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=1.5,
            confidence=0.9,
        )
        e2 = Event(
            video_id=vid,
            event_type=EventType.SPEECH_STARTED,
            start_timestamp_seconds=2.2,
            end_timestamp_seconds=2.2,
            confidence=0.9,
        )

        engine = TemporalRelationEngine(near_threshold_seconds=1.0)
        assert engine.is_near_in_time(e1, e2) is True
        assert engine.is_near_in_time(e1, e2, max_delta=0.5) is False


class TestEventTimelineAndEvidenceLinker:
    """Test EventTimeline ordering, interval search, and 6-part explanations."""

    def test_timeline_sorting(self) -> None:
        vid = uuid4()
        e3 = Event(
            video_id=vid,
            event_type=EventType.OBJECT_DISAPPEARED,
            start_timestamp_seconds=9.0,
            confidence=0.9,
        )
        e1 = Event(
            video_id=vid,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=1.0,
            confidence=0.9,
        )
        e2 = Event(
            video_id=vid,
            event_type=EventType.SPEECH_DETECTED,
            start_timestamp_seconds=4.0,
            end_timestamp_seconds=6.0,
            confidence=0.9,
        )

        timeline = EventTimeline([e3, e1, e2], video_id=vid)
        assert [e.event_type for e in timeline.events] == [
            EventType.OBJECT_APPEARED,
            EventType.SPEECH_DETECTED,
            EventType.OBJECT_DISAPPEARED,
        ]

    def test_events_between(self) -> None:
        vid = uuid4()
        e1 = Event(
            video_id=vid,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=1.0,
            confidence=0.9,
        )
        e2 = Event(
            video_id=vid,
            event_type=EventType.OBJECT_PRESENT,
            start_timestamp_seconds=2.0,
            end_timestamp_seconds=8.0,
            confidence=0.9,
        )
        e3 = Event(
            video_id=vid,
            event_type=EventType.SPEECH_DETECTED,
            start_timestamp_seconds=10.0,
            end_timestamp_seconds=12.0,
            confidence=0.9,
        )

        timeline = EventTimeline([e1, e2, e3], video_id=vid)
        in_range = timeline.events_between(3.0, 7.0)
        assert len(in_range) == 1
        assert in_range[0].event_type == EventType.OBJECT_PRESENT

    def test_six_part_evidence_explanation(self) -> None:
        vid = uuid4()
        trk_id = uuid4()
        ee = EventEvidence(
            timestamp_seconds=4.2,
            evidence_type=EvidenceType.TRACK,
            role="appearance",
            track_id=trk_id,
            provenance={"frame_number": 105, "class_name": "car"},
        )
        ev = Event(
            video_id=vid,
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=4.2,
            confidence=0.95,
            event_evidence=[ee],
            description="Car entered the lane",
        )

        explanation = EvidenceLinker.explain_event(ev)

        # Verify all 6 fundamental questions are answered
        assert "what" in explanation
        assert explanation["what"]["event_type"] == "object_appeared"
        assert explanation["what"]["description"] == "Car entered the lane"

        assert "when" in explanation
        assert explanation["when"]["start_seconds"] == 4.2
        assert explanation["when"]["is_instantaneous"] is True

        assert "which_entities" in explanation

        assert "supporting_evidence" in explanation
        assert len(explanation["supporting_evidence"]["records"]) == 1

        assert "temporal_anchors" in explanation
        assert explanation["temporal_anchors"]["supporting_frames"] == [105]

        assert "source_subsystem" in explanation
