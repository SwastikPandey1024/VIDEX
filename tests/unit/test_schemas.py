"""Unit tests for domain schemas.

Tests cover:
- Valid construction of every domain type.
- Validation of field constraints (confidence bounds, positive dimensions).
- Computed properties (BoundingBox.area, Track.duration_seconds, etc.).
- JSON serialisation and deserialisation round-trips.
- EvidenceType and EventType enum values.
"""

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from videx.domain.schemas import (
    AudioSegment,
    BoundingBox,
    CoordinateType,
    Detection,
    Event,
    EventType,
    Evidence,
    EvidenceType,
    Frame,
    FrameTimestamp,
    OCRObservation,
    Scene,
    TimestampSource,
    Track,
    TrajectoryPoint,
    Video,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def make_bbox(x: float = 10.0, y: float = 20.0, w: float = 100.0, h: float = 50.0) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=w, height=h)


def make_video() -> Video:
    return Video(
        source_path="/data/test.mp4",
        fps=25.0,
        duration_seconds=120.0,
        width=1920,
        height=1080,
    )


def make_frame(video_id: UUID | None = None) -> Frame:
    vid_id = video_id or uuid4()
    return Frame(
        video_id=vid_id,
        frame_number=0,
        timestamp_seconds=0.0,
        width=1920,
        height=1080,
    )


def make_detection(frame: Frame) -> Detection:
    return Detection(
        frame_id=frame.frame_id,
        video_id=frame.video_id,
        timestamp_seconds=frame.timestamp_seconds,
        class_name="person",
        class_id=0,
        confidence=0.92,
        bbox=make_bbox(),
        provider="yolov8",
    )


# ── BoundingBox ────────────────────────────────────────────────────────────


class TestBoundingBox:
    def test_valid_pixel_box(self) -> None:
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.coordinate_type == CoordinateType.PIXEL

    def test_computed_properties(self) -> None:
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x2 == 110.0
        assert bbox.y2 == 70.0
        assert bbox.center_x == 60.0
        assert bbox.center_y == 45.0
        assert bbox.area == 5000.0

    def test_normalised_box(self) -> None:
        bbox = BoundingBox(
            x=0.1, y=0.2, width=0.5, height=0.3, coordinate_type=CoordinateType.NORMALIZED
        )
        assert bbox.coordinate_type == CoordinateType.NORMALIZED

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x=0, y=0, width=0, height=10)

    def test_negative_height_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x=0, y=0, width=10, height=-1)

    def test_frozen(self) -> None:
        bbox = make_bbox()
        field_name = "x"
        with pytest.raises((ValidationError, TypeError)):
            setattr(bbox, field_name, 999)

    def test_serialise_deserialise(self) -> None:
        bbox = make_bbox()
        data = bbox.model_dump()
        bbox2 = BoundingBox.model_validate(data)
        assert bbox == bbox2


# ── Video ──────────────────────────────────────────────────────────────────


class TestVideo:
    def test_auto_uuid(self) -> None:
        v = make_video()
        assert isinstance(v.video_id, UUID)

    def test_ingested_at_is_utc(self) -> None:
        v = make_video()
        assert v.ingested_at.tzinfo is not None

    def test_optional_fields(self) -> None:
        v = Video(source_path="http://example.com/stream")
        assert v.fps is None
        assert v.duration_seconds is None
        assert v.video_codec is None

    def test_json_roundtrip(self) -> None:
        v = make_video()
        dumped = v.model_dump_json()
        v2 = Video.model_validate_json(dumped)
        assert v.video_id == v2.video_id
        assert v.source_path == v2.source_path


# ── Scene ──────────────────────────────────────────────────────────────────


class TestScene:
    def test_duration(self) -> None:
        s = Scene(
            video_id=uuid4(),
            scene_index=0,
            start_frame_number=0,
            end_frame_number=49,
            start_timestamp_seconds=0.0,
            end_timestamp_seconds=2.0,
        )
        assert s.duration_seconds == pytest.approx(2.0)
        assert s.frame_count == 50


# ── Frame ──────────────────────────────────────────────────────────────────


class TestFrame:
    def test_auto_uuid(self) -> None:
        f = make_frame()
        assert isinstance(f.frame_id, UUID)

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Frame(video_id=uuid4(), frame_number=0, timestamp_seconds=-1.0, width=640, height=480)


# ── Detection ─────────────────────────────────────────────────────────────


class TestDetection:
    def test_valid_detection(self) -> None:
        frame = make_frame()
        det = make_detection(frame)
        assert det.frame_id == frame.frame_id
        assert det.class_name == "person"
        assert 0.0 <= det.confidence <= 1.0

    def test_confidence_above_1_rejected(self) -> None:
        frame = make_frame()
        with pytest.raises(ValidationError):
            Detection(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                timestamp_seconds=0.0,
                class_name="car",
                class_id=2,
                confidence=1.5,
                bbox=make_bbox(),
                provider="test",
            )

    def test_json_roundtrip(self) -> None:
        frame = make_frame()
        det = make_detection(frame)
        data = det.model_dump_json()
        det2 = Detection.model_validate_json(data)
        assert det.detection_id == det2.detection_id
        assert isinstance(det2.frame_timestamp, FrameTimestamp)
        assert det2.frame_timestamp == det.frame_timestamp

    def test_detection_frame_timestamp_provenance(self) -> None:
        frame = make_frame()
        ts = FrameTimestamp(
            frame_index=frame.frame_number,
            pts_seconds=frame.timestamp_seconds,
            timestamp_source=TimestampSource.CONTAINER,
            is_repaired=True,
            repair_reason="backward_pts",
            original_pts_seconds=0.04,
        )
        det = Detection(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_number=frame.frame_number,
            timestamp_seconds=frame.timestamp_seconds,
            frame_timestamp=ts,
            class_name="person",
            class_id=0,
            confidence=0.9,
            bbox=make_bbox(),
            provider="test",
        )
        assert det.frame_timestamp is ts
        assert det.frame_timestamp.is_repaired is True
        assert det.frame_timestamp.repair_reason == "backward_pts"
        assert det.frame_timestamp.original_pts_seconds == 0.04
        assert det.frame_timestamp.timestamp_source == TimestampSource.CONTAINER

        # Roundtrip JSON
        data = det.model_dump_json()
        det2 = Detection.model_validate_json(data)
        assert isinstance(det2.frame_timestamp, FrameTimestamp)
        assert det2.frame_timestamp.is_repaired is True
        assert det2.frame_timestamp.repair_reason == "backward_pts"
        assert det2.frame_timestamp.original_pts_seconds == 0.04
        assert det2.frame_timestamp.timestamp_source == TimestampSource.CONTAINER


# ── Track ──────────────────────────────────────────────────────────────────


class TestTrack:
    def test_duration_and_frame_count(self) -> None:
        vid_id = uuid4()
        track = Track(
            video_id=vid_id,
            class_name="vehicle",
            first_seen_frame_number=10,
            last_seen_frame_number=59,
            first_seen_timestamp_seconds=0.4,
            last_seen_timestamp_seconds=2.4,
            provider="bytetrack",
        )
        assert track.duration_seconds == pytest.approx(2.0)
        assert track.frame_count == 50


# ── TrajectoryPoint ────────────────────────────────────────────────────────


class TestTrajectoryPoint:
    def test_frozen(self) -> None:
        pt = TrajectoryPoint(
            track_id=uuid4(),
            frame_id=uuid4(),
            frame_number=5,
            timestamp_seconds=0.2,
            bbox=make_bbox(),
            confidence=0.88,
        )
        field_name = "frame_number"
        with pytest.raises((ValidationError, TypeError)):
            setattr(pt, field_name, 99)

    def test_trajectory_point_frame_timestamp_provenance(self) -> None:
        ts = FrameTimestamp(
            frame_index=5,
            pts_seconds=0.2,
            timestamp_source=TimestampSource.DERIVED,
            is_repaired=True,
            repair_reason="interpolated",
        )
        pt = TrajectoryPoint(
            track_id=uuid4(),
            frame_id=uuid4(),
            frame_number=5,
            timestamp_seconds=0.2,
            frame_timestamp=ts,
            bbox=make_bbox(),
            confidence=0.88,
        )
        assert pt.frame_timestamp is ts
        assert pt.frame_timestamp.is_repaired is True
        assert pt.frame_timestamp.repair_reason == "interpolated"
        assert pt.frame_timestamp.timestamp_source == TimestampSource.DERIVED

        data = pt.model_dump_json()
        pt2 = TrajectoryPoint.model_validate_json(data)
        assert isinstance(pt2.frame_timestamp, FrameTimestamp)
        assert pt2.frame_timestamp.is_repaired is True
        assert pt2.frame_timestamp.repair_reason == "interpolated"


# ── OCRObservation ─────────────────────────────────────────────────────────


class TestOCRObservation:
    def test_valid_ocr(self) -> None:
        obs = OCRObservation(
            frame_id=uuid4(),
            video_id=uuid4(),
            timestamp_seconds=1.5,
            text="नमस्ते",
            language="hi",
            confidence=0.85,
            bbox=make_bbox(),
            provider="easyocr",
        )
        assert obs.text == "नमस्ते"
        assert obs.language == "hi"

    def test_no_bbox_is_valid(self) -> None:
        obs = OCRObservation(
            frame_id=uuid4(),
            video_id=uuid4(),
            timestamp_seconds=0.0,
            text="Hello",
            confidence=0.9,
            provider="test",
        )
        assert obs.bbox is None


# ── AudioSegment ───────────────────────────────────────────────────────────


class TestAudioSegment:
    def test_duration(self) -> None:
        seg = AudioSegment(
            video_id=uuid4(),
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=3.5,
            transcript="नमस्ते दुनिया",
            confidence=0.78,
            provider="whisper",
        )
        assert seg.duration_seconds == pytest.approx(2.5)


# ── Event ──────────────────────────────────────────────────────────────────


class TestEvent:
    def test_all_event_types(self) -> None:
        for event_type in EventType:
            e = Event(
                video_id=uuid4(),
                event_type=event_type,
                start_timestamp_seconds=0.0,
                confidence=0.7,
                description=f"Test event: {event_type}",
            )
            assert e.event_type == event_type

    def test_detected_at_is_utc(self) -> None:
        e = Event(
            video_id=uuid4(),
            event_type=EventType.OBJECT_APPEARED,
            start_timestamp_seconds=5.0,
            confidence=0.95,
            description="Person appeared",
        )
        assert e.detected_at.tzinfo is not None


# ── Evidence ───────────────────────────────────────────────────────────────


class TestEvidence:
    def make_evidence(self, **overrides: object) -> Evidence:
        defaults: dict[str, object] = {
            "evidence_type": EvidenceType.DETECTION,
            "source_module": "yolov8",
            "video_id": uuid4(),
            "confidence": 0.9,
            "description": "Person detected at timestamp 1.0s",
        }
        defaults.update(overrides)
        return Evidence(**defaults)

    def test_minimal_valid(self) -> None:
        ev = self.make_evidence()
        assert isinstance(ev.evidence_id, UUID)
        assert ev.frame_id is None
        assert ev.track_id is None
        assert ev.bbox is None

    def test_full_evidence(self) -> None:
        vid_id = uuid4()
        frame_id = uuid4()
        track_id = uuid4()
        obs_id = uuid4()
        ev = Evidence(
            evidence_type=EvidenceType.TRACK,
            source_module="bytetrack",
            video_id=vid_id,
            frame_id=frame_id,
            timestamp_seconds=2.5,
            bbox=make_bbox(),
            track_id=track_id,
            confidence=0.88,
            description="Vehicle tracked for 10 seconds",
            raw_payload={"class": "car", "track_length_frames": 250},
            supporting_observation_ids=[obs_id],
            tags=["vehicle", "zone-A"],
        )
        assert ev.frame_id == frame_id
        assert ev.track_id == track_id
        assert len(ev.supporting_observation_ids) == 1
        assert "vehicle" in ev.tags

    def test_all_evidence_types(self) -> None:
        for ev_type in EvidenceType:
            ev = self.make_evidence(evidence_type=ev_type)
            assert ev.evidence_type == ev_type

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            self.make_evidence(confidence=1.1)
        with pytest.raises(ValidationError):
            self.make_evidence(confidence=-0.1)

    def test_json_serialisation(self) -> None:
        ev = self.make_evidence(
            frame_id=uuid4(),
            timestamp_seconds=3.14,
            bbox=make_bbox(),
        )
        json_str = ev.model_dump_json()
        data = json.loads(json_str)
        assert "evidence_id" in data
        assert "created_at" in data

    def test_deserialisation_roundtrip(self) -> None:
        ev = self.make_evidence(
            evidence_type=EvidenceType.OCR,
            source_module="easyocr_hi",
            frame_id=uuid4(),
            timestamp_seconds=7.2,
            bbox=make_bbox(x=50, y=100, w=200, h=30),
            confidence=0.76,
            description="Text: नमस्ते",
            raw_payload={"text": "नमस्ते", "language": "hi"},
            tags=["text", "hindi"],
        )
        json_str = ev.model_dump_json()
        ev2 = Evidence.model_validate_json(json_str)
        assert ev.evidence_id == ev2.evidence_id
        assert ev.evidence_type == ev2.evidence_type
        assert ev.raw_payload == ev2.raw_payload
        assert ev.tags == ev2.tags
        assert ev.bbox == ev2.bbox

    def test_created_at_is_utc(self) -> None:
        ev = self.make_evidence()
        assert ev.created_at.tzinfo is not None
