"""Unit and integration tests for VIDEX Phase 2.0 Perception (Detection & Tracking).

Covers:
- DetectionProvider and TrackingProvider runtime protocol compliance
- Detection schema: bounding boxes, confidence filtering, empty detection handling
- FrameTimestamp provenance inheritance (CONTAINER / DERIVED / FrameTimestamp preserved)
- MockDetector deterministic behavior
- MockTracker deterministic behavior
- Track lifecycle: ACTIVE -> LOST -> TERMINATED, first_seen and last_seen updates
- Trajectory calculation: centroid history, pixel displacement, path length,
  pixel velocity, and cardinal directions
- Stationary object trajectory analysis
- Evidence creation with traceability to frame index, timestamp, and track IDs
- Pipeline integration test on synthetic video stream
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from uuid import UUID, uuid4

import numpy as np
import pytest

from videx.domain.schemas import (
    BoundingBox,
    Detection,
    EvidenceType,
    Frame,
    TrackStatus,
    TrajectoryPoint,
)
from videx.ingestion.base import (
    DecodedFrame,
    FrameTimestamp,
    TimestampSource,
    TimingMode,
    VideoReader,
)
from videx.perception.detection import (
    MockDetector,
    YOLO26Detector,
    YOLO26DetectorConfig,
    _resolve_frame_input,
)
from videx.perception.pipeline import PerceptionPipeline
from videx.perception.tracking import (
    BoTSORTConfig,
    BoTSORTTracker,
    MockTracker,
)
from videx.perception.trajectory import TrajectoryAnalyzer
from videx.providers.base import DetectionProvider, TrackingProvider

# ── Synthetic Video Reader for Perception Testing ─────────────────────────────


class SyntheticPerceptionReader(VideoReader):
    """Generates synthetic DecodedFrame objects for deterministic testing."""

    def __init__(self, frame_count: int = 10, fps: float = 30.0) -> None:
        self._video_id = uuid4()
        self._frame_count = frame_count
        self._fps = fps
        self._is_open = True
        self._timing_mode = TimingMode.EXACT

    @property
    def video_id(self) -> UUID | None:
        return self._video_id

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._frame_count

    @property
    def duration_seconds(self) -> float:
        return self._frame_count / self._fps

    @property
    def width(self) -> int:
        return 640

    @property
    def height(self) -> int:
        return 480

    @property
    def timing_mode(self) -> TimingMode:
        return self._timing_mode

    @property
    def is_opened(self) -> bool:
        return self._is_open

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()

    def frames(self) -> Iterator[DecodedFrame]:
        for i in range(self._frame_count):
            yield self.read_decoded_frame(i)

    def read_frame(self, frame_index: int) -> tuple[Frame, bytes]:
        import cv2

        df = self.read_decoded_frame(frame_index)
        frame_meta = df.to_domain_frame()
        arr = df.frame_array
        assert isinstance(arr, np.ndarray)
        success, buf = cv2.imencode(".jpg", arr)
        if not success or buf is None:
            raise RuntimeError("Encoding failed")
        return frame_meta, bytes(buf)

    def read_frame_at_timestamp(self, timestamp_seconds: float) -> tuple[Frame, bytes]:
        f_idx = self.get_frame_for_timestamp(timestamp_seconds)
        return self.read_frame(f_idx)

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        return frame_index / self._fps

    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int:
        return min(max(0, int(round(timestamp_seconds * self._fps))), self._frame_count - 1)

    def get_frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        pts_sec = frame_index / self._fps
        return FrameTimestamp(
            frame_index=frame_index,
            pts_seconds=pts_sec,
            timestamp_source=TimestampSource.CONTAINER,
        )

    def read_decoded_frame(self, frame_index: int) -> DecodedFrame:
        if 0 <= frame_index < self._frame_count:
            pts_sec = frame_index / self._fps
            return DecodedFrame(
                frame_index=frame_index,
                timestamp_seconds=pts_sec,
                width=640,
                height=480,
                frame_array=np.zeros((480, 640, 3), dtype=np.uint8),
                video_id=self._video_id,
                frame_timestamp=FrameTimestamp(
                    frame_index=frame_index,
                    pts_seconds=pts_sec,
                    timestamp_source=TimestampSource.CONTAINER,
                ),
            )
        raise IndexError(f"Frame index {frame_index} out of bounds")

    def read_decoded_frame_at_timestamp(self, timestamp_seconds: float) -> DecodedFrame:
        f_idx = self.get_frame_for_timestamp(timestamp_seconds)
        return self.read_decoded_frame(f_idx)


# ── Tests: Provider Protocol Compliance ──────────────────────────────────────


def test_detection_provider_protocol_compliance() -> None:
    mock_det = MockDetector()
    assert isinstance(mock_det, DetectionProvider)
    assert hasattr(mock_det, "detect")
    assert hasattr(mock_det, "detect_batch")
    assert hasattr(mock_det, "warmup")
    assert mock_det.provider_name == "mock_detector"


def test_tracking_provider_protocol_compliance() -> None:
    mock_trk = MockTracker()
    assert isinstance(mock_trk, TrackingProvider)
    assert hasattr(mock_trk, "update")
    assert hasattr(mock_trk, "reset")
    assert hasattr(mock_trk, "active_tracks")
    assert mock_trk.provider_name == "mock_tracker"


# ── Tests: Detection & Timestamp Provenance ──────────────────────────────────


def test_mock_detector_deterministic_output() -> None:
    det = MockDetector(
        canned_detections={
            45: [
                {
                    "class_name": "person",
                    "class_id": 0,
                    "confidence": 0.92,
                    "x": 20.0,
                    "y": 30.0,
                    "width": 80.0,
                    "height": 160.0,
                }
            ]
        }
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ts = FrameTimestamp(
        frame_index=45,
        pts_seconds=1.5,
        timestamp_source=TimestampSource.CONTAINER,
    )
    frame = DecodedFrame(
        frame_index=45,
        timestamp_seconds=1.5,
        width=640,
        height=480,
        frame_array=img,
        frame_timestamp=ts,
    )

    detections = det.detect(frame)
    assert len(detections) == 1
    d = detections[0]

    assert d.class_name == "person"
    assert d.confidence == 0.92
    assert d.frame_number == 45
    assert d.timestamp_seconds == 1.5
    # Strict temporal provenance: inherited timestamp object preserved
    assert d.attributes.get("timestamp_source") == TimestampSource.CONTAINER.value
    assert d.attributes.get("mock") is True


def test_detector_confidence_filtering() -> None:
    det = MockDetector(
        confidence_threshold=0.50,
        canned_detections={
            10: [
                {
                    "class_name": "car",
                    "class_id": 2,
                    "confidence": 0.45,
                }
            ]
        },
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = DecodedFrame(
        frame_index=10,
        timestamp_seconds=0.33,
        width=640,
        height=480,
        frame_array=img,
        frame_timestamp=FrameTimestamp(
            frame_index=10,
            pts_seconds=0.33,
            timestamp_source=TimestampSource.DERIVED,
            is_repaired=True,
            repair_reason="interpolated",
        ),
    )

    results = det.detect(frame)
    assert len(results) == 0


def test_detector_empty_detections() -> None:
    det = MockDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = DecodedFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        width=640,
        height=480,
        frame_array=img,
        frame_timestamp=FrameTimestamp(
            frame_index=0,
            pts_seconds=0.0,
            timestamp_source=TimestampSource.CONTAINER,
        ),
    )
    detections = det.detect(frame)
    assert detections == []


def test_resolve_frame_input_decoded_frame() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    ts = FrameTimestamp(
        frame_index=50,
        pts_seconds=2.0,
        timestamp_source=TimestampSource.CONTAINER,
    )
    df = DecodedFrame(
        frame_index=50,
        timestamp_seconds=2.0,
        width=200,
        height=100,
        frame_array=img,
        frame_timestamp=ts,
    )

    out_img, meta, out_ts = _resolve_frame_input(df, None)
    assert out_img.shape == (100, 200, 3)
    assert meta.frame_number == 50
    assert meta.timestamp_seconds == 2.0
    assert out_ts is ts


def test_resolve_frame_input_invalid_type() -> None:
    with pytest.raises(TypeError, match="frame_data must be DecodedFrame or bytes"):
        _resolve_frame_input(12345, None)  # type: ignore[arg-type]


# ── Tests: Tracking & Lifecycle ──────────────────────────────────────────────


def test_mock_tracker_lifecycle() -> None:
    tracker = MockTracker(iou_threshold=0.3)
    vid_id = uuid4()
    frame_id_0 = uuid4()

    # Frame 0: detection observed
    d0 = Detection(
        video_id=vid_id,
        frame_id=frame_id_0,
        frame_number=0,
        timestamp_seconds=0.0,
        class_name="person",
        class_id=0,
        confidence=0.9,
        bbox=BoundingBox(x=10.0, y=10.0, width=50.0, height=100.0),
        provider="mock_detector",
    )
    meta0 = Frame(
        frame_id=frame_id_0,
        video_id=vid_id,
        frame_number=0,
        timestamp_seconds=0.0,
        width=640,
        height=480,
    )

    pts0 = tracker.update([d0], meta0)
    assert len(pts0) == 1
    active_0 = tracker.active_tracks()
    assert len(active_0) == 1
    t0 = active_0[0]
    assert t0.status == TrackStatus.ACTIVE
    assert t0.first_seen_frame_number == 0
    assert t0.last_seen_frame_number == 0
    assert t0.first_seen_timestamp_seconds == 0.0
    assert t0.last_seen_timestamp_seconds == 0.0

    # Frame 1: detection observed again with small displacement
    frame_id_1 = uuid4()
    d1 = Detection(
        video_id=vid_id,
        frame_id=frame_id_1,
        frame_number=1,
        timestamp_seconds=0.033,
        class_name="person",
        class_id=0,
        confidence=0.92,
        bbox=BoundingBox(x=12.0, y=12.0, width=50.0, height=100.0),
        provider="mock_detector",
    )
    meta1 = Frame(
        frame_id=frame_id_1,
        video_id=vid_id,
        frame_number=1,
        timestamp_seconds=0.033,
        width=640,
        height=480,
    )
    pts1 = tracker.update([d1], meta1)
    assert len(pts1) == 1
    active_1 = tracker.active_tracks()
    assert len(active_1) == 1
    t1 = active_1[0]
    assert t1.track_id == t0.track_id
    assert t1.status == TrackStatus.ACTIVE
    assert t1.first_seen_frame_number == 0
    assert t1.last_seen_frame_number == 1
    assert t1.last_seen_timestamp_seconds == 0.033
    assert len(t1.detection_ids) == 2

    # Frame 2: Missed detection -> marked LOST
    meta2 = Frame(
        video_id=vid_id,
        frame_number=2,
        timestamp_seconds=0.066,
        width=640,
        height=480,
    )
    tracker.update([], meta2)
    # Active tracks should now be 0 since it is LOST
    assert len(tracker.active_tracks()) == 0
    all_trks = tracker.all_tracks()
    assert len(all_trks) == 1
    assert all_trks[0].status == TrackStatus.LOST

    # Frames 3..8: Missed repeatedly -> exceeds max_buffer=5 -> TERMINATED
    for f in range(3, 9):
        meta = Frame(
            video_id=vid_id,
            frame_number=f,
            timestamp_seconds=f * 0.033,
            width=640,
            height=480,
        )
        tracker.update([], meta)

    all_trks = tracker.all_tracks()
    assert len(all_trks) == 1
    assert all_trks[0].status == TrackStatus.TERMINATED


def test_tracker_reset() -> None:
    tracker = MockTracker()
    vid_id = uuid4()
    d = Detection(
        video_id=vid_id,
        frame_id=uuid4(),
        frame_number=0,
        timestamp_seconds=0.0,
        class_name="dog",
        class_id=16,
        confidence=0.88,
        bbox=BoundingBox(x=5.0, y=5.0, width=20.0, height=20.0),
        provider="mock_detector",
    )
    meta = Frame(
        video_id=vid_id,
        frame_number=0,
        timestamp_seconds=0.0,
        width=640,
        height=480,
    )
    tracker.update([d], meta)
    assert len(tracker.all_tracks()) == 1

    tracker.reset()
    assert len(tracker.all_tracks()) == 0


# ── Tests: Trajectory Calculation ────────────────────────────────────────────


def test_trajectory_analyzer_displacement_and_velocity() -> None:
    # 3 points moving east (+x right) in pixel space
    tid = uuid4()
    pts = [
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=0,
            timestamp_seconds=0.0,
            bbox=BoundingBox(x=100.0, y=200.0, width=20.0, height=20.0),
            confidence=0.9,
        ),
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=15,
            timestamp_seconds=0.5,
            bbox=BoundingBox(x=140.0, y=200.0, width=20.0, height=20.0),
            confidence=0.9,
        ),
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=30,
            timestamp_seconds=1.0,
            bbox=BoundingBox(x=180.0, y=200.0, width=20.0, height=20.0),
            confidence=0.9,
        ),
    ]

    analysis = TrajectoryAnalyzer.analyze_trajectory(pts)
    assert analysis["point_count"] == 3
    assert math.isclose(analysis["total_displacement_px"], 80.0, rel_tol=1e-4)
    assert math.isclose(analysis["path_length_px"], 80.0, rel_tol=1e-4)
    assert math.isclose(analysis["duration_seconds"], 1.0, rel_tol=1e-4)
    assert math.isclose(analysis["mean_pixel_velocity"], 80.0, rel_tol=1e-4)
    assert analysis["direction_cardinal"] == "right"


def test_trajectory_analyzer_stationary() -> None:
    tid = uuid4()
    pts = [
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=0,
            timestamp_seconds=0.0,
            bbox=BoundingBox(x=50.0, y=50.0, width=20.0, height=20.0),
            confidence=0.9,
        ),
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=30,
            timestamp_seconds=1.0,
            bbox=BoundingBox(x=50.5, y=50.5, width=20.0, height=20.0),
            confidence=0.9,
        ),
    ]
    analysis = TrajectoryAnalyzer.analyze_trajectory(pts)
    assert analysis["direction_cardinal"] == "stationary"
    assert analysis["total_displacement_px"] < 2.0


def test_trajectory_analyzer_diagonal_direction() -> None:
    # Moving down and right (southeast / down-right)
    tid = uuid4()
    pts = [
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=0,
            timestamp_seconds=0.0,
            bbox=BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0),
            confidence=0.9,
        ),
        TrajectoryPoint(
            track_id=tid,
            frame_id=uuid4(),
            frame_number=60,
            timestamp_seconds=2.0,
            bbox=BoundingBox(x=100.0, y=100.0, width=10.0, height=10.0),
            confidence=0.9,
        ),
    ]
    analysis = TrajectoryAnalyzer.analyze_trajectory(pts)
    assert analysis["direction_cardinal"] == "down-right"
    assert math.isclose(analysis["total_displacement_px"], math.sqrt(20000), rel_tol=1e-2)


def test_trajectory_analyzer_empty() -> None:
    analysis = TrajectoryAnalyzer.analyze_trajectory([])
    assert analysis["point_count"] == 0
    assert analysis["direction_cardinal"] == "unknown"
    assert analysis["total_displacement_px"] == 0.0


# ── Tests: Pipeline Integration & Evidence Preservation ──────────────────────


def test_perception_pipeline_integration() -> None:
    reader = SyntheticPerceptionReader(frame_count=5, fps=25.0)
    canned = {
        i: [
            {
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.95,
                "x": 20.0 + i * 5.0,
                "y": 20.0 + i * 2.0,
                "width": 40.0,
                "height": 80.0,
            }
        ]
        for i in range(5)
    }
    detector = MockDetector(canned_detections=canned)
    tracker = MockTracker(iou_threshold=0.2)

    pipeline = PerceptionPipeline(
        detector=detector,
        tracker=tracker,
        min_track_length=2,
    )

    result = pipeline.process_reader(reader)

    assert result.total_detections == 5
    assert result.total_tracks >= 1
    assert len(result.evidence) >= 1

    # Check evidence traceability
    ev = result.evidence[0]
    assert ev.evidence_type == EvidenceType.TRACK
    assert ev.video_id == reader.video_id
    assert ev.track_id is not None
    assert ev.confidence >= 0.9
    assert ev.source_module == "mock_tracker_tracker"
    assert "trajectory" in ev.raw_payload

    # Verify timestamp provenance preserved through the whole chain
    det0 = result.detections[0]
    assert det0.attributes.get("timestamp_source") == TimestampSource.CONTAINER.value


def test_botsort_and_yolo26_config_defaults() -> None:
    y_cfg = YOLO26DetectorConfig()
    assert y_cfg.model_path == "yolo26n.pt"
    assert y_cfg.confidence_threshold == 0.25
    assert y_cfg.iou_threshold == 0.45
    assert y_cfg.device == "cpu"

    b_cfg = BoTSORTConfig()
    assert b_cfg.track_high_thresh == 0.5
    assert b_cfg.track_buffer == 30
    assert b_cfg.match_thresh == 0.8
    assert b_cfg.with_reid is False


def test_frame_timestamp_identity_and_provenance_propagation() -> None:
    """Verify FrameTimestamp object identity and provenance fields propagate

    without modification from DecodedFrame -> Detection -> TrajectoryPoint.
    """
    ts = FrameTimestamp(
        frame_index=7,
        pts_seconds=0.28,
        timestamp_source=TimestampSource.CONTAINER,
        is_repaired=False,
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = DecodedFrame(
        frame_index=7,
        timestamp_seconds=0.28,
        width=640,
        height=480,
        frame_array=img,
        frame_timestamp=ts,
    )

    det_provider = MockDetector(
        canned_detections={
            7: [
                {
                    "class_name": "car",
                    "class_id": 2,
                    "confidence": 0.94,
                    "x": 100.0,
                    "y": 150.0,
                    "width": 80.0,
                    "height": 40.0,
                }
            ]
        }
    )
    detections = det_provider.detect(frame)
    assert len(detections) == 1
    d = detections[0]
    # Authoritative FrameTimestamp preserved on Detection
    assert d.frame_timestamp is ts
    assert d.frame_timestamp.timestamp_source == TimestampSource.CONTAINER
    assert d.frame_timestamp.frame_index == 7
    assert d.frame_timestamp.pts_seconds == 0.28
    assert not d.frame_timestamp.is_repaired

    tracker = BoTSORTTracker()
    points = tracker.update(detections, frame)
    assert len(points) == 1
    pt = points[0]
    # Authoritative FrameTimestamp preserved on TrajectoryPoint
    assert pt.frame_timestamp is ts
    assert pt.frame_timestamp.timestamp_source == TimestampSource.CONTAINER
    assert pt.frame_timestamp.pts_seconds == 0.28


def test_repaired_timestamp_propagation() -> None:
    """Verify that repaired timestamp status and reasons remain available downstream

    without recomputation across Detection and TrajectoryPoint.
    """
    repaired_ts = FrameTimestamp(
        frame_index=15,
        pts_seconds=0.60,
        timestamp_source=TimestampSource.DERIVED,
        is_repaired=True,
        repair_reason="backward_pts",
        original_pts_seconds=0.52,
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = DecodedFrame(
        frame_index=15,
        timestamp_seconds=0.60,
        width=640,
        height=480,
        frame_array=img,
        frame_timestamp=repaired_ts,
    )

    detector = MockDetector(
        canned_detections={
            15: [
                {
                    "class_name": "truck",
                    "class_id": 7,
                    "confidence": 0.89,
                    "x": 50.0,
                    "y": 60.0,
                    "width": 120.0,
                    "height": 70.0,
                }
            ]
        }
    )
    dets = detector.detect(frame)
    assert len(dets) == 1
    d = dets[0]
    assert d.frame_timestamp.is_repaired is True
    assert d.frame_timestamp.repair_reason == "backward_pts"
    assert d.frame_timestamp.original_pts_seconds == 0.52
    assert d.frame_timestamp.timestamp_source == TimestampSource.DERIVED

    tracker = BoTSORTTracker()
    points = tracker.update(dets, frame)
    assert len(points) == 1
    pt = points[0]
    assert pt.frame_timestamp.is_repaired is True
    assert pt.frame_timestamp.repair_reason == "backward_pts"
    assert pt.frame_timestamp.original_pts_seconds == 0.52
    assert pt.frame_timestamp.timestamp_source == TimestampSource.DERIVED


def test_detection_to_track_timestamp_preservation() -> None:
    """Verify that when frame_meta does not supply frame_timestamp directly,

    the tracker extracts and preserves the FrameTimestamp from the detections.
    """
    ts = FrameTimestamp(
        frame_index=3,
        pts_seconds=0.12,
        timestamp_source=TimestampSource.CONTAINER,
    )
    vid_id = uuid4()
    f_id = uuid4()
    det = Detection(
        video_id=vid_id,
        frame_id=f_id,
        frame_number=3,
        timestamp_seconds=0.12,
        frame_timestamp=ts,
        class_name="person",
        class_id=0,
        confidence=0.91,
        bbox=BoundingBox(x=10.0, y=10.0, width=30.0, height=60.0),
        provider="mock",
    )
    # Plain Frame without explicit frame_timestamp passed to tracker
    plain_frame = Frame(
        video_id=vid_id,
        frame_id=f_id,
        frame_number=3,
        timestamp_seconds=0.12,
        width=640,
        height=480,
    )
    tracker = BoTSORTTracker()
    points = tracker.update([det], plain_frame)
    assert len(points) == 1
    assert points[0].frame_timestamp is ts
    assert points[0].frame_timestamp.timestamp_source == TimestampSource.CONTAINER


def test_empty_detections_handling() -> None:
    """Verify detector and tracker handle empty detections gracefully."""
    detector = MockDetector(canned_detections={})
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ts = FrameTimestamp(frame_index=0, pts_seconds=0.0, timestamp_source=TimestampSource.CONTAINER)
    df = DecodedFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        width=100,
        height=100,
        frame_array=img,
        frame_timestamp=ts,
    )

    dets = detector.detect(df)
    assert dets == []

    tracker = BoTSORTTracker()
    pts = tracker.update([], df)
    assert pts == []
    assert tracker.active_tracks() == []


def test_botsort_lifecycle_and_track_id_persistence() -> None:
    """Verify BoTSORTTracker lifecycle: ACTIVE -> LOST -> re-associated to same ID -> TERMINATED."""
    tracker = BoTSORTTracker(BoTSORTConfig(track_buffer=3, new_track_thresh=0.5))
    vid_id = uuid4()

    # Frame 0: High-confidence detection initiates track
    ts0 = FrameTimestamp(frame_index=0, pts_seconds=0.0, timestamp_source=TimestampSource.CONTAINER)
    d0 = Detection(
        video_id=vid_id,
        frame_id=uuid4(),
        frame_number=0,
        timestamp_seconds=0.0,
        frame_timestamp=ts0,
        class_name="person",
        class_id=0,
        confidence=0.90,
        bbox=BoundingBox(x=100.0, y=100.0, width=50.0, height=100.0),
        provider="yolo26",
    )
    meta0 = Frame(
        video_id=vid_id,
        frame_number=0,
        timestamp_seconds=0.0,
        frame_timestamp=ts0,
        width=640,
        height=480,
    )
    pts0 = tracker.update([d0], meta0)
    assert len(pts0) == 1
    tracks0 = tracker.active_tracks()
    assert len(tracks0) == 1
    track_id = tracks0[0].track_id
    assert tracks0[0].status == TrackStatus.ACTIVE

    # Frame 1: Detection moves slightly -> track persists with identical track_id
    ts1 = FrameTimestamp(
        frame_index=1, pts_seconds=0.04, timestamp_source=TimestampSource.CONTAINER
    )
    d1 = Detection(
        video_id=vid_id,
        frame_id=uuid4(),
        frame_number=1,
        timestamp_seconds=0.04,
        frame_timestamp=ts1,
        class_name="person",
        class_id=0,
        confidence=0.92,
        bbox=BoundingBox(x=105.0, y=100.0, width=50.0, height=100.0),
        provider="yolo26",
    )
    meta1 = Frame(
        video_id=vid_id,
        frame_number=1,
        timestamp_seconds=0.04,
        frame_timestamp=ts1,
        width=640,
        height=480,
    )
    pts1 = tracker.update([d1], meta1)
    assert len(pts1) == 1
    tracks1 = tracker.active_tracks()
    assert len(tracks1) == 1
    assert tracks1[0].track_id == track_id  # Persistence
    assert tracks1[0].first_seen_frame_number == 0
    assert tracks1[0].last_seen_frame_number == 1

    # Frame 2: Missed detection (occlusion) -> track marked LOST
    meta2 = Frame(video_id=vid_id, frame_number=2, timestamp_seconds=0.08, width=640, height=480)
    tracker.update([], meta2)
    assert len(tracker.active_tracks()) == 0
    all_trks = tracker.all_tracks()
    assert len(all_trks) == 1
    assert all_trks[0].status == TrackStatus.LOST

    # Frame 3: Object reappears at predicted position -> re-associated to SAME track_id!
    ts3 = FrameTimestamp(
        frame_index=3, pts_seconds=0.12, timestamp_source=TimestampSource.CONTAINER
    )
    d3 = Detection(
        video_id=vid_id,
        frame_id=uuid4(),
        frame_number=3,
        timestamp_seconds=0.12,
        frame_timestamp=ts3,
        class_name="person",
        class_id=0,
        confidence=0.88,
        bbox=BoundingBox(x=115.0, y=100.0, width=50.0, height=100.0),
        provider="yolo26",
    )
    meta3 = Frame(
        video_id=vid_id,
        frame_number=3,
        timestamp_seconds=0.12,
        frame_timestamp=ts3,
        width=640,
        height=480,
    )
    pts3 = tracker.update([d3], meta3)
    assert len(pts3) == 1
    tracks3 = tracker.active_tracks()
    assert len(tracks3) == 1
    assert tracks3[0].track_id == track_id  # Re-associated!
    assert tracks3[0].status == TrackStatus.ACTIVE

    # Frames 4..8: Missing repeatedly -> exceeds track_buffer (3 frames) -> TERMINATED
    for f in range(4, 9):
        meta = Frame(
            video_id=vid_id, frame_number=f, timestamp_seconds=f * 0.04, width=640, height=480
        )
        tracker.update([], meta)

    assert len(tracker.active_tracks()) == 0
    terminated = tracker.all_tracks()
    assert len(terminated) == 1
    assert terminated[0].status == TrackStatus.TERMINATED


def test_botsort_kalman_filter_motion_prediction() -> None:
    """Verify that Kalman filter in BoTSORTTracker predicts bounding box forward when unobserved."""
    tracker = BoTSORTTracker(BoTSORTConfig(track_buffer=5))
    vid_id = uuid4()

    # Frame 0: at x=100
    d0 = Detection(
        video_id=vid_id,
        frame_id=uuid4(),
        frame_number=0,
        timestamp_seconds=0.0,
        class_name="car",
        class_id=2,
        confidence=0.9,
        bbox=BoundingBox(x=100.0, y=50.0, width=40.0, height=40.0),
        provider="yolo26",
    )
    tracker.update(
        [d0], Frame(video_id=vid_id, frame_number=0, timestamp_seconds=0.0, width=640, height=480)
    )

    # Frame 1: at x=120 (velocity vx ~ 20 px/frame)
    d1 = Detection(
        video_id=vid_id,
        frame_id=uuid4(),
        frame_number=1,
        timestamp_seconds=0.04,
        class_name="car",
        class_id=2,
        confidence=0.9,
        bbox=BoundingBox(x=120.0, y=50.0, width=40.0, height=40.0),
        provider="yolo26",
    )
    tracker.update(
        [d1], Frame(video_id=vid_id, frame_number=1, timestamp_seconds=0.04, width=640, height=480)
    )

    # Frame 2: Missed detection. Kalman filter must predict x > 120 (continuing rightward velocity)
    tracker.update(
        [], Frame(video_id=vid_id, frame_number=2, timestamp_seconds=0.08, width=640, height=480)
    )
    all_tracks = tracker.all_tracks()
    assert len(all_tracks) == 1
    pred_box = all_tracks[0].end_bbox
    assert pred_box is not None
    # Velocity vx was positive, so predicted x must be greater than previous measurement of 120
    assert pred_box.x > 120.0


def test_real_model_adapter_interface_with_mock_engine() -> None:
    """Test YOLO26Detector adapter interface using deterministic mock engine without weights.

    Verifies the entire contract:
      VideoReader -> DecodedFrame -> YOLO26Detector -> Detection -> BoTSORTTracker -> Track
    and asserts:
      - frame_index preserved
      - FrameTimestamp preserved
      - confidence populated
      - bbox valid
      - class information populated
    """

    class MockBoxes:
        def __init__(self) -> None:
            self.xyxy = [[120.0, 80.0, 240.0, 200.0]]
            self.conf = [0.88]
            self.cls = [2]

        def __len__(self) -> int:
            return 1

    class MockResult:
        def __init__(self) -> None:
            self.boxes = MockBoxes()
            self.names = {0: "person", 2: "car"}
            self.masks = None

    class MockYOLOModel:
        def __call__(self, *args: object, **kwargs: object) -> list[MockResult]:
            return [MockResult()]

    detector = YOLO26Detector(
        config=YOLO26DetectorConfig(model_path="mock_yolo26n.pt"),
        model=MockYOLOModel(),
    )
    tracker = BoTSORTTracker()

    # Create synthetic video stream
    reader = SyntheticPerceptionReader(frame_count=3, fps=25.0)
    for frame_idx, decoded_frame in enumerate(reader.frames()):
        # 1. VideoReader -> DecodedFrame
        assert decoded_frame.frame_index == frame_idx
        expected_ts = reader.get_frame_timestamp(frame_idx)
        assert decoded_frame.frame_timestamp == expected_ts

        # 2. DecodedFrame -> YOLO26Detector -> Detection
        detections = detector.detect(decoded_frame)
        assert len(detections) == 1
        det = detections[0]

        # Explicit assertions required by Phase 2.0R:
        assert det.frame_number == frame_idx, "frame_index must be preserved"
        assert det.frame_timestamp == expected_ts, "FrameTimestamp must be preserved"
        assert det.frame_timestamp.timestamp_source == TimestampSource.CONTAINER
        assert det.confidence == pytest.approx(0.88, abs=1e-3), "confidence must be populated"
        assert det.class_id == 2, "class id populated"
        assert det.class_name == "car", "class information populated"
        assert det.bbox.width == pytest.approx(120.0, abs=1e-2), "bbox width valid"
        assert det.bbox.height == pytest.approx(120.0, abs=1e-2), "bbox height valid"
        assert det.bbox.x == 120.0 and det.bbox.y == 80.0

        # 3. Detection -> BoTSORTTracker -> Track
        trajectory_points = tracker.update(detections, decoded_frame)
        assert len(trajectory_points) == 1
        pt = trajectory_points[0]
        assert pt.frame_number == frame_idx
        assert pt.frame_timestamp == expected_ts
        assert pt.confidence == det.confidence

    tracks = tracker.active_tracks()
    assert len(tracks) == 1
    assert tracks[0].class_name == "car"
    assert tracks[0].class_id == 2
    assert tracks[0].frame_count == 3
