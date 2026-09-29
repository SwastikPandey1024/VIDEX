"""Real YOLO26 acceptance test.

Executes only when YOLO26 weights are present locally (zero CI weight download dependency).
Validates the full perception pipeline:
  test video -> VideoReader -> DecodedFrame -> YOLO26Detector
  -> Detection -> BoTSORTTracker -> Track
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from videx.domain.schemas import FrameTimestamp
from videx.ingestion.reader import OpenCVVideoReader
from videx.perception.detection import YOLO26Detector, YOLO26DetectorConfig
from videx.perception.tracking import BoTSORTTracker


def _find_yolo26_weights() -> Path | None:
    """Find local weights file if present, without triggering network download."""
    env_path = os.environ.get("YOLO26_WEIGHTS_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    default_paths = [
        Path("yolo26n.pt"),
        Path("yolo26s.pt"),
        Path("yolo26m.pt"),
        Path("models/yolo26n.pt"),
        Path("weights/yolo26n.pt"),
    ]
    for p in default_paths:
        if p.is_file():
            return p

    return None


def _create_acceptance_video(video_path: Path, frame_count: int = 5, fps: float = 25.0) -> None:
    """Create test video from bundled ultralytics image or synthetic fallback."""
    img: np.ndarray | None = None
    try:
        import ultralytics

        pkg_dir = Path(ultralytics.__file__).parent
        bus_candidate = pkg_dir / "assets" / "bus.jpg"
        if bus_candidate.is_file():
            loaded = cv2.imread(str(bus_candidate))
            if loaded is not None:
                img = loaded
    except Exception:
        img = None

    if img is None:
        width, height = 320, 240
        fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        try:
            for i in range(frame_count):
                frame = np.full((height, width, 3), 40, dtype=np.uint8)
                box_x = 20 + i * 8
                box_y = 60 + i * 3
                cv2.rectangle(
                    frame, (box_x, box_y), (box_x + 50, box_y + 80), (220, 220, 220), -1
                )
                out.write(frame)
        finally:
            out.release()
    else:
        h, w = img.shape[:2]
        fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        try:
            for i in range(frame_count):
                # Gentle horizontal shift simulating continuous target movement
                shift_matrix = np.array(
                    [[1.0, 0.0, float(i * 2)], [0.0, 1.0, 0.0]], dtype=np.float32
                )
                shifted = cv2.warpAffine(img, shift_matrix, (w, h))
                out.write(shifted)
        finally:
            out.release()


def test_yolo26_real_weights_acceptance(tmp_path: Path) -> None:
    """Acceptance test running when real YOLO26 weights and ultralytics are present."""
    weights_path = _find_yolo26_weights()
    if weights_path is None:
        pytest.skip(
            "YOLO26 weights not found. Set YOLO26_WEIGHTS_PATH or place "
            "weights at yolo26n.pt to run real acceptance test."
        )

    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip(
            "ultralytics package not installed in environment; skipping real model test."
        )

    import ultralytics

    test_video = tmp_path / "acceptance_test.mp4"
    frame_count = 5
    _create_acceptance_video(test_video, frame_count=frame_count, fps=25.0)

    # 1. Open test video with VideoReader
    with OpenCVVideoReader(test_video) as reader:
        assert reader.total_frames == frame_count

        device = "cpu"
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                model_path=str(weights_path),
                confidence_threshold=0.25,
                device=device,
            )
        )
        # Verify real model loads successfully
        detector.warmup()
        assert detector._model is not None, "Model must be loaded into memory"

        tracker = BoTSORTTracker()

        all_detections = []
        all_trajectory_points = []
        latencies_ms: list[float] = []
        track_id_occurrences: dict[str, list[int]] = {}

        # 2. Iterate frames: VideoReader -> DecodedFrame -> YOLO26Detector -> Detection
        for frame_idx in range(reader.total_frames):
            decoded_frame = reader.read_decoded_frame(frame_idx)
            assert decoded_frame.frame_index == frame_idx
            assert decoded_frame.frame_timestamp is not None
            expected_ts = reader.get_frame_timestamp(frame_idx)
            assert decoded_frame.frame_timestamp == expected_ts

            # Measure inference latency
            t0 = time.perf_counter()
            detections = detector.detect(decoded_frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Assertions on each detection produced by real model
            for det in detections:
                assert det.frame_number == frame_idx, "frame_index preserved"
                assert det.frame_timestamp == expected_ts, "FrameTimestamp preserved exactly"
                assert isinstance(det.frame_timestamp, FrameTimestamp)
                assert 0.0 <= det.confidence <= 1.0, "confidence is in [0, 1]"
                assert det.bbox.width > 0 and det.bbox.height > 0, "bbox valid dimensions"
                assert det.bbox.x >= 0 and det.bbox.y >= 0, "bbox valid coordinates"
                assert bool(det.class_name), "class_name populated"
                assert det.class_id >= 0, "class_id populated"
                # Timestamp provenance assertions
                assert (
                    det.frame_timestamp.timestamp_source == expected_ts.timestamp_source
                ), "timestamp provenance preserved"
                assert det.frame_timestamp.is_repaired == expected_ts.is_repaired

            all_detections.extend(detections)

            # 3. Detection -> BoTSORTTracker -> TrajectoryPoint -> Track
            trajectory_points = tracker.update(detections, decoded_frame)
            for pt in trajectory_points:
                assert pt.frame_number == frame_idx
                assert pt.frame_timestamp == expected_ts, "trajectory points retain timestamp"
                assert pt.frame_timestamp.timestamp_source == expected_ts.timestamp_source
                track_id_occurrences.setdefault(str(pt.track_id), []).append(frame_idx)

            all_trajectory_points.extend(trajectory_points)

        # 4. Verify tracking results and ID persistence
        tracks = tracker.all_tracks()
        assert len(all_detections) > 0, "Real YOLO26 model must produce detections"
        assert len(tracks) >= 1, "BoTSORTTracker must produce at least one track"

        # Explicitly verify tracker produces stable IDs on successive frames
        multi_frame_tracks = [
            tid for tid, frames in track_id_occurrences.items() if len(frames) > 1
        ]
        assert len(multi_frame_tracks) >= 1, (
            "Tracker must produce stable IDs across successive frames"
        )

        for trk in tracks:
            assert trk.first_seen_frame_number <= trk.last_seen_frame_number
            assert trk.first_seen_timestamp_seconds <= trk.last_seen_timestamp_seconds

        # 5. Capture runtime telemetry
        mean_latency = sum(latencies_ms) / len(latencies_ms)
        min_latency = min(latencies_ms)
        max_latency = max(latencies_ms)
        print("\n=== Real YOLO26 Acceptance Telemetry ===")
        print(f"Model Filename:     {weights_path.name}")
        print(f"Ultralytics Ver:    {ultralytics.__version__}")
        print(f"Device Used:        {device}")
        print(f"Frames Processed:   {frame_count}")
        print(f"Detections Produced:{len(all_detections)}")
        print(f"Tracks Produced:    {len(tracks)}")
        print(f"Stable Multi-Frame: {len(multi_frame_tracks)}")
        print(
            f"Inference Latency:  mean={mean_latency:.2f}ms, "
            f"min={min_latency:.2f}ms, max={max_latency:.2f}ms"
        )
        print("========================================")
