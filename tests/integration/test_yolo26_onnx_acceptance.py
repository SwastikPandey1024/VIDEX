"""Real ONNX YOLO26 detector integration acceptance test.

Executes only when ONNX weights are present locally and gated by the explicit
environment variable YOLO26_ONNX_WEIGHTS_PATH.

Validates the complete perception pipeline:
  test video -> OpenCVVideoReader -> DecodedFrame -> YOLO26Detector (ONNX backend)
  -> Detection -> BoTSORTTracker -> TrajectoryPoint / Track

Also exercises actionable failure modes:
  - Missing ONNX weights file -> FileNotFoundError with actionable remediation
  - Invalid weight file extension -> ValueError
  - Missing onnxruntime dependency -> RuntimeError with installation instructions
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from videx.domain.schemas import CoordinateType, FrameTimestamp
from videx.ingestion.reader import OpenCVVideoReader
from videx.perception.detection import YOLO26Detector, YOLO26DetectorConfig
from videx.perception.tracking import BoTSORTTracker


def _find_yolo26_onnx_weights() -> Path | None:
    """Find local ONNX weights file if explicitly configured via environment variable."""
    env_path = os.environ.get("YOLO26_ONNX_WEIGHTS_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
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
                cv2.rectangle(frame, (box_x, box_y), (box_x + 50, box_y + 80), (220, 220, 220), -1)
                out.write(frame)
        finally:
            out.release()
    else:
        h, w = img.shape[:2]
        fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        try:
            for i in range(frame_count):
                shift_matrix = np.array(
                    [[1.0, 0.0, float(i * 2)], [0.0, 1.0, 0.0]], dtype=np.float32
                )
                shifted = cv2.warpAffine(img, shift_matrix, (w, h))
                out.write(shifted)
        finally:
            out.release()


def test_yolo26_onnx_real_weights_acceptance(tmp_path: Path) -> None:
    """Acceptance test running when YOLO26_ONNX_WEIGHTS_PATH is explicitly configured."""
    weights_path = _find_yolo26_onnx_weights()
    if weights_path is None:
        pytest.skip(
            "ONNX weights not configured. Set YOLO26_ONNX_WEIGHTS_PATH to an "
            "existing .onnx model file to run real ONNX acceptance test."
        )

    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime package not installed; skipping ONNX acceptance test.")

    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("ultralytics package not installed; skipping ONNX acceptance test.")

    test_video = tmp_path / "onnx_acceptance_test.mp4"
    frame_count = 5
    _create_acceptance_video(test_video, frame_count=frame_count, fps=25.0)

    # 1. Open test video with VideoReader
    with OpenCVVideoReader(test_video) as reader:
        assert reader.total_frames == frame_count

        device = "cpu"
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(weights_path),
                confidence_threshold=0.25,
                device=device,
            )
        )

        # Verify real ONNX model loads and warms up
        detector.warmup()
        assert detector._model is not None, "ONNX model must be loaded into memory"
        assert detector._is_warm is True, "Detector must be warm"
        assert detector.provider_name.startswith("yolo26_onnx_")

        tracker = BoTSORTTracker()

        all_detections = []
        all_trajectory_points = []
        latencies_ms: list[float] = []
        track_id_occurrences: dict[str, list[int]] = {}

        # 2. Iterate frames: VideoReader -> DecodedFrame -> ONNX YOLO26Detector -> Detection
        for frame_idx in range(reader.total_frames):
            decoded_frame = reader.read_decoded_frame(frame_idx)
            assert decoded_frame.frame_index == frame_idx
            assert decoded_frame.frame_timestamp is not None
            expected_ts = reader.get_frame_timestamp(frame_idx)
            assert decoded_frame.frame_timestamp == expected_ts

            t0 = time.perf_counter()
            detections = detector.detect(decoded_frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Assertions on each detection produced by ONNX model
            for det in detections:
                assert det.frame_number == frame_idx, "frame_index preserved"
                assert det.frame_timestamp == expected_ts, "FrameTimestamp preserved exactly"
                assert isinstance(det.frame_timestamp, FrameTimestamp)
                assert 0.0 <= det.confidence <= 1.0, "confidence is in [0, 1]"

                # Pixel coordinates assertions
                assert det.bbox.coordinate_type == CoordinateType.PIXEL
                assert det.bbox.width > 0 and det.bbox.height > 0, "bbox valid dimensions"
                assert det.bbox.x >= 0 and det.bbox.y >= 0, "bbox valid coordinates"
                assert bool(det.class_name), "class_name populated"
                assert det.class_id >= 0, "class_id populated"

                # Provenance assertions
                assert det.provider == detector.provider_name, "provider matches ONNX name"
                assert det.attributes["backend"] == "onnx"
                assert det.attributes["source_provider"] == detector.provider_name
                assert det.attributes["timestamp_source"] == str(expected_ts.timestamp_source.value)
                assert det.frame_timestamp.timestamp_source == expected_ts.timestamp_source
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
        assert len(all_detections) > 0, "Real ONNX model must produce detections"
        assert len(tracks) >= 1, "BoTSORTTracker must produce at least one track"

        # Verify tracker produces stable IDs on successive frames
        multi_frame_tracks = [
            tid for tid, frames in track_id_occurrences.items() if len(frames) > 1
        ]
        assert len(multi_frame_tracks) >= 1, (
            "Tracker must produce stable IDs across successive frames"
        )

        for trk in tracks:
            assert trk.first_seen_frame_number <= trk.last_seen_frame_number
            assert trk.first_seen_timestamp_seconds <= trk.last_seen_timestamp_seconds


class TestONNXAcceptanceActionableFailures:
    """Actionable failure acceptance tests for the ONNX detector integration."""

    def test_missing_onnx_weights_actionable_failure(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "non_existent_weights.onnx"
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(missing_file),
            )
        )
        with pytest.raises(FileNotFoundError, match="ONNX model file not found at"):
            detector.warmup()

    def test_invalid_extension_actionable_failure(self, tmp_path: Path) -> None:
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_text("invalid onnx format")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(fake_pt),
            )
        )
        with pytest.raises(ValueError, match=r"Expected an ONNX model file \(\.onnx\)"):
            detector.warmup()

    def test_missing_onnxruntime_actionable_failure(self, tmp_path: Path) -> None:
        fake_onnx = tmp_path / "valid.onnx"
        fake_onnx.write_text("dummy onnx header")
        detector = YOLO26Detector(
            config=YOLO26DetectorConfig(
                backend="onnx",
                onnx_model_path=str(fake_onnx),
            )
        )

        def mock_import(name: str) -> Any:
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(
                RuntimeError,
                match="ONNX Runtime is required when detector backend is 'onnx'",
            ):
                detector.warmup()
