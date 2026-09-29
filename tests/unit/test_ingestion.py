"""Unit tests for VIDEX video ingestion subsystem (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import av
import cv2
import numpy as np
import pytest

from videx.domain.schemas import VideoManifest
from videx.ingestion.base import (
    CorruptVideoError,
    DecodedFrame,
    FrameOutOfBoundsError,
    FrameTimestamp,
    TimestampOutOfBoundsError,
    TimestampSource,
    TimingMode,
    UnsupportedContainerError,
    VideoNotFoundError,
    VideoReader,
)
from videx.ingestion.manifest import build_manifest, format_manifest_summary
from videx.ingestion.metadata import (
    _parse_rational_fps,
    extract_metadata_opencv,
    extract_video_metadata,
)
from videx.ingestion.reader import OpenCVVideoReader
from videx.ingestion.scenes import PySceneDetectDetector, SingleSceneDetector
from videx.ingestion.service import VideoIngestionService
from videx.ingestion.timing import (
    CFRTimestampIndex,
    ExplicitTimestampIndex,
    PyAVTimestampIndex,
    create_timestamp_index,
    detect_vfr_from_timestamps,
    sanitize_timestamps,
)
from videx.ingestion.validation import validate_video_file


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a tiny 2-second, 2-scene synthetic MP4 video (50 frames at 25 fps).

    Frames 0..24: Solid red
    Frames 25..49: Solid blue
    """
    video_path = tmp_path / "synthetic_test.mp4"
    width, height = 160, 120
    fps = 25.0
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        # First scene: 25 red frames
        for _ in range(25):
            frame_red = np.full((height, width, 3), (0, 0, 255), dtype=np.uint8)
            out.write(frame_red)

        # Second scene: 25 blue frames
        for _ in range(25):
            frame_blue = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
            out.write(frame_blue)
    finally:
        out.release()

    return video_path


# ── Test Validation ────────────────────────────────────────────────────────


class TestVideoValidation:
    """Tests for validate_video_file."""

    def test_missing_file_raises_video_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"
        with pytest.raises(VideoNotFoundError) as exc_info:
            validate_video_file(missing)
        assert "not found" in str(exc_info.value).lower()

    def test_directory_path_raises_video_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(VideoNotFoundError) as exc_info:
            validate_video_file(tmp_path)
        assert "not a regular file" in str(exc_info.value).lower()

    def test_unsupported_extension_raises_unsupported_container(self, tmp_path: Path) -> None:
        text_file = tmp_path / "document.txt"
        text_file.write_text("not a video")
        with pytest.raises(UnsupportedContainerError) as exc_info:
            validate_video_file(text_file)
        assert "unsupported container" in str(exc_info.value).lower()
        assert exc_info.value.extension == ".txt"

    def test_zero_byte_file_raises_corrupt_video(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(CorruptVideoError) as exc_info:
            validate_video_file(empty)
        assert "0 bytes" in str(exc_info.value).lower()

    def test_garbage_content_raises_corrupt_video(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"THIS IS NOT A VALID MP4 CONTAINER HEADER 1234567890")
        with pytest.raises(CorruptVideoError) as exc_info:
            validate_video_file(corrupt)
        assert "corrupt or unreadable" in str(exc_info.value).lower()

    def test_valid_video_passes_validation(self, synthetic_video: Path) -> None:
        validated = validate_video_file(synthetic_video)
        assert validated.exists()
        assert validated.is_file()


# ── Test Metadata Extraction ───────────────────────────────────────────────


class TestMetadataExtraction:
    """Tests for metadata extraction."""

    def test_rational_fps_parser(self) -> None:
        assert _parse_rational_fps("30000/1001") == pytest.approx(29.970029, rel=1e-4)
        assert _parse_rational_fps("25/1") == 25.0
        assert _parse_rational_fps("24.0") == 24.0
        assert _parse_rational_fps("0/0") is None
        assert _parse_rational_fps("invalid") is None

    def test_extract_metadata_from_synthetic_video(self, synthetic_video: Path) -> None:
        meta = extract_video_metadata(synthetic_video)
        assert meta.filename == "synthetic_test.mp4"
        assert meta.filesize_bytes > 0
        assert meta.width == 160
        assert meta.height == 120
        assert meta.fps == pytest.approx(25.0, abs=0.5)
        assert meta.total_frames == 50
        assert meta.duration_seconds == pytest.approx(2.0, abs=0.2)
        assert meta.mime_type == "video/mp4"

    def test_corrupt_file_opencv_metadata_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_bytes(b"not valid video")
        with pytest.raises(CorruptVideoError):
            extract_metadata_opencv(bad_file)


# ── Test VideoReader ───────────────────────────────────────────────────────


class TestVideoReader:
    """Tests for OpenCVVideoReader."""

    def test_open_and_close(self, synthetic_video: Path) -> None:
        reader = OpenCVVideoReader(synthetic_video)
        assert not reader.is_opened
        reader.open()
        assert reader.is_opened
        reader.close()
        assert not reader.is_opened

    def test_context_manager(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            assert reader.is_opened
            assert reader.total_frames == 50
        assert not reader.is_opened

    def test_read_frame_by_index(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            frame_meta, frame_bytes = reader.read_frame(10)
            assert frame_meta.frame_number == 10
            assert frame_meta.width == 160
            assert frame_meta.height == 120
            assert frame_meta.timestamp_seconds == pytest.approx(10 / 25.0, abs=0.05)
            assert isinstance(frame_bytes, bytes)
            assert len(frame_bytes) > 0
            # JPEG magic bytes header check
            assert frame_bytes[:2] == b"\xff\xd8"

    def test_read_frame_out_of_bounds_negative(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            with pytest.raises(FrameOutOfBoundsError):
                reader.read_frame(-1)

    def test_read_frame_out_of_bounds_exceeded(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            with pytest.raises(FrameOutOfBoundsError):
                reader.read_frame(50)  # valid frames are 0..49

    def test_get_timestamp_for_frame(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            assert reader.get_timestamp_for_frame(0) == 0.0
            assert reader.get_timestamp_for_frame(25) == pytest.approx(1.0, abs=0.01)

    def test_get_frame_for_timestamp(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            assert reader.get_frame_for_timestamp(0.0) == 0
            assert reader.get_frame_for_timestamp(1.0) == 25
            assert reader.get_frame_for_timestamp(1.96) == 49

    def test_read_frame_at_timestamp(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            frame_meta, frame_bytes = reader.read_frame_at_timestamp(1.0)
            assert frame_meta.frame_number == 25
            assert frame_meta.timestamp_seconds == pytest.approx(1.0, abs=0.05)
            assert len(frame_bytes) > 0

    def test_timestamp_out_of_bounds_negative(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            with pytest.raises(TimestampOutOfBoundsError):
                reader.read_frame_at_timestamp(-0.5)

    def test_timestamp_out_of_bounds_exceeded(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            with pytest.raises(TimestampOutOfBoundsError):
                reader.read_frame_at_timestamp(10.0)

    def test_frame_retrieval_is_deterministic(self, synthetic_video: Path) -> None:
        """Verify that repeatedly reading the same frame produces identical bytes."""
        with OpenCVVideoReader(synthetic_video) as reader:
            _, bytes1 = reader.read_frame(15)
            _, bytes2 = reader.read_frame(15)
            assert bytes1 == bytes2


# ── Test Scene Detection ───────────────────────────────────────────────────


class TestSceneDetection:
    """Tests for scene detection implementations."""

    def test_single_scene_detector(self) -> None:
        detector = SingleSceneDetector()
        assert detector.detector_name == "single_scene"
        vid = uuid4()
        scenes = detector.detect_scenes(
            video_path="dummy.mp4",
            video_id=vid,
            fps=25.0,
            total_frames=100,
            duration_seconds=4.0,
        )
        assert len(scenes) == 1
        sc = scenes[0]
        assert sc.scene_index == 0
        assert sc.video_id == vid
        assert sc.start_frame_number == 0
        assert sc.end_frame_number == 99
        assert sc.start_timestamp_seconds == 0.0
        assert sc.end_timestamp_seconds == 4.0

    def test_pyscenedetect_detector_synthetic_video(self, synthetic_video: Path) -> None:
        detector = PySceneDetectDetector(threshold=20.0, min_scene_len_frames=5)
        vid = uuid4()
        scenes = detector.detect_scenes(
            video_path=synthetic_video,
            video_id=vid,
            fps=25.0,
            total_frames=50,
            duration_seconds=2.0,
        )
        # Should detect scenes (either 2 scenes from cut or 1 fallback)
        assert len(scenes) >= 1
        for idx, sc in enumerate(scenes):
            assert sc.scene_index == idx
            assert sc.video_id == vid
            assert sc.start_timestamp_seconds <= sc.end_timestamp_seconds
            assert sc.start_frame_number <= sc.end_frame_number

    def test_pyscenedetect_fallback_on_nonexistent(self, tmp_path: Path) -> None:
        detector = PySceneDetectDetector()
        vid = uuid4()
        scenes = detector.detect_scenes(
            video_path=tmp_path / "non_existent.mp4",
            video_id=vid,
            fps=25.0,
            total_frames=50,
            duration_seconds=2.0,
        )
        assert len(scenes) == 1
        assert scenes[0].video_id == vid


# ── Test VideoManifest ─────────────────────────────────────────────────────


class TestVideoManifest:
    """Tests for VideoManifest domain model and formatting."""

    def test_build_manifest(self, synthetic_video: Path) -> None:
        meta = extract_video_metadata(synthetic_video)
        scenes = SingleSceneDetector().detect_scenes(
            video_path=synthetic_video,
            video_id=uuid4(),
            fps=meta.fps,
            total_frames=meta.total_frames,
            duration_seconds=meta.duration_seconds,
        )
        manifest = build_manifest(synthetic_video, meta, scenes)
        assert manifest.filename == "synthetic_test.mp4"
        assert manifest.width == 160
        assert manifest.height == 120
        assert manifest.total_frames == 50
        assert len(manifest.scenes) == 1

    def test_manifest_serialization_roundtrip(self, synthetic_video: Path) -> None:
        meta = extract_video_metadata(synthetic_video)
        scenes = SingleSceneDetector().detect_scenes(
            video_path=synthetic_video,
            video_id=uuid4(),
            fps=meta.fps,
            total_frames=meta.total_frames,
            duration_seconds=meta.duration_seconds,
        )
        manifest = build_manifest(synthetic_video, meta, scenes)

        json_str = manifest.model_dump_json()
        data = json.loads(json_str)
        reconstructed = VideoManifest.model_validate(data)

        assert reconstructed.video_id == manifest.video_id
        assert reconstructed.total_frames == manifest.total_frames
        assert len(reconstructed.scenes) == len(manifest.scenes)

    def test_manifest_to_video_conversion(self, synthetic_video: Path) -> None:
        meta = extract_video_metadata(synthetic_video)
        scenes = SingleSceneDetector().detect_scenes(
            video_path=synthetic_video,
            video_id=uuid4(),
            fps=meta.fps,
            total_frames=meta.total_frames,
            duration_seconds=meta.duration_seconds,
        )
        manifest = build_manifest(synthetic_video, meta, scenes)
        video = manifest.to_video()

        assert video.video_id == manifest.video_id
        assert video.source_path == manifest.source_path
        assert video.duration_seconds == manifest.duration_seconds
        assert video.total_frames == manifest.total_frames
        assert video.metadata["scene_count"] == len(manifest.scenes)

    def test_format_manifest_summary(self, synthetic_video: Path) -> None:
        meta = extract_video_metadata(synthetic_video)
        scenes = SingleSceneDetector().detect_scenes(
            video_path=synthetic_video,
            video_id=uuid4(),
            fps=meta.fps,
            total_frames=meta.total_frames,
            duration_seconds=meta.duration_seconds,
        )
        manifest = build_manifest(synthetic_video, meta, scenes)
        summary = format_manifest_summary(manifest)

        assert "VIDEX Video Manifest Summary" in summary
        assert str(manifest.video_id) in summary
        assert "160 x 120" in summary


# ── Test VideoIngestionService ─────────────────────────────────────────────


class TestVideoIngestionService:
    """Tests for VideoIngestionService."""

    def test_ingest_service_full_workflow(self, synthetic_video: Path) -> None:
        service = VideoIngestionService(detect_scenes=True)
        manifest = service.ingest(synthetic_video)

        assert manifest.filename == "synthetic_test.mp4"
        assert manifest.total_frames == 50
        assert manifest.duration_seconds > 0.0
        assert len(manifest.scenes) >= 1

    def test_ingest_service_no_scenes_option(self, synthetic_video: Path) -> None:
        service = VideoIngestionService(detect_scenes=False)
        manifest = service.ingest(synthetic_video)

        assert len(manifest.scenes) == 1
        assert manifest.scenes[0].scene_index == 0

    def test_ingest_service_create_reader(self, synthetic_video: Path) -> None:
        service = VideoIngestionService()
        manifest = service.ingest(synthetic_video)

        with service.create_reader(manifest) as reader:
            assert reader.total_frames == manifest.total_frames
            assert reader.fps == manifest.fps
            assert reader.width == manifest.width
            assert reader.height == manifest.height
            frame, _ = reader.read_frame(0)
            assert frame.frame_number == 0


# ── Test Timestamp Semantics (F-01) ────────────────────────────────────────


class TestTimestampSemantics:
    """Tests for CFR and VFR timestamp mapping and detection heuristics."""

    def test_cfr_index_mapping(self) -> None:
        idx = CFRTimestampIndex(fps=25.0, total_frames=100, duration_seconds=4.0)
        assert not idx.is_vfr
        assert idx.fps == 25.0
        assert idx.total_frames == 100
        assert idx.duration_seconds == 4.0

        # Frame -> timestamp
        assert idx.get_timestamp_for_frame(0) == 0.0
        assert idx.get_timestamp_for_frame(25) == 1.0
        assert idx.get_timestamp_for_frame(99) == pytest.approx(3.96, abs=1e-3)

        # Timestamp -> frame
        assert idx.get_frame_for_timestamp(0.0) == 0
        assert idx.get_frame_for_timestamp(1.0) == 25
        assert idx.get_frame_for_timestamp(3.96) == 99

    def test_cfr_boundary_timestamps(self) -> None:
        idx = CFRTimestampIndex(fps=25.0, total_frames=50, duration_seconds=2.0)

        # Boundary checks
        with pytest.raises(FrameOutOfBoundsError):
            idx.get_timestamp_for_frame(-1)
        with pytest.raises(FrameOutOfBoundsError):
            idx.get_timestamp_for_frame(50)

        with pytest.raises(TimestampOutOfBoundsError):
            idx.get_frame_for_timestamp(-0.1)
        with pytest.raises(TimestampOutOfBoundsError):
            idx.get_frame_for_timestamp(2.5)  # exceeds duration + epsilon

        # Epsilon boundary tolerance (2.0 + 0.02)
        assert idx.get_frame_for_timestamp(2.02) == 49

    def test_vfr_explicit_index_mapping(self) -> None:
        # Non-uniform timestamps representing variable frame rate
        pts = [0.0, 0.033, 0.085, 0.120, 0.200, 0.250]
        idx = ExplicitTimestampIndex(timestamps=pts, nominal_fps=24.0)
        assert idx.is_vfr
        assert idx.total_frames == 6
        assert idx.duration_seconds == 0.250

        # Exact PTS lookup
        assert idx.get_timestamp_for_frame(0) == 0.0
        assert idx.get_timestamp_for_frame(2) == 0.085
        assert idx.get_timestamp_for_frame(5) == 0.250

        # Timestamp to nearest frame mapping via binary search
        assert idx.get_frame_for_timestamp(0.0) == 0
        assert idx.get_frame_for_timestamp(0.030) == 1  # closer to 0.033 than 0.0
        assert idx.get_frame_for_timestamp(0.090) == 2  # closer to 0.085 than 0.120
        assert idx.get_frame_for_timestamp(0.245) == 5  # closer to 0.250

        with pytest.raises(FrameOutOfBoundsError):
            idx.get_timestamp_for_frame(6)
        with pytest.raises(TimestampOutOfBoundsError):
            idx.get_frame_for_timestamp(1.0)

    def test_vfr_detection_heuristic(self) -> None:
        # Test CFR heuristic (r_fps == avg_fps within 0.5%)
        r_fps = 25.0
        avg_fps = 25.0
        discrepancy = abs(r_fps - avg_fps) / avg_fps
        assert discrepancy <= 0.005  # classified as CFR

        # Test VFR heuristic (r_fps diverges from avg_fps by > 0.5%)
        r_fps_vfr = 30.0
        avg_fps_vfr = 28.5
        discrepancy_vfr = abs(r_fps_vfr - avg_fps_vfr) / avg_fps_vfr
        assert discrepancy_vfr > 0.005  # classified as VFR

    def test_create_timestamp_index_factory(self) -> None:
        cfr = create_timestamp_index(fps=30.0, total_frames=90, duration_seconds=3.0, is_vfr=False)
        assert isinstance(cfr, CFRTimestampIndex)
        assert not cfr.is_vfr

        vfr = create_timestamp_index(
            fps=30.0,
            total_frames=3,
            duration_seconds=0.1,
            is_vfr=True,
            explicit_timestamps=[0.0, 0.04, 0.10],
        )
        assert isinstance(vfr, ExplicitTimestampIndex)
        assert vfr.is_vfr


# ── Test VideoReader Hardening (F-02, F-03, F-05, F-06, F-07) ───────────────


class TestVideoReaderHardening:
    """Tests for raw frame access, timestamp correctness, seek optimization, and properties."""

    def test_video_reader_protocol_properties(self, synthetic_video: Path) -> None:
        reader: VideoReader = OpenCVVideoReader(synthetic_video)
        assert reader.total_frames == 50
        assert reader.fps == pytest.approx(25.0, abs=0.5)
        assert reader.duration_seconds == pytest.approx(2.0, abs=0.2)
        assert reader.width == 160
        assert reader.height == 120

    def test_read_decoded_frame_raw_access(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            decoded: DecodedFrame = reader.read_decoded_frame(5)
            assert isinstance(decoded, DecodedFrame)
            assert decoded.frame_index == 5
            assert decoded.width == 160
            assert decoded.height == 120
            assert isinstance(decoded.frame_array, np.ndarray)
            assert decoded.frame_array.shape == (120, 160, 3)
            assert decoded.frame_array.dtype == np.uint8

            # Test on-demand JPEG encoding
            jpeg_bytes = decoded.to_jpeg(quality=90)
            assert isinstance(jpeg_bytes, bytes)
            assert len(jpeg_bytes) > 0
            assert jpeg_bytes[:2] == b"\xff\xd8"

            # Test conversion to persistent Frame domain model
            domain_frame = decoded.to_domain_frame()
            assert domain_frame.frame_number == 5
            assert domain_frame.width == 160
            assert domain_frame.height == 120

    def test_read_decoded_frame_at_timestamp(self, synthetic_video: Path) -> None:
        with OpenCVVideoReader(synthetic_video) as reader:
            decoded = reader.read_decoded_frame_at_timestamp(1.0)
            assert decoded.frame_index == 25
            assert decoded.timestamp_seconds == pytest.approx(1.0, abs=0.05)

    def test_frame_timestamp_correctness(self, synthetic_video: Path) -> None:
        """Verify returned frame timestamp is correctly associated with decoded frame."""
        with OpenCVVideoReader(synthetic_video) as reader:
            frame_0, _ = reader.read_frame(0)
            assert frame_0.frame_number == 0
            assert frame_0.timestamp_seconds == 0.0

            frame_1, _ = reader.read_frame(1)
            assert frame_1.frame_number == 1
            assert frame_1.timestamp_seconds == pytest.approx(1 / 25.0, abs=1e-3)

            frame_25, _ = reader.read_frame(25)
            assert frame_25.frame_number == 25
            assert frame_25.timestamp_seconds == pytest.approx(1.0, abs=1e-3)

    def test_sequential_read_optimization(self, synthetic_video: Path) -> None:
        """Verify sequential reading succeeds without seek errors and maintains position."""
        with OpenCVVideoReader(synthetic_video) as reader:
            # Read sequentially 0, 1, 2, 3
            for i in range(4):
                f, _ = reader.read_frame(i)
                assert f.frame_number == i
                assert f.timestamp_seconds == pytest.approx(i / 25.0, abs=1e-3)

            # Non-sequential seek to frame 20
            f_20, _ = reader.read_frame(20)
            assert f_20.frame_number == 20

            # Continue sequentially from 21
            f_21, _ = reader.read_frame(21)
            assert f_21.frame_number == 21

    def test_video_codec_standardization(self, synthetic_video: Path) -> None:
        service = VideoIngestionService(detect_scenes=False)
        manifest = service.ingest(synthetic_video)
        video = manifest.to_video()

        # Both schemas now canonically expose video_codec
        assert video.video_codec == manifest.video_codec
        assert manifest.is_vfr is False
        assert video.metadata["is_vfr"] is False


@pytest.fixture
def vfr_video_fixture(tmp_path: Path) -> Path:
    """Generate a genuine variable frame rate (VFR) video using PyAV.

    Encodes 5 frames with explicitly irregular presentation timestamps:
    Frame 0: 0.00s
    Frame 1: 0.10s (delta 0.10s)
    Frame 2: 0.35s (delta 0.25s)
    Frame 3: 0.45s (delta 0.10s)
    Frame 4: 0.90s (delta 0.45s)
    """
    video_path = tmp_path / "vfr_test_video.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("mpeg4", rate=100)
    stream.width = 160
    stream.height = 120
    stream.pix_fmt = "yuv420p"

    pts_seconds = [0.0, 0.10, 0.35, 0.45, 0.90]
    for pts in pts_seconds:
        frame = av.VideoFrame(160, 120, "yuv420p")
        frame.pts = int(round(pts * 100))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return video_path


# ── Test Packet-Accurate Temporal Index (Phase 1.2) ─────────────────────────


class TestPacketAccurateTemporalIndex:
    """Comprehensive tests for PyAV packet/container temporal indexing and VFR handling."""

    def test_vfr_irregular_timestamps_mapping(self) -> None:
        pts = [0.0, 0.05, 0.20, 0.25, 0.60]
        idx = PyAVTimestampIndex(timestamps=pts, nominal_fps=25.0, is_vfr=True)
        assert idx.is_vfr is True
        assert idx.total_frames == 5
        assert idx.duration_seconds == 0.60
        assert idx.timestamps == pts

        # Forward lookup
        assert idx.get_timestamp_for_frame(0) == 0.0
        assert idx.get_timestamp_for_frame(1) == 0.05
        assert idx.get_timestamp_for_frame(2) == 0.20
        assert idx.get_timestamp_for_frame(3) == 0.25
        assert idx.get_timestamp_for_frame(4) == 0.60

        # Reverse lookup (timestamp to nearest frame)
        assert idx.get_frame_for_timestamp(0.0) == 0
        assert idx.get_frame_for_timestamp(0.04) == 1
        assert idx.get_frame_for_timestamp(0.18) == 2
        assert idx.get_frame_for_timestamp(0.24) == 3
        assert idx.get_frame_for_timestamp(0.55) == 4

        with pytest.raises(FrameOutOfBoundsError):
            idx.get_timestamp_for_frame(5)
        with pytest.raises(TimestampOutOfBoundsError):
            idx.get_frame_for_timestamp(1.5)

    def test_vfr_monotonic_pts_sanitization(self) -> None:
        # Non-monotonic (backward PTS: 0.08 then 0.04)
        raw = [0.0, 0.08, 0.04, 0.12]
        sanitized = sanitize_timestamps(raw, nominal_fps=25.0)
        assert len(sanitized) == 4
        # Verify strict monotonicity
        for i in range(1, len(sanitized)):
            assert sanitized[i] > sanitized[i - 1]
        assert sanitized[0] == 0.0
        assert sanitized[1] == 0.08
        assert sanitized[3] == 0.12
        # Backward frame repaired between 0.08 and 0.12
        assert 0.08 < sanitized[2] < 0.12

    def test_vfr_duplicate_timestamp_handling(self) -> None:
        # Duplicate timestamp (0.04 appears twice)
        raw = [0.0, 0.04, 0.04, 0.08]
        sanitized = sanitize_timestamps(raw, nominal_fps=25.0)
        assert len(sanitized) == 4
        for i in range(1, len(sanitized)):
            assert sanitized[i] > sanitized[i - 1]
        assert sanitized[0] == 0.0
        assert sanitized[1] == 0.04
        assert 0.04 < sanitized[2] < 0.08
        assert sanitized[3] == 0.08

    def test_vfr_missing_timestamp_handling(self) -> None:
        # Missing at start, middle, and end
        raw = [None, 0.04, None, 0.12, None]
        sanitized = sanitize_timestamps(raw, nominal_fps=25.0)
        assert len(sanitized) == 5
        for i in range(1, len(sanitized)):
            assert sanitized[i] > sanitized[i - 1]
        assert sanitized[0] == 0.0
        assert sanitized[1] == 0.04
        assert 0.04 < sanitized[2] < 0.12
        assert sanitized[3] == 0.12
        assert sanitized[4] > 0.12

    def test_vfr_detection_exact_variance(self) -> None:
        # Perfectly uniform intervals (25 fps = 0.04s) -> CFR
        cfr = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
        assert detect_vfr_from_timestamps(cfr) is False

        # Irregular intervals (0.04, 0.15, 0.04, 0.30) -> VFR
        vfr = [0.0, 0.04, 0.19, 0.23, 0.53]
        assert detect_vfr_from_timestamps(vfr) is True

        # Short sequence (< 3 timestamps) cannot definitively diagnose VFR
        assert detect_vfr_from_timestamps([0.0, 0.04]) is False

    def test_cfr_regression_fast_path(self) -> None:
        idx = create_timestamp_index(
            fps=30.0,
            total_frames=60,
            duration_seconds=2.0,
            is_vfr=False,
        )
        assert isinstance(idx, CFRTimestampIndex)
        assert idx.is_vfr is False
        assert idx.get_timestamp_for_frame(0) == 0.0
        assert idx.get_timestamp_for_frame(30) == 1.0
        assert idx.get_frame_for_timestamp(1.0) == 30

    def test_real_generated_vfr_fixture_indexing(self, vfr_video_fixture: Path) -> None:
        idx = PyAVTimestampIndex.from_video(vfr_video_fixture)
        assert isinstance(idx, PyAVTimestampIndex)
        assert idx.is_vfr is True
        assert idx.total_frames == 5
        assert idx.duration_seconds == pytest.approx(0.90, abs=0.01)

        expected = [0.0, 0.10, 0.35, 0.45, 0.90]
        for i, exp_pts in enumerate(expected):
            assert idx.get_timestamp_for_frame(i) == pytest.approx(exp_pts, abs=0.01)

        # Reverse lookup
        assert idx.get_frame_for_timestamp(0.0) == 0
        assert idx.get_frame_for_timestamp(0.10) == 1
        assert idx.get_frame_for_timestamp(0.35) == 2
        assert idx.get_frame_for_timestamp(0.45) == 3
        assert idx.get_frame_for_timestamp(0.90) == 4

    def test_vfr_reader_frame_timestamp_parity(self, vfr_video_fixture: Path) -> None:
        """Verify OpenCV frame decoding and PyAV timestamp indexing maintain exact parity."""
        idx = PyAVTimestampIndex.from_video(vfr_video_fixture)
        with OpenCVVideoReader(vfr_video_fixture, timing_index=idx) as reader:
            assert reader.total_frames == 5
            expected_pts = [0.0, 0.10, 0.35, 0.45, 0.90]

            for i in range(5):
                decoded = reader.read_decoded_frame(i)
                assert decoded.frame_index == i
                assert decoded.timestamp_seconds == pytest.approx(expected_pts[i], abs=0.01)
                assert isinstance(decoded.frame_array, np.ndarray)
                assert decoded.frame_array.shape == (120, 160, 3)

    def test_pyav_timestamp_index_nonexistent_or_corrupt(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"
        with pytest.raises(VideoNotFoundError):
            PyAVTimestampIndex.from_video(missing)

        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(CorruptVideoError):
            PyAVTimestampIndex.from_video(empty)


@pytest.fixture
def bframe_video_fixture(tmp_path: Path) -> Path:
    """Generate an H.264 video with intentional B-frame packet reordering.

    Encoding parameters:
      - 10 frames
      - bf: 2 (allow 2 consecutive B-frames)
      - b_strategy: 0 (fixed B-frame structure)
      - g: 10 (GOP size of 10)
    This creates an encoded packet stream whose packet demux order (DTS) differs
    from frame presentation order (PTS).
    """
    video_path = tmp_path / "bframe_test.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=25)
    stream.width = 160
    stream.height = 120
    stream.pix_fmt = "yuv420p"
    stream.options = {"bf": "2", "b_strategy": "0", "g": "10"}

    # Write 10 synthetic frames with distinctive grayscale values
    for i in range(10):
        # Unique color value per frame to detect any ordering drift
        val = int(20 + i * 20)
        img = np.full((120, 160, 3), val, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return video_path


# ── Test Phase 1.2R: Temporal Truth & Exactness Hardening ──────────────────


class TestTemporalTruthAndExactness:
    """Tests hardening VFR classification, provenance, and codec reordering."""

    def test_bframe_codec_reordering_and_opencv_parity(self, bframe_video_fixture: Path) -> None:
        """Verify that demux packet order (DTS) differs from decode presentation order (PTS)

        and that OpenCV frame decoding matches PyAV presentation order exactly without drift.
        """
        # 1. Demux packets and inspect packet.dts and packet.pts
        packet_dts: list[int] = []
        packet_pts: list[int] = []
        with av.open(str(bframe_video_fixture)) as container:
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                if packet.dts is not None and packet.pts is not None:
                    packet_dts.append(packet.dts)
                    packet_pts.append(packet.pts)

        # 2. Decode frames and inspect decoded_frame.pts
        decoded_frame_pts: list[int] = []
        with av.open(str(bframe_video_fixture)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if frame.pts is not None:
                    decoded_frame_pts.append(frame.pts)

        # 3. Demonstrate DTS/decode-order vs PTS/presentation-order relationships:
        # a) Demux packets arrive in decode order: packet.dts is monotonic
        assert len(packet_dts) == 10
        for i in range(1, len(packet_dts)):
            assert packet_dts[i] >= packet_dts[i - 1], "Packets must have monotonic DTS"

        # b) Due to B-frames, packet.pts in demux order is non-monotonic
        is_packet_pts_monotonic = all(
            packet_pts[i] >= packet_pts[i - 1] for i in range(1, len(packet_pts))
        )
        assert not is_packet_pts_monotonic, (
            "Due to B-frames, packet PTS in demux stream must be non-monotonic"
        )

        # c) Decoded frames are emitted in presentation order:
        #    decoded_frame.pts is strictly monotonically increasing
        assert len(decoded_frame_pts) == 10
        for i in range(1, len(decoded_frame_pts)):
            assert (
                decoded_frame_pts[i] > decoded_frame_pts[i - 1]
            ), "Decoded frames must have strictly increasing PTS"

        # d) The set of packet PTS values matches the set of decoded frame PTS values
        assert sorted(packet_pts) == decoded_frame_pts, (
            "Decoded frame PTS sequence must equal the sorted packet PTS set"
        )

        # 2. Build PyAV timestamp index and verify monotonic presentation timestamps
        idx = PyAVTimestampIndex.from_video(bframe_video_fixture)
        assert idx.total_frames == 10
        for i in range(10):
            ts = idx.get_timestamp_for_frame(i)
            assert ts == pytest.approx(i * 0.04, abs=0.01)

        # 3. Verify OpenCV reader reads frames in exact presentation order with matching colors
        with OpenCVVideoReader(bframe_video_fixture, timing_index=idx) as reader:
            assert reader.total_frames == 10
            for i in range(10):
                decoded = reader.read_decoded_frame(i)
                assert decoded.frame_index == i
                assert decoded.timestamp_seconds == pytest.approx(i * 0.04, abs=0.01)
                # Confirm pixel content corresponds to frame i (val = 20 + i * 20)
                expected_val = 20 + i * 20
                assert isinstance(decoded.frame_array, np.ndarray)
                actual_val = int(decoded.frame_array[60, 80, 0])
                # Within tolerance for x264 YUV lossy compression
                assert abs(actual_val - expected_val) <= 12

    def test_exact_mode_vfr_detection_with_false_negative_heuristic(self, tmp_path: Path) -> None:
        """Verify that when metadata heuristics report is_vfr=False, EXACT mode

        analyzes actual presentation timestamps and correctly classifies the video as VFR.
        """
        # Create a video where nominal fps metadata appears constant/CFR,
        # but actual packet PTS intervals vary significantly.
        video_path = tmp_path / "subtle_vfr.mp4"
        container = av.open(str(video_path), mode="w")
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 160
        stream.height = 120
        stream.pix_fmt = "yuv420p"

        # 5 frames with irregular PTS: 0.0, 0.04, 0.12, 0.16, 0.32
        pts_list = [0, 4, 12, 16, 32]
        for pts in pts_list:
            frame = av.VideoFrame(160, 120, "yuv420p")
            frame.pts = pts
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Ingest with default EXACT mode
        service_exact = VideoIngestionService(detect_scenes=False, timing_mode=TimingMode.EXACT)
        manifest_exact = service_exact.ingest(video_path)

        assert manifest_exact.timing_mode == "exact"
        # Even if container hint says CFR or unknown, authoritative index discovers VFR
        assert manifest_exact.is_vfr is True

        # Now test FAST mode with explicit is_vfr_hint=False
        idx_fast = create_timestamp_index(
            fps=25.0,
            total_frames=5,
            duration_seconds=0.32,
            is_vfr_hint=False,
            video_path=video_path,
            mode=TimingMode.FAST,
        )
        assert isinstance(idx_fast, CFRTimestampIndex)
        assert idx_fast.timing_mode == TimingMode.FAST
        assert idx_fast.is_vfr is False

    def test_timestamp_provenance_container_vs_derived(self) -> None:
        """Verify that original source timestamps are tagged as CONTAINER,

        while missing, non-monotonic, or interpolated timestamps are tagged as DERIVED.
        """
        # Input raw PTS with:
        # [0.0, None, 0.10, 0.10 (duplicate), 0.05 (non-monotonic), 0.30]
        raw_pts: list[float | None] = [0.0, None, 0.10, 0.10, 0.05, 0.30]
        fps = 20.0  # nominal delta = 0.05

        sanitized = sanitize_timestamps(raw_pts, nominal_fps=fps)
        assert len(sanitized) == 6

        # Frame 0: original container PTS 0.0
        assert sanitized[0].source == TimestampSource.CONTAINER
        assert sanitized[0].is_repaired is False
        assert sanitized[0].pts_seconds == 0.0

        # Frame 1: missing (None) -> DERIVED
        assert sanitized[1].source == TimestampSource.DERIVED
        assert sanitized[1].is_repaired is True
        assert sanitized[1].repair_reason == "missing_pts"
        assert sanitized[1].pts_seconds == 0.05

        # Frame 2: original 0.10 -> CONTAINER
        assert sanitized[2].source == TimestampSource.CONTAINER
        assert sanitized[2].is_repaired is False
        assert sanitized[2].pts_seconds == 0.10

        # Frame 3: duplicate (0.10 == 0.10) -> DERIVED
        assert sanitized[3].source == TimestampSource.DERIVED
        assert sanitized[3].is_repaired is True
        assert sanitized[3].repair_reason == "duplicate_pts"
        assert sanitized[3].original_pts_seconds == 0.10
        assert sanitized[3].pts_seconds == pytest.approx(0.15)

        # Frame 4: non-monotonic (0.05 < 0.15) -> DERIVED
        assert sanitized[4].source == TimestampSource.DERIVED
        assert sanitized[4].is_repaired is True
        assert sanitized[4].repair_reason == "non_monotonic_pts"

        # Frame 5: original 0.30 -> CONTAINER
        assert sanitized[5].source == TimestampSource.CONTAINER
        assert sanitized[5].is_repaired is False
        assert sanitized[5].pts_seconds == 0.30

    def test_reader_provides_provenance_metadata(self, vfr_video_fixture: Path) -> None:
        """Verify that VideoReader and DecodedFrame expose FrameTimestamp and timing_mode."""
        service = VideoIngestionService(detect_scenes=False, timing_mode=TimingMode.EXACT)
        manifest = service.ingest(vfr_video_fixture)

        with service.create_reader(manifest) as reader:
            assert reader.timing_mode == TimingMode.EXACT
            frame_ts = reader.get_frame_timestamp(2)
            assert isinstance(frame_ts, FrameTimestamp)
            assert frame_ts.source == TimestampSource.CONTAINER
            assert frame_ts.frame_index == 2
            assert frame_ts.pts_seconds == pytest.approx(0.35, abs=0.01)

            decoded = reader.read_decoded_frame(2)
            assert decoded.frame_timestamp is not None
            assert decoded.frame_timestamp.source == TimestampSource.CONTAINER
            assert decoded.frame_timestamp.pts_seconds == pytest.approx(0.35, abs=0.01)


