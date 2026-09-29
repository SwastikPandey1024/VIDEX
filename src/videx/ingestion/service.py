"""High-level video ingestion service.

Coordinates validation, technical metadata extraction, scene segmentation,
and manifest creation.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from videx.domain.schemas import VideoManifest
from videx.ingestion.base import SceneDetector, TimingMode, VideoReader
from videx.ingestion.manifest import build_manifest
from videx.ingestion.metadata import extract_video_metadata
from videx.ingestion.reader import OpenCVVideoReader
from videx.ingestion.scenes import PySceneDetectDetector, SingleSceneDetector
from videx.ingestion.timing import create_timestamp_index
from videx.ingestion.validation import validate_video_file


class VideoIngestionService:
    """Service orchestrating the complete Phase 1 video ingestion workflow."""

    def __init__(
        self,
        scene_detector: SceneDetector | None = None,
        detect_scenes: bool = True,
        supported_extensions: tuple[str, ...] | list[str] | None = None,
        timing_mode: TimingMode | str = TimingMode.EXACT,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            scene_detector: Custom SceneDetector implementation. If None,
                PySceneDetectDetector is used when detect_scenes=True.
            detect_scenes: Whether to run content-aware scene detection.
                If False, SingleSceneDetector is used.
            supported_extensions: Optional override for allowed container extensions.
            timing_mode: Policy for timestamp indexing (EXACT vs FAST). Default EXACT.
        """
        self.detect_scenes = detect_scenes
        self.supported_extensions = supported_extensions
        self.timing_mode = (
            TimingMode(timing_mode.lower()) if isinstance(timing_mode, str) else timing_mode
        )

        if scene_detector is not None:
            self._scene_detector = scene_detector
        elif detect_scenes:
            self._scene_detector = PySceneDetectDetector()
        else:
            self._scene_detector = SingleSceneDetector()

    def ingest(
        self,
        video_path: Path | str,
        video_id: UUID | None = None,
    ) -> VideoManifest:
        """Validate an input video and produce a structured VideoManifest.

        Steps:
        1. Validate file existence, format, container, and readability.
        2. Extract stream and container metadata (duration, fps, codecs, audio).
        3. Build authoritative timestamp index according to timing policy (EXACT / FAST).
        4. Detect scene boundaries using configured SceneDetector.
        5. Construct and return validated VideoManifest domain model.

        Args:
            video_path: Path to target video file.
            video_id: Optional existing UUID to assign to the video.

        Returns:
            Validated VideoManifest.

        Raises:
            VideoNotFoundError: If the file does not exist.
            UnsupportedContainerError: If container extension is not supported.
            CorruptVideoError: If file is empty or unreadable.
        """
        vid = video_id or uuid4()
        validated_path = validate_video_file(
            video_path,
            supported_extensions=self.supported_extensions,
        )

        metadata = extract_video_metadata(validated_path)

        # Build authoritative timestamp index according to configured timing mode
        timing_index = create_timestamp_index(
            fps=metadata.fps,
            total_frames=metadata.total_frames,
            duration_seconds=metadata.duration_seconds,
            is_vfr_hint=metadata.is_vfr,
            video_path=validated_path,
            mode=self.timing_mode,
        )

        scenes = self._scene_detector.detect_scenes(
            video_path=validated_path,
            video_id=vid,
            fps=metadata.fps,
            total_frames=metadata.total_frames,
            duration_seconds=metadata.duration_seconds,
        )

        return build_manifest(
            video_path=validated_path,
            metadata=metadata,
            scenes=scenes,
            video_id=vid,
            is_vfr=timing_index.is_vfr,
            timing_mode=self.timing_mode.value,
            is_vfr_hint=metadata.is_vfr,
        )

    def create_reader(
        self,
        target: VideoManifest | Path | str,
    ) -> VideoReader:
        """Instantiate a deterministic VideoReader for a video or manifest.

        Args:
            target: VideoManifest or path to a validated video file.

        Returns:
            VideoReader implementation.
        """
        if isinstance(target, VideoManifest):
            timing = create_timestamp_index(
                fps=target.fps,
                total_frames=target.total_frames,
                duration_seconds=target.duration_seconds,
                is_vfr_hint=target.is_vfr_hint,
                video_path=target.source_path,
                mode=self.timing_mode,
            )
            return OpenCVVideoReader(
                video_path=target.source_path,
                video_id=target.video_id,
                timing_index=timing,
                timing_mode=self.timing_mode,
            )
        return OpenCVVideoReader(video_path=target, timing_mode=self.timing_mode)
