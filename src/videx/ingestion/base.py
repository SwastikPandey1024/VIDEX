"""Base interfaces and exceptions for video ingestion.

Defines protocols and exceptions governing video validation, metadata extraction,
deterministic frame access, and scene detection.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from videx.domain.schemas import Frame, FrameTimestamp, Scene, TimestampSource

# ── Exceptions ─────────────────────────────────────────────────────────────


class IngestionError(Exception):
    """Base exception for all video ingestion and validation errors."""


class VideoValidationError(IngestionError):
    """Raised when video file validation fails."""


class VideoNotFoundError(VideoValidationError):
    """Raised when the specified video file does not exist on disk."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"Video file not found at path: {path}")
        self.path = Path(path)


class UnsupportedContainerError(VideoValidationError):
    """Raised when the video container format or file extension is not supported."""

    def __init__(self, path: Path | str, extension: str, supported: list[str]) -> None:
        supported_str = ", ".join(supported)
        super().__init__(
            f"Unsupported container extension '{extension}' for file '{path}'. "
            f"Supported extensions: {supported_str}"
        )
        self.path = Path(path)
        self.extension = extension
        self.supported = supported


class CorruptVideoError(VideoValidationError):
    """Raised when a video file is unreadable, empty (0 bytes), or corrupt."""

    def __init__(self, path: Path | str, reason: str) -> None:
        super().__init__(f"Video file '{path}' is corrupt or unreadable: {reason}")
        self.path = Path(path)
        self.reason = reason


class FrameAccessError(IngestionError):
    """Base exception for errors accessing video frames."""


class FrameOutOfBoundsError(FrameAccessError):
    """Raised when an requested frame index is negative or exceeds available frames."""

    def __init__(self, frame_index: int, total_frames: int) -> None:
        super().__init__(
            f"Frame index {frame_index} is out of bounds for video with {total_frames} frames "
            f"(valid range: 0 to {max(0, total_frames - 1)})"
        )
        self.frame_index = frame_index
        self.total_frames = total_frames


class TimestampOutOfBoundsError(FrameAccessError):
    """Raised when an requested timestamp is negative or exceeds total video duration."""

    def __init__(self, timestamp_seconds: float, duration_seconds: float) -> None:
        super().__init__(
            f"Timestamp {timestamp_seconds:.3f}s is out of bounds for video duration "
            f"{duration_seconds:.3f}s (valid range: 0.0s to {duration_seconds:.3f}s)"
        )
        self.timestamp_seconds = timestamp_seconds
        self.duration_seconds = duration_seconds


# ── Decoded Frame Representation ───────────────────────────────────────────


class DecodedFrame:
    """In-memory representation of a decoded video frame.

    Provides direct access to the raw NumPy image array for downstream
    perception engines (detection, tracking, OCR) without forcing
    eager JPEG re-encoding.
    """

    __slots__ = (
        "frame_index",
        "timestamp_seconds",
        "width",
        "height",
        "frame_array",
        "video_id",
        "scene_id",
        "frame_timestamp",
    )

    def __init__(
        self,
        frame_index: int,
        timestamp_seconds: float,
        width: int,
        height: int,
        frame_array: object,
        video_id: UUID | None = None,
        scene_id: UUID | None = None,
        frame_timestamp: FrameTimestamp | None = None,
    ) -> None:
        self.frame_index = frame_index
        self.timestamp_seconds = timestamp_seconds
        self.width = width
        self.height = height
        self.frame_array = frame_array
        self.video_id = video_id
        self.scene_id = scene_id
        self.frame_timestamp = frame_timestamp

    def to_jpeg(self, quality: int = 95) -> bytes:
        """Encode the raw frame array to JPEG bytes on demand."""
        import cv2
        import numpy as np

        if not isinstance(self.frame_array, np.ndarray):
            raise CorruptVideoError(
                f"frame_{self.frame_index}",
                "Frame array is not a valid numpy ndarray",
            )
        success, enc_buf = cv2.imencode(
            ".jpg", self.frame_array, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success or enc_buf is None:
            raise CorruptVideoError(
                f"frame_{self.frame_index}",
                f"Failed to encode frame {self.frame_index} to JPEG bytes",
            )
        return bytes(enc_buf)

    def to_domain_frame(self, frame_data_path: str | None = None) -> Frame:
        """Convert to the persistent VIDEX Frame domain schema."""
        from uuid import uuid4

        return Frame(
            video_id=self.video_id or uuid4(),
            scene_id=self.scene_id,
            frame_number=self.frame_index,
            timestamp_seconds=self.timestamp_seconds,
            frame_timestamp=self.frame_timestamp,
            width=self.width,
            height=self.height,
            frame_data_path=frame_data_path,
        )


class TimingMode(StrEnum):
    """Timing policy governing temporal indexing precision.

    - EXACT: Authoritative container packet/frame scan via PyAV. Default in VIDEX.
    - FAST: Mathematical CFR formula when container metadata supports it.
    """

    EXACT = "exact"
    FAST = "fast"


# ── Protocols ──────────────────────────────────────────────────────────────


@runtime_checkable
class TimestampIndex(Protocol):
    """Protocol for mapping between sequential frame numbers and presentation timestamps."""

    @property
    def is_vfr(self) -> bool:
        """Whether this index models variable frame rate timing."""
        ...

    @property
    def fps(self) -> float:
        """Nominal or average frames per second."""
        ...

    @property
    def total_frames(self) -> int:
        """Total number of frames in the index."""
        ...

    @property
    def duration_seconds(self) -> float:
        """Total duration spanned by the index in seconds."""
        ...

    @property
    def timing_mode(self) -> TimingMode:
        """Timing policy mode that produced this index."""
        ...

    @property
    def has_repaired_timestamps(self) -> bool:
        """Whether any timestamps in the index were derived/repaired."""
        ...

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """Return the presentation timestamp in seconds for a 0-based frame index."""
        ...

    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int:
        """Return the closest 0-based frame index for a presentation timestamp in seconds."""
        ...

    def get_frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        """Return the detailed provenance record for a 0-based frame index."""
        ...


@runtime_checkable
class VideoReader(Protocol):
    """Protocol for deterministic video frame access.

    Supports reading frames by 0-based frame index or by fractional timestamp,
    ensuring frame dimensions, timestamp, raw frame array, or encoded image bytes are accessible.
    """

    @property
    def video_id(self) -> UUID | None:
        """UUID of the video being read, if assigned."""
        ...

    @property
    def is_opened(self) -> bool:
        """Whether the underlying video stream/handle is open."""
        ...

    @property
    def total_frames(self) -> int:
        """Total frame count of the video."""
        ...

    @property
    def fps(self) -> float:
        """Nominal or average frame rate (frames per second)."""
        ...

    @property
    def duration_seconds(self) -> float:
        """Total duration of the video in seconds."""
        ...

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        ...

    @property
    def timing_mode(self) -> TimingMode:
        """Active timing mode governing frame timestamps."""
        ...

    def open(self) -> None:
        """Open or initialize the video reader."""
        ...

    def close(self) -> None:
        """Release all video decoder resources and file handles."""
        ...

    def read_decoded_frame(self, frame_index: int) -> DecodedFrame:
        """Read a frame by index returning raw NumPy array representation.

        Args:
            frame_index: Non-negative integer index of the frame.

        Returns:
            DecodedFrame containing raw frame array and metadata.

        Raises:
            FrameOutOfBoundsError: If frame_index is negative or >= total_frames.
            CorruptVideoError: If decoding fails for the requested frame.
        """
        ...

    def read_decoded_frame_at_timestamp(self, timestamp_seconds: float) -> DecodedFrame:
        """Read a frame at timestamp returning raw NumPy array representation.

        Args:
            timestamp_seconds: Non-negative time offset in seconds.

        Returns:
            DecodedFrame containing raw frame array and metadata.

        Raises:
            TimestampOutOfBoundsError: If timestamp is negative or > duration.
            CorruptVideoError: If decoding fails.
        """
        ...

    def read_frame(self, frame_index: int) -> tuple[Frame, bytes]:
        """Read a single frame by its 0-based sequential frame index.

        Args:
            frame_index: Non-negative integer index of the frame.

        Returns:
            A tuple of (Frame domain model, encoded JPEG image bytes).

        Raises:
            FrameOutOfBoundsError: If frame_index is negative or >= total_frames.
            CorruptVideoError: If decoding fails for the requested frame.
        """
        ...

    def read_frame_at_timestamp(self, timestamp_seconds: float) -> tuple[Frame, bytes]:
        """Read the frame corresponding to the specified video timestamp.

        Maps the requested timestamp to the nearest appropriate frame,
        respecting presentation timestamp (PTS) metadata.

        Args:
            timestamp_seconds: Non-negative time offset in seconds.

        Returns:
            A tuple of (Frame domain model, encoded JPEG image bytes).

        Raises:
            TimestampOutOfBoundsError: If timestamp is negative or > duration.
            CorruptVideoError: If decoding fails.
        """
        ...

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """Calculate or look up the presentation timestamp for a frame index."""
        ...

    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int:
        """Map a timestamp to the closest valid frame index."""
        ...

    def get_frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        """Return the detailed provenance record for a 0-based frame index."""
        ...

    def __enter__(self) -> VideoReader:
        """Context manager entry."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit, ensures close() is invoked."""
        ...


@runtime_checkable
class SceneDetector(Protocol):
    """Protocol for segmenting video into logical scene boundaries."""

    @property
    def detector_name(self) -> str:
        """Unique identifier for this scene detector implementation."""
        ...

    def detect_scenes(
        self,
        video_path: Path | str,
        video_id: UUID,
        fps: float,
        total_frames: int | None = None,
        duration_seconds: float | None = None,
    ) -> list[Scene]:
        """Detect scene boundaries in a video.

        Args:
            video_path: Path to the target video file.
            video_id: Associated parent Video UUID.
            fps: Frame rate of the video.
            total_frames: Optional known total frame count.
            duration_seconds: Optional known video duration in seconds.

        Returns:
            Chronologically ordered list of non-overlapping Scene domain models.
            If no scene cuts are found, a single scene spanning the entire
            duration must be returned.
        """
        ...


__all__ = [
    "CorruptVideoError",
    "DecodedFrame",
    "FrameAccessError",
    "FrameOutOfBoundsError",
    "FrameTimestamp",
    "IngestionError",
    "SceneDetector",
    "TimestampIndex",
    "TimestampOutOfBoundsError",
    "TimestampSource",
    "TimingMode",
    "UnsupportedContainerError",
    "VideoNotFoundError",
    "VideoReader",
    "VideoValidationError",
]
