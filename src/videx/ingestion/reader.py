"""Deterministic video frame reader implementation.

Provides random-access frame retrieval by sequential index and presentation timestamp,
with bounds checking, time mapping, sequential seek optimization, and raw/encoded frame access.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import cv2

from videx.domain.schemas import Frame
from videx.ingestion.base import (
    CorruptVideoError,
    DecodedFrame,
    FrameOutOfBoundsError,
    FrameTimestamp,
    TimestampIndex,
    TimestampOutOfBoundsError,
    TimingMode,
    VideoReader,
)
from videx.ingestion.metadata import RawVideoMetadata, extract_video_metadata
from videx.ingestion.timing import create_timestamp_index


class OpenCVVideoReader(VideoReader):
    """Deterministic frame reader powered by OpenCV VideoCapture.

    Guarantees deterministic random access across video frames, returning
    typed Frame domain models paired with raw NumPy arrays or encoded JPEG bytes.
    Optimizes sequential frame reads by avoiding redundant seeks when reading
    consecutive frames.
    """

    def __init__(
        self,
        video_path: Path | str,
        video_id: UUID | None = None,
        metadata: RawVideoMetadata | None = None,
        timing_index: TimestampIndex | None = None,
        timing_mode: TimingMode | str = TimingMode.EXACT,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        self._video_id = video_id or uuid4()
        self._metadata = metadata or extract_video_metadata(self.video_path)
        self._timing_mode = TimingMode(timing_mode)

        # Timing index governs deterministic presentation timestamps
        if timing_index is not None:
            self._timing_index = timing_index
        else:
            self._timing_index = create_timestamp_index(
                fps=self._metadata.fps,
                total_frames=self._metadata.total_frames,
                duration_seconds=self._metadata.duration_seconds,
                is_vfr=self._metadata.is_vfr,
                video_path=self.video_path,
                mode=self._timing_mode,
            )

        self._cap: cv2.VideoCapture | None = None
        self._is_opened = False
        self._current_frame_pos: int = -1

        # Synchronize parameters with authoritative timing index
        if self._timing_index.total_frames > 0:
            self._total_frames: int = self._timing_index.total_frames
        else:
            self._total_frames = self._metadata.total_frames

        if self._timing_index.duration_seconds > 0.0:
            self._duration_seconds: float = self._timing_index.duration_seconds
        else:
            self._duration_seconds = self._metadata.duration_seconds

        self._fps: float = self._timing_index.fps or self._metadata.fps
        self._width: int = self._metadata.width
        self._height: int = self._metadata.height

    @property
    def video_id(self) -> UUID | None:
        """UUID of the video being read."""
        return self._video_id

    @property
    def is_opened(self) -> bool:
        """Whether the video capture stream is currently opened."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def total_frames(self) -> int:
        """Total frame count of the video."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Nominal or average frame rate (frames per second)."""
        return self._fps

    @property
    def duration_seconds(self) -> float:
        """Total duration of the video in seconds."""
        return self._duration_seconds

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def timing_index(self) -> TimestampIndex:
        """Active TimestampIndex governing frame-to-time mapping."""
        return self._timing_index

    @property
    def timing_mode(self) -> TimingMode:
        """Active timing policy mode."""
        return self._timing_index.timing_mode

    def open(self) -> None:
        """Open the video stream."""
        if self.is_opened:
            return

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise CorruptVideoError(self.video_path, "OpenCV VideoCapture failed to open file")
        self._is_opened = True
        self._current_frame_pos = 0

    def close(self) -> None:
        """Release video decoder resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False
        self._current_frame_pos = -1

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """Calculate or look up the presentation timestamp for a 0-based frame index.

        Delegates to the active TimestampIndex (CFR formula or explicit PTS table).

        Args:
            frame_index: 0-based integer frame index.

        Returns:
            Timestamp in seconds.

        Raises:
            FrameOutOfBoundsError: If frame_index is negative or >= total_frames.
        """
        if frame_index < 0 or (self._total_frames > 0 and frame_index >= self._total_frames):
            raise FrameOutOfBoundsError(frame_index, self._total_frames)

        return self._timing_index.get_timestamp_for_frame(frame_index)

    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int:
        """Map a timestamp in seconds to the closest 0-based frame index.

        Delegates to the active TimestampIndex.

        Args:
            timestamp_seconds: Timestamp in seconds.

        Returns:
            0-based frame index.

        Raises:
            TimestampOutOfBoundsError: If timestamp is negative or exceeds video duration.
        """
        epsilon = 0.05
        if timestamp_seconds < 0.0 or (
            self._duration_seconds > 0.0 and timestamp_seconds > self._duration_seconds + epsilon
        ):
            raise TimestampOutOfBoundsError(timestamp_seconds, self._duration_seconds)

        return self._timing_index.get_frame_for_timestamp(timestamp_seconds)

    def read_decoded_frame(self, frame_index: int) -> DecodedFrame:
        """Read a frame by index returning raw NumPy array representation.

        Optimizes sequential reading: if the requested frame matches the current
        decoder position, seek overhead is eliminated.

        Note on OpenCV timing: In OpenCV with FFmpeg, querying cap.get(CAP_PROP_POS_MSEC)
        after cap.read() returns the position of the *next* frame rather than the
        decoded frame, and pos_msec on frame 0 is often 0.0. To guarantee deterministic,
        reproducible timestamp association across CFR and VFR media, presentation
        timestamps are governed by the reader's TimestampIndex.

        Args:
            frame_index: 0-based frame number.

        Returns:
            DecodedFrame containing raw frame array and metadata.

        Raises:
            FrameOutOfBoundsError: If frame_index < 0 or >= total_frames.
            CorruptVideoError: If decoding the frame fails.
        """
        if frame_index < 0 or (self._total_frames > 0 and frame_index >= self._total_frames):
            raise FrameOutOfBoundsError(frame_index, self._total_frames)

        if not self.is_opened:
            self.open()

        assert self._cap is not None  # for type checking

        # Sequential seek optimization: only seek if position doesn't match
        if self._current_frame_pos != frame_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            self._current_frame_pos = frame_index

        ret, mat = self._cap.read()
        if not ret or mat is None:
            self._current_frame_pos = -1
            raise CorruptVideoError(
                self.video_path,
                f"Failed to decode frame at index {frame_index}",
            )

        # Decoder has now advanced to the next sequential position
        self._current_frame_pos += 1

        actual_ts = self.get_timestamp_for_frame(frame_index)
        frame_ts = self.get_frame_timestamp(frame_index)

        return DecodedFrame(
            frame_index=frame_index,
            timestamp_seconds=actual_ts,
            width=mat.shape[1],
            height=mat.shape[0],
            frame_array=mat,
            video_id=self.video_id,
            frame_timestamp=frame_ts,
        )

    def get_frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        """Return full temporal provenance for a 0-based frame index."""
        return self._timing_index.get_frame_timestamp(frame_index)

    def read_decoded_frame_at_timestamp(self, timestamp_seconds: float) -> DecodedFrame:
        """Read the frame corresponding to a fractional video timestamp.

        Args:
            timestamp_seconds: Timestamp in seconds.

        Returns:
            DecodedFrame containing raw frame array and metadata.

        Raises:
            TimestampOutOfBoundsError: If timestamp is negative or > duration.
            CorruptVideoError: If frame decoding fails.
        """
        target_frame = self.get_frame_for_timestamp(timestamp_seconds)
        return self.read_decoded_frame(target_frame)

    def read_frame(self, frame_index: int) -> tuple[Frame, bytes]:
        """Read a frame by index returning Frame domain model and JPEG bytes.

        Preserved for backward compatibility; serializes DecodedFrame on demand.

        Args:
            frame_index: 0-based frame number.

        Returns:
            Tuple of (Frame domain model, JPEG encoded bytes).

        Raises:
            FrameOutOfBoundsError: If frame_index < 0 or >= total_frames.
            CorruptVideoError: If decoding or JPEG encoding fails.
        """
        decoded = self.read_decoded_frame(frame_index)
        return decoded.to_domain_frame(), decoded.to_jpeg()

    def read_frame_at_timestamp(self, timestamp_seconds: float) -> tuple[Frame, bytes]:
        """Read the frame corresponding to a fractional video timestamp.

        Args:
            timestamp_seconds: Timestamp in seconds.

        Returns:
            Tuple of (Frame domain model, JPEG encoded bytes).

        Raises:
            TimestampOutOfBoundsError: If timestamp is negative or > duration.
            CorruptVideoError: If frame decoding fails.
        """
        target_frame = self.get_frame_for_timestamp(timestamp_seconds)
        return self.read_frame(target_frame)

    def __enter__(self) -> OpenCVVideoReader:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()
