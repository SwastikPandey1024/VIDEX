"""OCR Frame Selection and Region Cropping Scheduling abstraction.

Avoids naive per-frame OCR by scheduling evaluation temporally, at scene boundaries,
or targeted around perception bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from videx.domain.schemas import BoundingBox, Frame
from videx.ingestion.base import DecodedFrame


class OCRSchedulingStrategy(StrEnum):
    """Scheduling strategy for OCR frame evaluation."""

    ALL_FRAMES = "all_frames"
    """Process every decoded frame."""

    FIXED_INTERVAL = "fixed_interval"
    """Sample frames at a fixed frame-count interval."""

    TEMPORAL_INTERVAL = "temporal_interval"
    """Sample frames at fixed temporal duration in seconds (e.g. 1 frame/sec)."""

    SCENE_BOUNDARY = "scene_boundary"
    """Sample keyframes at scene change boundaries."""

    DETECTION_GUIDED = "detection_guided"
    """Sample frame crops guided by object perception bounding boxes."""


@dataclass
class OCRSchedulerConfig:
    """Configuration for OCR frame selection."""

    strategy: OCRSchedulingStrategy = OCRSchedulingStrategy.FIXED_INTERVAL
    interval_frames: int = 5
    frame_interval: int | None = None
    interval_seconds: float = 0.5
    temporal_interval_seconds: float | None = None
    scene_boundary_frames: set[int] = field(default_factory=set)
    scene_boundary_frame_indices: set[int] | None = None

    def __post_init__(self) -> None:
        if self.frame_interval is not None:
            self.interval_frames = self.frame_interval
        if self.temporal_interval_seconds is not None:
            self.interval_seconds = self.temporal_interval_seconds
        if self.scene_boundary_frame_indices is not None:
            self.scene_boundary_frames = set(self.scene_boundary_frame_indices)


class OCRScheduler:
    """Evaluates whether incoming video frames should undergo OCR processing."""

    def __init__(self, config: OCRSchedulerConfig | None = None) -> None:
        self.config = config or OCRSchedulerConfig()
        self._last_processed_timestamp: float = -1.0
        self._processed_count: int = 0

    def should_ocr(
        self,
        frame_or_index: DecodedFrame | Frame | int,
        timestamp_seconds: float | None = None,
    ) -> bool:
        """Alias for should_process, accepting DecodedFrame, Frame, or frame index."""
        if isinstance(frame_or_index, int):
            ts = 0.0 if timestamp_seconds is None else timestamp_seconds
            return self.should_process(frame_or_index, ts)
        elif isinstance(frame_or_index, DecodedFrame):
            return self.should_process(frame_or_index.frame_index, frame_or_index.timestamp_seconds)
        else:
            return self.should_process(
                frame_or_index.frame_number, frame_or_index.timestamp_seconds
            )

    def should_process(
        self,
        frame_number: int,
        timestamp_seconds: float,
    ) -> bool:
        """Determine if current frame should be sent to the OCR pipeline."""
        strategy = self.config.strategy

        if strategy == OCRSchedulingStrategy.ALL_FRAMES:
            self._processed_count += 1
            return True

        if strategy == OCRSchedulingStrategy.FIXED_INTERVAL:
            if frame_number % max(1, self.config.interval_frames) == 0:
                self._processed_count += 1
                return True
            return False

        if strategy == OCRSchedulingStrategy.TEMPORAL_INTERVAL:
            if (
                self._last_processed_timestamp < 0.0
                or (timestamp_seconds - self._last_processed_timestamp)
                >= self.config.interval_seconds
            ):
                self._last_processed_timestamp = timestamp_seconds
                self._processed_count += 1
                return True
            return False

        if strategy == OCRSchedulingStrategy.SCENE_BOUNDARY:
            if frame_number in self.config.scene_boundary_frames:
                self._processed_count += 1
                return True
            return False

        # Default fallback: process first frame and intervals
        if frame_number == 0 or frame_number % max(1, self.config.interval_frames) == 0:
            self._processed_count += 1
            return True
        return False

    @staticmethod
    def extract_region_crop(
        frame_array: np.ndarray,
        bbox: BoundingBox,
        margin_px: int = 4,
    ) -> np.ndarray:
        """Extract a bounding box sub-crop for detection-guided OCR.

        Clamps coordinates to frame boundaries with an optional safety margin.
        """
        h, w = frame_array.shape[:2]
        x1 = max(0, int(bbox.x) - margin_px)
        y1 = max(0, int(bbox.y) - margin_px)
        x2 = min(w, int(bbox.x + bbox.width) + margin_px)
        y2 = min(h, int(bbox.y + bbox.height) + margin_px)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((10, 10, 3), dtype=np.uint8)

        return frame_array[y1:y2, x1:x2].copy()

    @staticmethod
    def create_cropped_frame(
        original: DecodedFrame,
        bbox: BoundingBox,
        margin_px: int = 4,
    ) -> DecodedFrame:
        """Create a new DecodedFrame representing an object-guided ROI crop.

        Inherits the authoritative FrameTimestamp provenance from the source frame.
        """
        if not isinstance(original.frame_array, np.ndarray):
            raise TypeError(
                f"DecodedFrame.frame_array must be np.ndarray, got {type(original.frame_array)}"
            )
        crop_arr = OCRScheduler.extract_region_crop(original.frame_array, bbox, margin_px)
        return DecodedFrame(
            frame_index=original.frame_index,
            timestamp_seconds=original.timestamp_seconds,
            width=crop_arr.shape[1],
            height=crop_arr.shape[0],
            frame_array=crop_arr,
            video_id=original.video_id,
            frame_timestamp=original.frame_timestamp,
        )
