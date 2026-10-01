"""Deterministic visual crop extraction from bounding-box evidence."""

from __future__ import annotations

import math
from uuid import UUID

import cv2
import numpy as np

from videx.domain.schemas import BoundingBox, CoordinateType
from videx.ingestion.base import DecodedFrame
from videx.semantic.schemas import CropRegion


class CropExtractionError(ValueError):
    """Raised when crop geometry is invalid, degenerated, or outside frame bounds."""


class CropExtractor:
    """Extracts bounded visual crops with context padding and frame-boundary clamping.

    Produces CropRegion models treated as derived artifacts referencing canonical
    evidence, preserving Presentation Timestamps (PTS) and frame indices.
    """

    def __init__(self, default_context_margin: float = 0.15, jpeg_quality: int = 90) -> None:
        self.default_context_margin = max(0.0, default_context_margin)
        self.jpeg_quality = jpeg_quality

    def extract_crop(
        self,
        frame: DecodedFrame | np.ndarray,
        bbox: BoundingBox,
        source_evidence_id: UUID,
        context_margin: float | None = None,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
        encode_jpeg: bool = True,
    ) -> CropRegion:
        """Extract a clamped crop patch from a frame array or DecodedFrame.

        Args:
            frame: Either an in-memory DecodedFrame or a NumPy image array (H, W, C).
            bbox: BoundingBox indicating target spatial region.
            source_evidence_id: Canonical Evidence ID of the detection/observation.
            context_margin: Optional margin override (e.g. 0.15 for 15% context).
            frame_index: Explicit frame index override if frame is raw ndarray.
            timestamp_seconds: Explicit timestamp override if frame is raw ndarray.
            encode_jpeg: Whether to encode the cropped patch to JPEG bytes.

        Returns:
            CropRegion with original and clamped bounding boxes and image bytes.
        """
        margin = self.default_context_margin if context_margin is None else max(0.0, context_margin)

        # 1. Resolve frame array and dimensions
        if isinstance(frame, DecodedFrame):
            frame_arr = frame.frame_array
            if not isinstance(frame_arr, np.ndarray):
                raise CropExtractionError(
                    f"DecodedFrame array must be numpy.ndarray, got {type(frame_arr)}"
                )
            f_idx = frame.frame_index if frame_index is None else frame_index
            ts_sec = frame.timestamp_seconds if timestamp_seconds is None else timestamp_seconds
            img = frame_arr
        elif isinstance(frame, np.ndarray):
            f_idx = 0 if frame_index is None else frame_index
            ts_sec = 0.0 if timestamp_seconds is None else timestamp_seconds
            img = frame
        else:
            raise TypeError(f"frame must be DecodedFrame or numpy.ndarray, got {type(frame)}")

        if img.ndim < 2:
            raise CropExtractionError(
                f"Image array must have at least 2 dimensions, got shape {img.shape}"
            )

        frame_h, frame_w = img.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            raise CropExtractionError(f"Invalid frame dimensions ({frame_w}x{frame_h})")

        # 2. Convert normalized box to pixel coordinates if needed
        box_x, box_y, box_w, box_h = self._to_pixel_coords(bbox, frame_w, frame_h)

        # 3. Validate box geometry
        if (
            math.isnan(box_x)
            or math.isnan(box_y)
            or math.isnan(box_w)
            or math.isnan(box_h)
            or box_w < 1.0
            or box_h < 1.0
        ):
            raise CropExtractionError(
                f"Invalid or degenerate box coordinates: ({box_x:.1f}, {box_y:.1f}, "
                f"{box_w:.1f}, {box_h:.1f})"
            )

        # 4. Expand box with context margin
        exp_w = box_w * (1.0 + 2.0 * margin)
        exp_h = box_h * (1.0 + 2.0 * margin)
        exp_x = box_x - (box_w * margin)
        exp_y = box_y - (box_h * margin)

        # 5. Clamp to frame boundaries [0, W] and [0, H]
        clamp_x1 = max(0, int(math.floor(exp_x)))
        clamp_y1 = max(0, int(math.floor(exp_y)))
        clamp_x2 = min(frame_w, int(math.ceil(exp_x + exp_w)))
        clamp_y2 = min(frame_h, int(math.ceil(exp_y + exp_h)))

        clamp_w = max(0, clamp_x2 - clamp_x1)
        clamp_h = max(0, clamp_y2 - clamp_y1)

        # Minimum crop size check (at least 2x2 pixels)
        if clamp_w < 2 or clamp_h < 2:
            raise CropExtractionError(
                f"Clamped crop region ({clamp_w}x{clamp_h}) is too small to extract visual context"
            )

        clamped_box = BoundingBox(
            x=float(clamp_x1),
            y=float(clamp_y1),
            width=float(clamp_w),
            height=float(clamp_h),
            coordinate_type=CoordinateType.PIXEL,
        )

        # 6. Extract slice from NumPy array
        patch = img[clamp_y1:clamp_y2, clamp_x1:clamp_x2]

        image_bytes: bytes | None = None
        if encode_jpeg:
            success, enc_buf = cv2.imencode(
                ".jpg",
                patch,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if success and enc_buf is not None:
                image_bytes = bytes(enc_buf)

        return CropRegion(
            frame_index=f_idx,
            timestamp_seconds=round(ts_sec, 3),
            original_bbox=bbox,
            clamped_bbox=clamped_box,
            context_margin=margin,
            source_evidence_id=source_evidence_id,
            image_bytes=image_bytes,
        )

    @staticmethod
    def _to_pixel_coords(
        bbox: BoundingBox, frame_w: int, frame_h: int
    ) -> tuple[float, float, float, float]:
        """Convert BoundingBox to absolute pixel coordinates if normalized."""
        if bbox.coordinate_type == CoordinateType.NORMALIZED:
            return (
                bbox.x * frame_w,
                bbox.y * frame_h,
                bbox.width * frame_w,
                bbox.height * frame_h,
            )
        return bbox.x, bbox.y, bbox.width, bbox.height
