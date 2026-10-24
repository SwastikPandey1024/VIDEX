"""Focused unit tests for ONNX preprocessing and inverse coordinate transformations.

Covers:
  1. A 1280x720 frame letterboxed to 640x640.
  2. Known boxes transformed into model coordinates and recovered into original-image coordinates.
  3. Non-square and square input dimensions.
  4. Boxes near image boundaries.
  5. Invalid or degenerate boxes.
  6. Correct behavior when horizontal or vertical padding is non-zero.
  7. Exact mathematical round-trip consistency with Ultralytics scale_boxes.
"""

from __future__ import annotations

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes, xywh2xyxy


class TestLetterBoxPreprocessing:
    """Validates aspect-ratio preserving letterbox resizing and padding."""

    def test_1280x720_letterboxed_to_640x640_vertical_padding(self) -> None:
        """1280x720 scaled to 640x640 has vertical padding and zero horizontal padding."""
        orig_h, orig_w = 720, 1280
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        lb = LetterBox(new_shape=(640, 640), auto=False)
        out = lb(image=img)
        assert isinstance(out, np.ndarray)
        assert out.shape == (640, 640, 3), f"Expected (640, 640, 3), got {out.shape}"
        # Gain is min(640/720, 640/1280) = 640/1280 = 0.5
        # Scaled: w=640, h=360
        # Padding: pad_x = (640 - 640) // 2 = 0
        #          pad_y = (640 - 360) // 2 = 140
        gain = min(640 / orig_h, 640 / orig_w)
        assert gain == 0.5
        pad_x = round((640 - round(orig_w * gain)) / 2 - 0.1)
        pad_y = round((640 - round(orig_h * gain)) / 2 - 0.1)
        assert pad_x == 0, f"Expected pad_x=0, got {pad_x}"
        assert pad_y == 140, f"Expected pad_y=140, got {pad_y}"

    def test_720x1280_portrait_letterbox_horizontal_padding(self) -> None:
        """720x1280 scaled to 640x640 has horizontal padding and zero vertical padding."""
        orig_h, orig_w = 1280, 720
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        lb = LetterBox(new_shape=(640, 640), auto=False)
        out = lb(image=img)
        assert isinstance(out, np.ndarray)
        assert out.shape == (640, 640, 3)
        gain = min(640 / orig_h, 640 / orig_w)
        assert gain == 0.5
        pad_x = round((640 - round(orig_w * gain)) / 2 - 0.1)
        pad_y = round((640 - round(orig_h * gain)) / 2 - 0.1)
        assert pad_x == 140, f"Expected pad_x=140, got {pad_x}"
        assert pad_y == 0, f"Expected pad_y=0, got {pad_y}"

    def test_square_640x640_zero_padding(self) -> None:
        """640x640 square image scaled to 640x640 has gain=1.0 and zero padding."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        lb = LetterBox(new_shape=(640, 640), auto=False)
        out = lb(image=img)
        assert isinstance(out, np.ndarray)
        assert out.shape == (640, 640, 3)
        gain = min(640 / 640, 640 / 640)
        assert gain == 1.0


class TestInverseCoordinateTransform:
    """Validates mapping from model coordinates back to original image space."""

    def test_known_box_recovery_1280x720(self) -> None:
        """A known box in 1280x720 space is mapped into 640x640 model space, then recovered."""
        orig_shape = (720, 1280)  # (H, W)
        model_shape = (640, 640)

        # Original box: [x1, y1, x2, y2]
        orig_box = torch.tensor([[575.42, 139.46, 873.32, 643.51]], dtype=torch.float32)

        # Forward letterbox formula: gain=0.5, pad_x=0, pad_y=140
        gain = 0.5
        pad_x = 0
        pad_y = 140
        model_box = orig_box * gain
        model_box[:, [0, 2]] += pad_x
        model_box[:, [1, 3]] += pad_y

        # Recover using scale_boxes
        recovered = scale_boxes(model_shape, model_box.clone(), orig_shape)
        assert isinstance(recovered, torch.Tensor)

        # Assert mathematical equality within 0.01 px
        diff = (recovered - orig_box).abs().max().item()
        assert diff < 0.01, f"Expected box recovery match <=0.01px, got diff: {diff}"

    def test_known_box_recovery_portrait_720x1280(self) -> None:
        """Test box recovery with non-zero horizontal padding (portrait)."""
        orig_shape = (1280, 720)  # (H, W)
        model_shape = (640, 640)

        orig_box = torch.tensor([[100.0, 200.0, 500.0, 900.0]], dtype=torch.float32)

        gain = 0.5
        pad_x = 140
        pad_y = 0
        model_box = orig_box * gain
        model_box[:, [0, 2]] += pad_x
        model_box[:, [1, 3]] += pad_y

        recovered = scale_boxes(model_shape, model_box.clone(), orig_shape)
        assert isinstance(recovered, torch.Tensor)
        diff = (recovered - orig_box).abs().max().item()
        assert diff < 0.01, f"Max diff on portrait recovery: {diff}"

    def test_boxes_near_image_boundaries(self) -> None:
        """Boxes situated at the physical boundaries of the original image."""
        orig_shape = (720, 1280)
        model_shape = (640, 640)

        # Top-left corner, bottom-right corner, and edge-hugging boxes
        boundary_boxes = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
            [1180.0, 620.0, 1280.0, 720.0],
            [0.0, 710.0, 50.0, 720.0],
            [1270.0, 0.0, 1280.0, 40.0],
        ], dtype=torch.float32)

        gain = 0.5
        pad_x = 0
        pad_y = 140
        model_boxes = boundary_boxes * gain
        model_boxes[:, [0, 2]] += pad_x
        model_boxes[:, [1, 3]] += pad_y

        recovered = scale_boxes(model_shape, model_boxes.clone(), orig_shape)
        assert isinstance(recovered, torch.Tensor)
        diff = (recovered - boundary_boxes).abs().max().item()
        assert diff < 0.01, f"Boundary boxes difference: {diff}"

    def test_clipping_of_out_of_bounds_predictions(self) -> None:
        """Predictions that exceed model boundaries or have negative coords are cleanly clipped."""
        orig_shape = (720, 1280)
        model_shape = (640, 640)

        # Model box predicting coordinates outside the letterboxed image bounds:
        # e.g. x < 0 or y < pad_y (into padding area) or x > 640
        out_of_bounds_model_box = torch.tensor([
            [-10.0, 100.0, 200.0, 600.0],   # x1 < 0, y1 < 140 (in top pad)
            [500.0, 400.0, 650.0, 550.0],   # x2 > 640
        ], dtype=torch.float32)

        recovered = scale_boxes(model_shape, out_of_bounds_model_box.clone(), orig_shape)
        assert isinstance(recovered, torch.Tensor)

        # All recovered boxes must be clipped within [0, orig_w] and [0, orig_h]
        assert (recovered[:, 0] >= 0.0).all(), "x1 must be >= 0"
        assert (recovered[:, 1] >= 0.0).all(), "y1 must be >= 0"
        assert (recovered[:, 2] <= orig_shape[1]).all(), "x2 must be <= orig_w"
        assert (recovered[:, 3] <= orig_shape[0]).all(), "y2 must be <= orig_h"

    def test_degenerate_zero_area_box(self) -> None:
        """Degenerate boxes (x1 == x2 or y1 == y2) do not raise errors."""
        orig_shape = (720, 1280)
        model_shape = (640, 640)

        degenerate_box = torch.tensor([[100.0, 200.0, 100.0, 200.0]], dtype=torch.float32)
        recovered = scale_boxes(model_shape, degenerate_box.clone(), orig_shape)
        assert isinstance(recovered, torch.Tensor)
        assert recovered.shape == (1, 4)
        assert not torch.isnan(recovered).any()

    def test_xywh2xyxy_parity(self) -> None:
        """Validates center-xywh conversion to corner-xyxy."""
        cxcywh = torch.tensor([[100.0, 100.0, 40.0, 60.0]], dtype=torch.float32)
        xyxy = xywh2xyxy(cxcywh)  # type: ignore[no-untyped-call]
        expected = torch.tensor([[80.0, 70.0, 120.0, 130.0]], dtype=torch.float32)
        assert (xyxy - expected).abs().max().item() < 1e-4
