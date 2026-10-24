"""Tests for the targeted 12-frame PyTorch vs ONNX Runtime parity diagnostic.

Validates:
  1. Structural integrity and completeness of targeted_12frame_parity.json
  2. Cold start (model load + warmup) isolation
  3. High-confidence perception parity (100% of person detections matched)
  4. Root-cause isolation for unmatched detections
  5. Mathematical parity metrics (median IoU >= 0.98, zero class mismatches)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

TARGETED_JSON_PATH = Path("outputs/profiling/targeted_12frame_parity.json")
EXPECTED_FRAMES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


@pytest.fixture(scope="module")
def targeted_parity_data() -> dict[str, Any]:
    if not TARGETED_JSON_PATH.exists():
        pytest.skip(f"Targeted parity JSON not found: {TARGETED_JSON_PATH}")
    with TARGETED_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


class TestTargetedParitySchema:
    def test_top_level_keys(self, targeted_parity_data: dict[str, Any]) -> None:
        for key in ("timestamp_utc", "target_frames", "cold_start",
                    "config_a_videx_detector_defaults", "config_b_ultralytics_benchmark_defaults"):
            assert key in targeted_parity_data, f"Missing key: {key!r}"

    def test_target_frames_list(self, targeted_parity_data: dict[str, Any]) -> None:
        assert targeted_parity_data["target_frames"] == EXPECTED_FRAMES

    def test_cold_start_latencies(self, targeted_parity_data: dict[str, Any]) -> None:
        cs = targeted_parity_data["cold_start"]
        for key in ("pt_model_load_ms", "ort_model_load_ms", "pt_warmup_ms", "ort_warmup_ms"):
            assert cs[key] > 0, f"Cold start latency {key} must be positive"


class TestTargetedParityMetrics:
    def test_median_iou_meets_gate(self, targeted_parity_data: dict[str, Any]) -> None:
        ca = targeted_parity_data["config_a_videx_detector_defaults"]
        cb = targeted_parity_data["config_b_ultralytics_benchmark_defaults"]
        ca_iou = ca["metrics"]["median_iou"]
        cb_iou = cb["metrics"]["median_iou"]
        assert ca_iou >= 0.98, f"Config A median IoU {ca_iou} < 0.98"
        assert cb_iou >= 0.98, f"Config B median IoU {cb_iou} < 0.98"

    def test_zero_class_mismatches(self, targeted_parity_data: dict[str, Any]) -> None:
        ca = targeted_parity_data["config_a_videx_detector_defaults"]
        cb = targeted_parity_data["config_b_ultralytics_benchmark_defaults"]
        assert ca["counts"]["class_mismatches"] == 0
        assert cb["counts"]["class_mismatches"] == 0

    def test_exact_count_agreement_rate(self, targeted_parity_data: dict[str, Any]) -> None:
        """In Config A, at least 11 of the 12 sampled frames have identical detection counts."""
        ca = targeted_parity_data["config_a_videx_detector_defaults"]
        assert ca["counts"]["exact_count_frames"] >= 11

    def test_person_class_perfect_parity(self, targeted_parity_data: dict[str, Any]) -> None:
        """Verify that every single person detection across all 12 frames is matched."""
        ca = targeted_parity_data["config_a_videx_detector_defaults"]
        person_ious = []
        for frame in ca["per_frame"]:
            person_matches = [
                m for m in frame["matched"]
                if m["pt_cls"] == 0
            ]
            f_idx = frame["frame_index"]
            assert len(person_matches) == 1, (
                f"Frame {f_idx} expected 1 matched person detection, got {len(person_matches)}"
            )
            person_ious.append(person_matches[0]["iou"])

        import numpy as np
        median_person_iou = float(np.median(person_ious))
        assert median_person_iou >= 0.98, f"Median person IoU {median_person_iou} < 0.98"
        # 10 of 12 frames have near-perfect IoU >= 0.99;
        # frames 150 and 180 have reaching/head-glare boundary divergence
        assert sum(1 for iou in person_ious if iou >= 0.98) >= 10

    def test_frame_210_unmatched_root_cause(self, targeted_parity_data: dict[str, Any]) -> None:
        """Confirm frame 210's single unmatched detection is dining table sub-threshold in ONNX."""
        ca = targeted_parity_data["config_a_videx_detector_defaults"]
        frame_210 = next(f for f in ca["per_frame"] if f["frame_index"] == 210)
        assert frame_210["unmatched_pt_count"] == 1
        assert frame_210["unmatched_ort_count"] == 0
        un_pt = frame_210["unmatched_pt"][0]
        assert un_pt["class_name"] == "dining table"
        assert "sub_threshold_in_onnx" in un_pt["cause"]
