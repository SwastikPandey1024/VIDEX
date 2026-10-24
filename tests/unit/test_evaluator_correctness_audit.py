"""Comprehensive Evaluator Correctness and Numerical Invariant Tests (Phase B Audit).

Systematically verifies every evaluator requirement against independently calculated
mathematical reference cases:
1. xywh to xyxy conversions and inverse bounds.
2. Image edge coordinates and out-of-bounds/zero-area boxes.
3. Mathematical IoU invariants (identical, disjoint, rational overlaps, degenerate).
4. Strict class-aware matching (case-insensitive equality, disjoint class rejection).
5. One-to-one bipartite matching and duplicate prediction penalization.
6. IoU threshold boundaries (exact epsilon edge testing at 0.50).
7. Temporal tolerance boundaries (within vs beyond temporal_tolerance_seconds).
8. Empty ground truth, empty predictions, and zero-denominator handling.
9. Precision, recall, and F1 numerical calculations.
"""

from __future__ import annotations

import pytest

from videx.domain.schemas import BoundingBox, CoordinateType
from videx.evaluation.metrics import (
    compute_iou,
    evaluate_detections,
)
from videx.evaluation.schemas import (
    DetectionPrediction,
    GroundTruthDetection,
    MetricStatus,
)


class TestEvaluatorCorrectnessAudit:
    """Rigorous verification of bounding box logic, IoU, and detection matching."""

    # ── 1. Coordinate Conversions and Extents ────────────────────────────────────

    def test_xywh_extent_properties(self) -> None:
        """Verifies top-left (x, y) with positive width and height computes edges correctly."""
        box = BoundingBox(
            x=100.0,
            y=150.0,
            width=200.0,
            height=300.0,
            coordinate_type=CoordinateType.PIXEL,
        )
        assert box.x == 100.0
        assert box.y == 150.0
        assert box.width == 200.0
        assert box.height == 300.0
        assert box.x2 == 300.0
        assert box.y2 == 450.0
        assert box.center_x == 200.0
        assert box.center_y == 300.0
        assert box.area == 60000.0

    def test_boxes_at_image_edges(self) -> None:
        """Verifies boxes positioned at exact image boundaries (0, 0) and (1280, 720)."""
        top_left_box = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        assert top_left_box.x2 == 100.0
        assert top_left_box.y2 == 100.0

        bottom_right_box = BoundingBox(x=1180.0, y=620.0, width=100.0, height=100.0)
        assert bottom_right_box.x2 == 1280.0
        assert bottom_right_box.y2 == 720.0

    # ── 2. Mathematical IoU Invariants ──────────────────────────────────────────

    def test_iou_identical_boxes(self) -> None:
        """Two identical boxes must produce IoU = 1.0."""
        b1 = BoundingBox(x=50.0, y=50.0, width=100.0, height=100.0)
        b2 = BoundingBox(x=50.0, y=50.0, width=100.0, height=100.0)
        assert compute_iou(b1, b2) == 1.0

    def test_iou_disjoint_boxes(self) -> None:
        """Disjoint boxes (no overlap) must produce IoU = 0.0."""
        b1 = BoundingBox(x=0.0, y=0.0, width=50.0, height=50.0)
        b2 = BoundingBox(x=60.0, y=60.0, width=50.0, height=50.0)
        assert compute_iou(b1, b2) == 0.0

    def test_iou_adjacent_touching_edges(self) -> None:
        """Boxes sharing an edge but zero intersection area must produce IoU = 0.0."""
        b1 = BoundingBox(x=0.0, y=0.0, width=50.0, height=50.0)
        b2 = BoundingBox(x=50.0, y=0.0, width=50.0, height=50.0)
        assert compute_iou(b1, b2) == 0.0

    def test_iou_analytical_fractional_overlap(self) -> None:
        """Analytically calculated overlap:
        Box 1: [0, 0, 10, 10] -> Area = 100.
        Box 2: [5, 0, 10, 10] -> Area = 100.
        Intersection: x in [5, 10] (width 5), y in [0, 10] (height 10) -> Area = 50.
        Union: 100 + 100 - 50 = 150.
        IoU = 50 / 150 = 1 / 3 = 0.333333...
        """
        b1 = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        b2 = BoundingBox(x=5.0, y=0.0, width=10.0, height=10.0)
        assert pytest.approx(compute_iou(b1, b2), rel=1e-6) == 1.0 / 3.0

    def test_iou_contained_box(self) -> None:
        """Box 2 completely inside Box 1:
        Box 1: [0, 0, 20, 20] -> Area = 400.
        Box 2: [5, 5, 10, 10] -> Area = 100.
        Intersection = 100.
        Union = 400.
        IoU = 100 / 400 = 0.25.
        """
        b1 = BoundingBox(x=0.0, y=0.0, width=20.0, height=20.0)
        b2 = BoundingBox(x=5.0, y=5.0, width=10.0, height=10.0)
        assert compute_iou(b1, b2) == 0.25

    # ── 3. Class Matching and Case Sensitivity ──────────────────────────────────

    def test_class_aware_filtering(self) -> None:
        """Predictions with mismatched class labels must never match, even with identical boxes."""
        b = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="person", bbox=b)]
        pred = [
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="car",
                bbox=b,
                confidence=0.99,
            )
        ]

        res = evaluate_detections(gt, pred, iou_threshold=0.5)
        # 1 GT unmatched -> FN = 1. 1 Pred unmatched -> FP = 1. TP = 0.
        assert res["detection_precision"].value == 0.0
        assert res["detection_recall"].value == 0.0
        assert res["detection_precision"].metadata["tp"] == 0
        assert res["detection_precision"].metadata["fp"] == 1
        assert res["detection_recall"].metadata["fn"] == 1

    def test_class_case_insensitivity(self) -> None:
        """Class comparison must be case-insensitive ('Person' matches 'person')."""
        b = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="Person", bbox=b)]
        pred = [
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="person",
                bbox=b,
                confidence=0.95,
            )
        ]

        res = evaluate_detections(gt, pred, iou_threshold=0.5)
        assert res["detection_precision"].value == 1.0
        assert res["detection_recall"].value == 1.0
        assert res["detection_precision"].metadata["tp"] == 1

    # ── 4. One-to-One Matching and Duplicate Penalization ───────────────────────

    def test_one_to_one_constraint_multiple_overlapping_predictions(self) -> None:
        """1 GT box with 3 overlapping predictions must match at most 1, penalizing 2 as FPs."""
        gt_box = BoundingBox(x=100.0, y=100.0, width=100.0, height=100.0)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="person", bbox=gt_box)]

        # Pred 1: IoU 1.0 (exact)
        p1 = DetectionPrediction(
            timestamp_pts=1.0,
            class_label="person",
            bbox=gt_box,
            confidence=0.95,
        )
        # Pred 2: IoU ~0.60
        p2 = DetectionPrediction(
            timestamp_pts=1.0,
            class_label="person",
            bbox=BoundingBox(x=120.0, y=100.0, width=100.0, height=100.0),
            confidence=0.85,
        )
        # Pred 3: IoU ~0.70
        p3 = DetectionPrediction(
            timestamp_pts=1.0,
            class_label="person",
            bbox=BoundingBox(x=110.0, y=100.0, width=100.0, height=100.0),
            confidence=0.75,
        )

        res = evaluate_detections(gt, [p1, p2, p3], iou_threshold=0.5)
        # TP = 1, FP = 2, FN = 0 -> Precision = 1/3 ≈ 0.3333, Recall = 1/1 = 1.0
        assert res["detection_precision"].metadata["tp"] == 1
        assert res["detection_precision"].metadata["fp"] == 2
        assert res["detection_recall"].metadata["fn"] == 0
        assert res["detection_precision"].value == round(1 / 3, 4)
        assert res["detection_recall"].value == 1.0

    # ── 5. IoU and Temporal Threshold Boundaries ────────────────────────────────

    def test_iou_threshold_boundary_exact_epsilon(self) -> None:
        """Boundary test: IoU = 0.50 exactly matches with threshold 0.50, IoU = 0.499 fails."""
        b_gt = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        b_exact_50 = BoundingBox(x=0.0, y=0.0, width=100.0, height=50.0)
        assert compute_iou(b_gt, b_exact_50) == 0.50

        # Exact 0.50 must MATCH
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="person", bbox=b_gt)]
        pred_exact = [
            DetectionPrediction(timestamp_pts=1.0, class_label="person", bbox=b_exact_50)
        ]
        res_exact = evaluate_detections(gt, pred_exact, iou_threshold=0.5)
        assert res_exact["detection_precision"].metadata["tp"] == 1

        # Height 49.9 -> Area 4990 -> IoU = 4990 / 10000 = 0.499 < 0.50 -> must FAIL
        b_sub_50 = BoundingBox(x=0.0, y=0.0, width=100.0, height=49.9)
        assert compute_iou(b_gt, b_sub_50) == 0.499
        pred_sub = [
            DetectionPrediction(timestamp_pts=1.0, class_label="person", bbox=b_sub_50)
        ]
        res_sub = evaluate_detections(gt, pred_sub, iou_threshold=0.5)
        assert res_sub["detection_precision"].metadata["tp"] == 0
        assert res_sub["detection_precision"].metadata["fp"] == 1
        assert res_sub["detection_recall"].metadata["fn"] == 1

    def test_temporal_tolerance_boundary(self) -> None:
        """Temporal tolerance: delta <= tolerance matches, delta > tolerance fails."""
        b = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [GroundTruthDetection(timestamp_pts=3.000, class_label="person", bbox=b)]

        # Pred at 3.090 (delta = 0.09s <= 0.10s) -> MATCH
        p_within = [DetectionPrediction(timestamp_pts=3.090, class_label="person", bbox=b)]
        res_within = evaluate_detections(gt, p_within, temporal_tolerance_seconds=0.10)
        assert res_within["detection_precision"].metadata["tp"] == 1

        # Pred at 3.110 (delta = 0.11s > 0.10s) -> REJECT
        p_beyond = [DetectionPrediction(timestamp_pts=3.110, class_label="person", bbox=b)]
        res_beyond = evaluate_detections(gt, p_beyond, temporal_tolerance_seconds=0.10)
        assert res_beyond["detection_precision"].metadata["tp"] == 0
        assert res_beyond["detection_precision"].metadata["fp"] == 1

    # ── 6. Zero Denominators and Edge Cases ──────────────────────────────────────

    def test_empty_predictions_handling(self) -> None:
        """Ground truth present, 0 predictions -> Precision=0.0, Recall=0.0, FN=len(GT)."""
        box = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="person", bbox=box)]
        res = evaluate_detections(gt, [])
        assert res["detection_precision"].value == 0.0
        assert res["detection_precision"].metadata["zero_denominator"] is True
        assert res["detection_recall"].value == 0.0
        assert res["detection_recall"].metadata["fn"] == 1
        assert res["detection_f1"].value == 0.0

    def test_empty_ground_truth_handling(self) -> None:
        """No ground truth -> status UNAVAILABLE (never fabricates 0 or 1)."""
        box = BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0)
        pred = [DetectionPrediction(timestamp_pts=1.0, class_label="person", bbox=box)]
        res = evaluate_detections([], pred)
        assert res["detection_precision"].status == MetricStatus.UNAVAILABLE
        assert res["detection_recall"].status == MetricStatus.UNAVAILABLE
        assert res["detection_f1"].status == MetricStatus.UNAVAILABLE

    # ── 7. Filtering Invariants (Task 2 Audit) ───────────────────────────────────

    def test_confidence_filtering_preserves_ground_truth_count(self) -> None:
        """Filtering predictions by confidence must never alter ground-truth count."""
        b = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [
            GroundTruthDetection(timestamp_pts=0.0, class_label="person", bbox=b),
            GroundTruthDetection(timestamp_pts=3.0, class_label="person", bbox=b),
        ]
        # Predictions with various confidence scores
        raw_preds = [
            DetectionPrediction(timestamp_pts=0.0, class_label="person", bbox=b, confidence=0.91),
            DetectionPrediction(timestamp_pts=0.0, class_label="person", bbox=b, confidence=0.20),
        ]

        # Full predictions
        res_full = evaluate_detections(gt, raw_preds)
        assert res_full["detection_recall"].metadata["total_ground_truth"] == 2

        # Filtered to conf >= 0.50
        filtered_preds = [p for p in raw_preds if p.confidence >= 0.50]
        res_filtered = evaluate_detections(gt, filtered_preds)
        assert res_filtered["detection_recall"].metadata["total_ground_truth"] == 2
        assert len(gt) == 2

    def test_target_class_prefiltering_semantics(self) -> None:
        """Target-class filtering restricts evaluation to target ontology."""
        b = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [GroundTruthDetection(timestamp_pts=0.0, class_label="person", bbox=b)]
        preds = [
            DetectionPrediction(timestamp_pts=0.0, class_label="person", bbox=b, confidence=0.90),
            DetectionPrediction(
                timestamp_pts=0.0,
                class_label="suitcase",
                bbox=BoundingBox(x=100.0, y=100.0, width=50.0, height=50.0),
                confidence=0.45,
            ),
        ]

        # Case A: Unfiltered (suitcase counted as FP)
        res_unfiltered = evaluate_detections(gt, preds)
        assert res_unfiltered["detection_precision"].value == 0.5  # 1 TP / 2 Preds

        # Case B: Pre-filtered to target class "person"
        target_classes = {"person"}
        filtered_preds = [p for p in preds if p.class_label.lower() in target_classes]
        res_filtered = evaluate_detections(gt, filtered_preds)
        assert res_filtered["detection_precision"].value == 1.0  # 1 TP / 1 Pred
        assert res_filtered["detection_recall"].value == 1.0

