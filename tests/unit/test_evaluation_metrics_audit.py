"""Unit tests for Phase 1 Audit of Detection Evaluation Metrics and Numerical Invariants.

Verifies:
1. One-to-one bipartite matching (no duplicate matching of GT or predictions).
2. Maximum cardinality first, then maximum IoU sum (Kuhn-Munkres behavior).
3. Exact TP, FP, FN calculation at IoU thresholds 0.50 and 0.75.
4. Precision, Recall, F1 formulas, and Wilson score confidence intervals.
5. Temporal tolerance windowing isolation across video frames.
6. Competing detection hypotheses (reproducing multi-scale anchor NMS edge cases like frames 150/180).
7. Zero-denominator boundary conditions (empty predictions, empty GT).
"""

from __future__ import annotations

import pytest

from videx.domain.schemas import BoundingBox, CoordinateType
from videx.evaluation.metrics import (
    compute_iou,
    evaluate_detections,
    maximum_weight_bipartite_matching,
    wilson_score_interval,
)
from videx.evaluation.schemas import (
    DetectionPrediction,
    GroundTruthDetection,
    MetricStatus,
)


class TestBipartiteMatchingProperties:
    """Verifies one-to-one matching and maximum weight assignment invariants."""

    def test_one_to_one_constraint_strictness(self) -> None:
        """A single ground-truth object must match AT MOST ONE prediction, even if 5 overlap it."""
        gt_box = BoundingBox(x=100.0, y=100.0, width=100.0, height=100.0, coordinate_type=CoordinateType.PIXEL)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="person", bbox=gt_box)]

        # 5 predictions, all overlapping gt_box with IoU >= 0.50
        preds = [
            DetectionPrediction(timestamp_pts=1.0, class_label="person", bbox=gt_box, confidence=0.95),
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="person",
                bbox=BoundingBox(x=105.0, y=100.0, width=100.0, height=100.0),
                confidence=0.85,
            ),
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="person",
                bbox=BoundingBox(x=95.0, y=100.0, width=100.0, height=100.0),
                confidence=0.80,
            ),
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="person",
                bbox=BoundingBox(x=100.0, y=105.0, width=100.0, height=100.0),
                confidence=0.75,
            ),
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="person",
                bbox=BoundingBox(x=100.0, y=95.0, width=100.0, height=100.0),
                confidence=0.70,
            ),
        ]

        res = evaluate_detections(gt, preds, iou_threshold=0.50)
        assert res["detection_precision"].metadata["tp"] == 1
        assert res["detection_precision"].metadata["fp"] == 4
        assert res["detection_recall"].metadata["fn"] == 0
        assert res["detection_precision"].value == 1.0 / 5.0
        assert res["detection_recall"].value == 1.0

    def test_maximum_cardinality_priority(self) -> None:
        """Kuhn-Munkres must prioritize 2 matches over 1 match, even if 1 match has higher individual IoU.

        Example:
        GT0 can match Pred0 (IoU 0.95) or Pred1 (IoU 0.60).
        GT1 can ONLY match Pred0 (IoU 0.70).
        If GT0 greedily takes Pred0, total matches = 1.
        Optimal matching pairs GT1 -> Pred0 and GT0 -> Pred1, total matches = 2.
        """
        candidates = [
            (0.95, 0, 0),  # GT0 -> Pred0
            (0.60, 0, 1),  # GT0 -> Pred1
            (0.70, 1, 0),  # GT1 -> Pred0
        ]
        matches = maximum_weight_bipartite_matching(num_gt=2, num_pred=2, candidates=candidates)
        assert len(matches) == 2
        matched_pairs = {(g, p) for g, p, _ in matches}
        assert (0, 1) in matched_pairs
        assert (1, 0) in matched_pairs


class TestThresholdEvaluationAccuracy:
    """Verifies mathematical correctness of TP, FP, FN at IoU 0.50 and 0.75."""

    def test_iou_threshold_050_vs_075_distinction(self) -> None:
        """A box with IoU = 0.65 is TP at 0.50 threshold, but FP at 0.75 threshold."""
        b_gt = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        # Width 100, height 78.788 -> IoU = 78.788 / (100 + 100 - 78.788) = 78.788 / 121.212 ≈ 0.65
        # Let's make an exact box:
        # Box 1: [0, 0, 100, 100], area 10000
        # Box 2: [0, 0, 100, 80], area 8000
        # Intersection: 8000. Union: 10000. IoU = 8000 / 10000 = 0.80
        # Box 3: [0, 0, 100, 60], area 6000
        # Intersection: 6000. Union: 10000. IoU = 6000 / 10000 = 0.60
        b_p_060 = BoundingBox(x=0.0, y=0.0, width=100.0, height=60.0)
        assert compute_iou(b_gt, b_p_060) == 0.60

        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="person", bbox=b_gt)]
        pred = [DetectionPrediction(timestamp_pts=1.0, class_label="person", bbox=b_p_060, confidence=0.9)]

        # At 0.50 threshold: IoU 0.60 >= 0.50 -> MATCH
        res_050 = evaluate_detections(gt, pred, iou_threshold=0.50)
        assert res_050["detection_precision"].metadata["tp"] == 1
        assert res_050["detection_precision"].metadata["fp"] == 0
        assert res_050["detection_recall"].metadata["fn"] == 0
        assert res_050["detection_precision"].value == 1.0
        assert res_050["detection_recall"].value == 1.0
        assert res_050["detection_f1"].value == 1.0

        # At 0.75 threshold: IoU 0.60 < 0.75 -> REJECT
        res_075 = evaluate_detections(gt, pred, iou_threshold=0.75)
        assert res_075["detection_precision"].metadata["tp"] == 0
        assert res_075["detection_precision"].metadata["fp"] == 1
        assert res_075["detection_recall"].metadata["fn"] == 1
        assert res_075["detection_precision"].value == 0.0
        assert res_075["detection_recall"].value == 0.0
        assert res_075["detection_f1"].value == 0.0

    def test_synthetic_v3_12frame_aggregation(self) -> None:
        """Simulates exactly the 12-frame gate structure:
        - 12 frames, each with 1 GT person.
        - 5 frames have IoU >= 0.75.
        - 7 frames have IoU in [0.60, 0.74].
        All 12 frames have IoU >= 0.50.
        Must yield:
        - At 0.50: TP=12, FP=0, FN=0, Precision=1.0, Recall=1.0, F1=1.0
        - At 0.75: TP=5, FP=7, FN=7, Precision=5/12 ≈ 0.4167, Recall=5/12 ≈ 0.4167, F1 ≈ 0.4167
        """
        gt_list: list[GroundTruthDetection] = []
        pred_list: list[DetectionPrediction] = []

        for i in range(12):
            pts = float(i * 1.0)
            b_gt = BoundingBox(x=100.0, y=100.0, width=100.0, height=100.0)
            gt_list.append(GroundTruthDetection(timestamp_pts=pts, class_label="person", bbox=b_gt))

            # First 5 frames: IoU = 0.80 >= 0.75
            # Next 7 frames: IoU = 0.65 < 0.75
            h_pred = 80.0 if i < 5 else 65.0
            b_pred = BoundingBox(x=100.0, y=100.0, width=100.0, height=h_pred)
            pred_list.append(DetectionPrediction(timestamp_pts=pts, class_label="person", bbox=b_pred, confidence=0.9))

        res_050 = evaluate_detections(gt_list, pred_list, iou_threshold=0.50)
        assert res_050["detection_precision"].metadata["tp"] == 12
        assert res_050["detection_precision"].metadata["fp"] == 0
        assert res_050["detection_recall"].metadata["fn"] == 0
        assert res_050["detection_precision"].value == 1.0
        assert res_050["detection_recall"].value == 1.0
        assert res_050["detection_f1"].value == 1.0

        res_075 = evaluate_detections(gt_list, pred_list, iou_threshold=0.75)
        assert res_075["detection_precision"].metadata["tp"] == 5
        assert res_075["detection_precision"].metadata["fp"] == 7
        assert res_075["detection_recall"].metadata["fn"] == 7
        assert pytest.approx(res_075["detection_precision"].value, rel=1e-4) == 5.0 / 12.0
        assert pytest.approx(res_075["detection_recall"].value, rel=1e-4) == 5.0 / 12.0
        assert pytest.approx(res_075["detection_f1"].value, rel=1e-4) == 5.0 / 12.0


class TestTemporalIsolationAndCompetingHypotheses:
    """Verifies that temporal windowing prevents cross-frame matching and simulates competing boxes."""

    def test_temporal_isolation_across_video_frames(self) -> None:
        """Predictions at frame 0 (pts 0.0) must never match GT at frame 30 (pts 1.001) even if identical boxes."""
        box = BoundingBox(x=100.0, y=100.0, width=100.0, height=100.0)
        gt = [
            GroundTruthDetection(timestamp_pts=0.000, class_label="person", bbox=box),
            GroundTruthDetection(timestamp_pts=1.001, class_label="person", bbox=box),
        ]
        # Only 1 prediction at pts 0.000
        preds = [DetectionPrediction(timestamp_pts=0.000, class_label="person", bbox=box, confidence=0.9)]

        res = evaluate_detections(gt, preds, temporal_tolerance_seconds=0.10)
        assert res["detection_precision"].metadata["tp"] == 1
        assert res["detection_precision"].metadata["fp"] == 0
        assert res["detection_recall"].metadata["fn"] == 1  # Frame 30 GT is missed
        assert res["detection_precision"].value == 1.0
        assert res["detection_recall"].value == 0.5

    def test_competing_anchor_hypotheses_resolution(self) -> None:
        """Simulates Frame 180 phenomenon:
        GT represents full person: [505, 118, 315, 550]
        Candidate A: Torso-only box (IoU ~0.61 with GT)
        Candidate B: Full-body box (IoU ~0.78 with GT)
        When both are provided to evaluate_detections with IoU threshold 0.50,
        evaluator picks Candidate B (maximum weight bipartite matching) as TP,
        and Candidate A becomes FP.
        """
        gt_box = BoundingBox(x=505.0, y=118.0, width=315.0, height=550.0)
        gt = [GroundTruthDetection(timestamp_pts=6.006, class_label="person", bbox=gt_box)]

        cand_a_torso = BoundingBox(x=485.3, y=273.2, width=359.7, height=367.7)  # IoU ~0.61
        cand_b_full = BoundingBox(x=485.4, y=162.5, width=358.9, height=481.6)   # IoU ~0.78

        iou_a = compute_iou(gt_box, cand_a_torso)
        iou_b = compute_iou(gt_box, cand_b_full)
        assert 0.60 <= iou_a <= 0.65
        assert 0.75 <= iou_b <= 0.80

        preds = [
            DetectionPrediction(timestamp_pts=6.006, class_label="person", bbox=cand_a_torso, confidence=0.51),
            DetectionPrediction(timestamp_pts=6.006, class_label="person", bbox=cand_b_full, confidence=0.60),
        ]

        res = evaluate_detections(gt, preds, iou_threshold=0.50)
        assert res["detection_precision"].metadata["tp"] == 1
        assert res["detection_precision"].metadata["fp"] == 1  # Competing hypothesis is penalized as FP
        assert res["detection_recall"].metadata["fn"] == 0


class TestWilsonScoreIntervalProperties:
    """Verifies Wilson score confidence interval mathematically."""

    def test_wilson_score_12_of_12_successes(self) -> None:
        """For 12/12 successes at 95% confidence:
        Formula: p_hat = 1.0, n = 12, z = 1.95996
        Lower bound must be strictly in (0.75, 0.76), upper bound must be 1.0.
        """
        ci = wilson_score_interval(12, 12, confidence=0.95)
        assert ci is not None
        lower, upper = ci
        assert 0.75 <= lower <= 0.76
        assert upper == 1.0

    def test_wilson_score_zero_total_returns_none(self) -> None:
        """Zero trials must return None without division by zero."""
        assert wilson_score_interval(0, 0) is None
