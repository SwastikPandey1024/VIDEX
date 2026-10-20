"""Metric calculation functions and statistical uncertainty estimators for VIDEX.

Provides mathematically verified implementations of:
- Bounding-box detection precision, recall, F1 (one-to-one greedy bipartite matching)
- OCR Character Error Rate (CER) and ASR Word Error Rate (WER) via Levenshtein edit distance
- Temporal event interval matching (tIoU) precision, recall, F1
- QA answerability and abstention classification metrics
- Structural evidence-reference integrity scoring
- Wilson score confidence interval estimation for proportions
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from typing import Any, TypeVar

from videx.domain.schemas import BoundingBox
from videx.evaluation.schemas import (
    DetectionPrediction,
    EventPrediction,
    GroundTruthDetection,
    GroundTruthEvent,
    GroundTruthQA,
    MetricResult,
    MetricStatus,
    QAPrediction,
)

__all__ = [
    "DetectionPrediction",
    "EventPrediction",
    "QAPrediction",
    "compute_cer",
    "compute_iou",
    "compute_temporal_iou",
    "compute_wer",
    "evaluate_detections",
    "evaluate_evidence_reference_integrity",
    "evaluate_qa_abstention",
    "evaluate_temporal_events",
    "levenshtein_edit_counts",
    "maximum_weight_bipartite_matching",
    "min_cost_assignment",
    "normalize_text",
    "wilson_score_interval",
]

T = TypeVar("T")


# ── Statistical Uncertainty Helpers ─────────────────────────────────────────


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Computes the Wilson score confidence interval for a binomial proportion.

    Unlike naive normal approximation (Wald interval), the Wilson score interval:
    - Never produces bounds outside [0.0, 1.0]
    - Handles zero successes (0/N) and complete successes (N/N)
      without collapsing or dividing by zero.

    Args:
        successes: Number of successful outcomes (k)
        total: Number of trials or sample units (n)
        confidence: Desired two-sided confidence level (default: 0.95, z ≈ 1.95996)

    Returns:
        (lower_bound, upper_bound) as floats in [0.0, 1.0], or None if total == 0.
    """
    if total <= 0:
        return None

    # Normal critical value for two-sided confidence
    # Standard normal quantile approximation for confidence:
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        # Fallback standard normal quantile approximation via inverse error function
        z = math.sqrt(2.0) * _erfinv(confidence)

    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def _erfinv(y: float) -> float:
    """Winitzki approximation of inverse error function for arbitrary confidence levels."""
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    ln_term = math.log(1.0 - y * y)
    term1 = 2.0 / (math.pi * a) + ln_term / 2.0
    inner = term1 * term1 - ln_term / a
    return sgn * math.sqrt(math.sqrt(inner) - term1)


# ── Levenshtein Edit Distance & Text Error Rates ────────────────────────────


def levenshtein_edit_counts(seq1: Sequence[T], seq2: Sequence[T]) -> tuple[int, int, int]:
    """Computes substitutions, deletions, and insertions between reference seq1 and hypothesis seq2.

    Returns:
        (substitutions, deletions, insertions)
    """
    m, n = len(seq1), len(seq2)
    # dp[i][j] = (total_cost, subs, dels, ins)
    # Optimized 2-row dynamic programming
    prev: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(n + 1)]

    for i in range(1, m + 1):
        curr: list[tuple[int, int, int, int]] = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * n
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                cost_sub = prev[j - 1][0] + 1
                cost_del = prev[j][0] + 1  # Deletion from reference
                cost_ins = curr[j - 1][0] + 1  # Insertion into reference

                min_cost = min(cost_sub, cost_del, cost_ins)
                if min_cost == cost_sub:
                    curr[j] = (cost_sub, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
                elif min_cost == cost_del:
                    curr[j] = (cost_del, prev[j][1], prev[j][2] + 1, prev[j][3])
                else:
                    curr[j] = (cost_ins, curr[j - 1][1], curr[j - 1][2], curr[j - 1][3] + 1)
        prev = curr

    _, subs, dels, ins = prev[n]
    return (subs, dels, ins)


def normalize_text(text: str, case_sensitive: bool = False, strip_punct: bool = False) -> str:
    """Documented normalization rules for OCR and ASR evaluation.

    Rules applied:
    - Unicode NFKC decomposition to standardize Hindi diacritics and ligatures.
    - Optional lowercase folding.
    - Consecutive whitespace collapsed into single space; leading/trailing stripped.
    """
    norm = unicodedata.normalize("NFKC", text)
    if not case_sensitive:
        norm = norm.lower()
    if strip_punct:
        norm = "".join(ch for ch in norm if not unicodedata.category(ch).startswith("P"))
    # Collapse multiple spaces
    return " ".join(norm.split())


def compute_cer(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> tuple[float, dict[str, int]]:
    """Computes Character Error Rate (CER) = (S + D + I) / N_ref.

    Returns:
        (cer_value, {"substitutions": S, "deletions": D, "insertions": I, "reference_chars": N_ref})
    """
    ref_norm = normalize_text(reference, case_sensitive=case_sensitive)
    hyp_norm = normalize_text(hypothesis, case_sensitive=case_sensitive)

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)
    n_ref = len(ref_chars)

    if n_ref == 0:
        # Edge case: reference is empty
        return (
            0.0 if len(hyp_chars) == 0 else 1.0,
            {
                "substitutions": 0,
                "deletions": 0,
                "insertions": len(hyp_chars),
                "reference_chars": 0,
            },
        )

    subs, dels, ins = levenshtein_edit_counts(ref_chars, hyp_chars)
    cer = (subs + dels + ins) / n_ref
    return (
        round(cer, 4),
        {"substitutions": subs, "deletions": dels, "insertions": ins, "reference_chars": n_ref},
    )


def compute_wer(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> tuple[float, dict[str, int]]:
    """Computes Word Error Rate (WER) = (S + D + I) / N_ref."""
    ref_norm = normalize_text(reference, case_sensitive=case_sensitive, strip_punct=True)
    hyp_norm = normalize_text(hypothesis, case_sensitive=case_sensitive, strip_punct=True)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    n_ref = len(ref_words)

    if n_ref == 0:
        return (
            0.0 if len(hyp_words) == 0 else 1.0,
            {
                "substitutions": 0,
                "deletions": 0,
                "insertions": len(hyp_words),
                "reference_words": 0,
            },
        )

    subs, dels, ins = levenshtein_edit_counts(ref_words, hyp_words)
    wer = (subs + dels + ins) / n_ref
    return (
        round(wer, 4),
        {"substitutions": subs, "deletions": dels, "insertions": ins, "reference_words": n_ref},
    )


# ── Bipartite Matching Algorithms ───────────────────────────────────────────


def min_cost_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Kuhn-Munkres (Hungarian) algorithm for n x m cost matrix.

    Finds a complete assignment of rows to columns minimizing total cost.
    Optimized O(n^2 * m) implementation in pure Python.
    """
    if not cost_matrix or not cost_matrix[0]:
        return []

    n = len(cost_matrix)
    m = len(cost_matrix[0])
    transposed = False

    if n > m:
        cost_matrix = [[cost_matrix[i][j] for i in range(n)] for j in range(m)]
        n, m = m, n
        transposed = True

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] > 0:
            row = p[j] - 1
            col = j - 1
            if transposed:
                assignment.append((col, row))
            else:
                assignment.append((row, col))

    assignment.sort(key=lambda x: x[0])
    return assignment


def maximum_weight_bipartite_matching(
    num_gt: int,
    num_pred: int,
    candidates: list[tuple[float, int, int]],
) -> list[tuple[int, int, float]]:
    """Finds maximum-cardinality, then maximum-weight bipartite matching.

    candidates: list of (weight, gt_idx, pred_idx) where weight is positive (e.g. IoU >= threshold).
    Returns list of matched (gt_idx, pred_idx, weight).

    To guarantee maximum cardinality first and sum of weights second, each valid edge receives
    weight = 100.0 + edge_weight (since edge_weight <= 1.0). Any additional match yields >= 100.0,
    strictly dominating any fractional weight trade-offs.
    """
    if num_gt == 0 or num_pred == 0 or not candidates:
        return []

    edge_map: dict[tuple[int, int], float] = {}
    for w, g, p in candidates:
        edge_map[(g, p)] = max(edge_map.get((g, p), 0.0), w)

    big_val = 1000.0
    cost_matrix: list[list[float]] = []
    for g in range(num_gt):
        row: list[float] = []
        for p in range(num_pred):
            if (g, p) in edge_map:
                w = edge_map[(g, p)]
                row.append(big_val - (100.0 + w))
            else:
                row.append(big_val)
        cost_matrix.append(row)

    assignment = min_cost_assignment(cost_matrix)
    matches: list[tuple[int, int, float]] = []
    for g, p in assignment:
        if (g, p) in edge_map:
            matches.append((g, p, edge_map[(g, p)]))

    return matches


# ── Bounding Box Detection Evaluation ───────────────────────────────────────


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes.

    Coordinates are specified in pixel or normalized space with origin (0, 0)
    at the top-left corner. Boxes with zero or negative width/height are clipped
    to zero area.
    """
    if box1.width <= 0.0 or box1.height <= 0.0 or box2.width <= 0.0 or box2.height <= 0.0:
        return 0.0

    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    union_area = box1.area + box2.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def evaluate_detections(
    ground_truth: Sequence[GroundTruthDetection],
    predictions: Sequence[DetectionPrediction],
    iou_threshold: float = 0.5,
    temporal_tolerance_seconds: float = 0.1,
    matching_strategy: str = "optimal",
) -> dict[str, MetricResult]:
    """Evaluates object detection using one-to-one bipartite matching.

    Coordinate conventions:
    - Bounding boxes use top-left origin (x, y) with non-negative width and height.
    - Non-positive width or height is clipped to 0 area.

    Matching criteria:
    1. Exact class_label match (case-insensitive).
    2. Temporal proximity within temporal_tolerance_seconds.
    3. Spatial overlap: IoU >= iou_threshold.
    4. One-to-one constraint: Each GT and prediction can be matched at most once.
       Duplicate predictions for the same object are strictly penalized as False Positives.
    5. Matching strategy:
       - 'optimal' (default): Maximum-cardinality, maximum-weight bipartite matching
         via the Kuhn-Munkres algorithm.
       - 'greedy': Greedy assignment sorted descending by IoU.
    """
    total_gt = len(ground_truth)
    total_pred = len(predictions)

    if total_gt == 0:
        return {
            "detection_precision": MetricResult(
                name="detection_precision",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth detections annotated",
            ),
            "detection_recall": MetricResult(
                name="detection_recall",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth detections annotated",
            ),
            "detection_f1": MetricResult(
                name="detection_f1",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth detections annotated",
            ),
        }

    # Generate candidate matches satisfying class, time, and IoU thresholds
    candidates: list[tuple[float, int, int]] = []
    for g_idx, gt in enumerate(ground_truth):
        for p_idx, pred in enumerate(predictions):
            if gt.class_label.lower() != pred.class_label.lower():
                continue
            if abs(gt.timestamp_pts - pred.timestamp_pts) > temporal_tolerance_seconds:
                continue
            iou = compute_iou(gt.bbox, pred.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, g_idx, p_idx))

    if matching_strategy == "greedy":
        candidates.sort(key=lambda x: x[0], reverse=True)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        tp = 0
        for _, g_idx, p_idx in candidates:
            if g_idx not in matched_gt and p_idx not in matched_pred:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                tp += 1
    else:
        # Default: optimal maximum-weight bipartite matching
        optimal_matches = maximum_weight_bipartite_matching(total_gt, total_pred, candidates)
        tp = len(optimal_matches)

    fp = total_pred - tp
    fn = total_gt - tp

    zero_pred_denom = (tp + fp) == 0
    zero_gt_denom = (tp + fn) == 0

    precision = (tp / (tp + fp)) if not zero_pred_denom else 0.0
    recall = (tp / (tp + fn)) if not zero_gt_denom else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    p_ci = wilson_score_interval(tp, tp + fp) if not zero_pred_denom else None
    r_ci = wilson_score_interval(tp, total_gt) if not zero_gt_denom else None

    p_meta: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_threshold": iou_threshold,
        "temporal_tolerance_seconds": temporal_tolerance_seconds,
        "matching_strategy": matching_strategy,
    }
    if zero_pred_denom:
        p_meta["zero_denominator"] = True
        p_meta["undefined_reason"] = (
            "Zero positive predictions made (TP + FP = 0); precision is formally undefined"
        )

    r_meta: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_ground_truth": total_gt,
        "matching_strategy": matching_strategy,
    }
    if zero_gt_denom:
        r_meta["zero_denominator"] = True
        r_meta["undefined_reason"] = (
            "Zero ground truth detections annotated (TP + FN = 0); recall is formally undefined"
        )

    f1_meta: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "matching_strategy": matching_strategy,
    }
    if zero_pred_denom or zero_gt_denom:
        f1_meta["zero_denominator"] = True

    return {
        "detection_precision": MetricResult(
            name="detection_precision",
            status=MetricStatus.AVAILABLE,
            value=round(precision, 4),
            sample_size=tp + fp,
            confidence_interval=p_ci,
            ci_method="wilson_score" if p_ci else None,
            metadata=p_meta,
        ),
        "detection_recall": MetricResult(
            name="detection_recall",
            status=MetricStatus.AVAILABLE,
            value=round(recall, 4),
            sample_size=total_gt,
            confidence_interval=r_ci,
            ci_method="wilson_score" if r_ci else None,
            metadata=r_meta,
        ),
        "detection_f1": MetricResult(
            name="detection_f1",
            status=MetricStatus.AVAILABLE,
            value=round(f1, 4),
            sample_size=total_gt,
            metadata=f1_meta,
        ),
    }


# ── Temporal Event Matching ─────────────────────────────────────────────────


def compute_temporal_iou(
    interval_1: tuple[float, float],
    interval_2: tuple[float, float],
) -> float:
    """Computes temporal Intersection over Union (tIoU) between two time intervals.

    Intervals are specified as (start_seconds, end_seconds).
    For non-zero duration intervals: tIoU = intersection_duration / union_duration.
    For point events (zero duration) at identical timestamps: tIoU = 1.0.
    """
    s1, e1 = interval_1
    s2, e2 = interval_2

    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    inter_dur = max(0.0, inter_end - inter_start)

    union_start = min(s1, s2)
    union_end = max(e1, e2)
    union_dur = max(0.0, union_end - union_start)

    if union_dur <= 0.0:
        return 1.0 if s1 == s2 else 0.0
    return inter_dur / union_dur


def evaluate_temporal_events(
    ground_truth: Sequence[GroundTruthEvent],
    predictions: Sequence[EventPrediction],
    tiou_threshold: float = 0.5,
    matching_strategy: str = "optimal",
) -> dict[str, MetricResult]:
    """Evaluates temporal events using one-to-one bipartite matching over tIoU."""
    total_gt = len(ground_truth)
    total_pred = len(predictions)

    if total_gt == 0:
        return {
            "event_precision": MetricResult(
                name="event_precision",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth events annotated",
            ),
            "event_recall": MetricResult(
                name="event_recall",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth events annotated",
            ),
            "event_f1": MetricResult(
                name="event_f1",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth events annotated",
            ),
        }

    candidates: list[tuple[float, int, int]] = []
    for g_idx, gt in enumerate(ground_truth):
        for p_idx, pred in enumerate(predictions):
            if gt.event_type.lower() != pred.event_type.lower():
                continue
            tiou = compute_temporal_iou(
                (gt.start_seconds, gt.end_seconds),
                (pred.start_seconds, pred.end_seconds),
            )
            if tiou >= tiou_threshold:
                candidates.append((tiou, g_idx, p_idx))

    boundary_errors: list[float] = []

    if matching_strategy == "greedy":
        candidates.sort(key=lambda x: x[0], reverse=True)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        tp = 0
        for _, g_idx, p_idx in candidates:
            if g_idx not in matched_gt and p_idx not in matched_pred:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                tp += 1
                gt = ground_truth[g_idx]
                pr = predictions[p_idx]
                err = (
                    abs(gt.start_seconds - pr.start_seconds) + abs(gt.end_seconds - pr.end_seconds)
                ) / 2.0
                boundary_errors.append(err)
    else:
        optimal_matches = maximum_weight_bipartite_matching(total_gt, total_pred, candidates)
        tp = len(optimal_matches)
        for g_idx, p_idx, _ in optimal_matches:
            gt = ground_truth[g_idx]
            pr = predictions[p_idx]
            err = (
                abs(gt.start_seconds - pr.start_seconds) + abs(gt.end_seconds - pr.end_seconds)
            ) / 2.0
            boundary_errors.append(err)

    fp = total_pred - tp
    fn = total_gt - tp

    zero_pred_denom = (tp + fp) == 0
    zero_gt_denom = (tp + fn) == 0

    precision = (tp / (tp + fp)) if not zero_pred_denom else 0.0
    recall = (tp / (tp + fn)) if not zero_gt_denom else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_boundary_err = sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0.0

    p_ci = wilson_score_interval(tp, tp + fp) if not zero_pred_denom else None
    r_ci = wilson_score_interval(tp, total_gt) if not zero_gt_denom else None

    p_meta: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tiou_threshold": tiou_threshold,
        "matching_strategy": matching_strategy,
    }
    if zero_pred_denom:
        p_meta["zero_denominator"] = True
        p_meta["undefined_reason"] = (
            "Zero positive event predictions (TP + FP = 0); precision is formally undefined"
        )

    r_meta: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_ground_truth": total_gt,
        "matching_strategy": matching_strategy,
    }
    if zero_gt_denom:
        r_meta["zero_denominator"] = True
        r_meta["undefined_reason"] = (
            "Zero ground truth events annotated (TP + FN = 0); recall is formally undefined"
        )

    return {
        "event_precision": MetricResult(
            name="event_precision",
            status=MetricStatus.AVAILABLE,
            value=round(precision, 4),
            sample_size=tp + fp,
            confidence_interval=p_ci,
            ci_method="wilson_score" if p_ci else None,
            metadata=p_meta,
        ),
        "event_recall": MetricResult(
            name="event_recall",
            status=MetricStatus.AVAILABLE,
            value=round(recall, 4),
            sample_size=total_gt,
            confidence_interval=r_ci,
            ci_method="wilson_score" if r_ci else None,
            metadata=r_meta,
        ),
        "event_f1": MetricResult(
            name="event_f1",
            status=MetricStatus.AVAILABLE,
            value=round(f1, 4),
            sample_size=total_gt,
            metadata={
                "precision": precision,
                "recall": recall,
                "mean_temporal_boundary_error_sec": round(mean_boundary_err, 4),
                "matching_strategy": matching_strategy,
            },
        ),
    }


# ── QA Answerability & Abstention Evaluation ────────────────────────────────


def evaluate_qa_abstention(
    ground_truth: Sequence[GroundTruthQA],
    predictions: Sequence[QAPrediction],
) -> dict[str, MetricResult]:
    """Evaluates question answerability and abstention classification.

    Explicitly models and scores:
    1. tp_abstain: Unanswerable question (is_answerable=False) where model explicitly
       reported insufficient evidence (did_abstain=True).
    2. tn_answer: Answerable question (is_answerable=True) where model attempted an answer
       (did_abstain=False).
    3. fp_refusal: Answerable question where model incorrectly refused to answer (did_abstain=True).
    4. fn_hallucination: Unanswerable question where model attempted an answer instead of abstaining
       (did_abstain=False).
    5. missing_predictions: Question annotated in ground truth for which no prediction was supplied.
       Missing predictions are NEVER scored as correct abstentions.
    """
    if not ground_truth:
        return {
            "qa_abstention_accuracy": MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No ground truth QA items annotated",
            )
        }

    pred_map = {str(p.qa_id): p for p in predictions}

    tp_abstain = 0
    tn_answer = 0
    fp_refusal = 0
    fn_hallucination = 0
    missing_predictions = 0

    unanswerable_total = 0
    answerable_total = 0

    for gt in ground_truth:
        if not gt.is_answerable:
            unanswerable_total += 1
        else:
            answerable_total += 1

        pred = pred_map.get(str(gt.qa_id))
        if pred is None:
            missing_predictions += 1
            continue

        if not gt.is_answerable:
            # Expected abstention
            if pred.did_abstain:
                tp_abstain += 1
            else:
                fn_hallucination += 1
        else:
            # Expected answer
            if not pred.did_abstain:
                tn_answer += 1
            else:
                fp_refusal += 1

    total_evaluated = tp_abstain + tn_answer + fp_refusal + fn_hallucination

    if total_evaluated == 0:
        return {
            "qa_abstention_accuracy": MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No matching predictions found for annotated QA items",
                metadata={"missing_predictions": missing_predictions},
            )
        }

    correct_decisions = tp_abstain + tn_answer
    accuracy = correct_decisions / total_evaluated

    # Abstention precision = tp / (tp + fp)
    total_abstained = tp_abstain + fp_refusal
    zero_abstain_denom = total_abstained == 0
    abstention_precision = (tp_abstain / total_abstained) if not zero_abstain_denom else 0.0

    # Abstention recall = tp / (tp + fn)
    unanswerable_evaluated = tp_abstain + fn_hallucination
    zero_unans_denom = unanswerable_evaluated == 0
    abstention_recall = (tp_abstain / unanswerable_evaluated) if not zero_unans_denom else 0.0

    # Abstention F1
    if (abstention_precision + abstention_recall) > 0.0:
        abstention_f1 = (
            2.0
            * abstention_precision
            * abstention_recall
            / (abstention_precision + abstention_recall)
        )
    else:
        abstention_f1 = 0.0

    confusion_meta: dict[str, Any] = {
        "total_ground_truth": len(ground_truth),
        "total_evaluated": total_evaluated,
        "missing_predictions": missing_predictions,
        "answerable_total": answerable_total,
        "unanswerable_total": unanswerable_total,
        "tp_abstain": tp_abstain,
        "tn_answer": tn_answer,
        "fp_refusal": fp_refusal,
        "fn_hallucination": fn_hallucination,
        # Backward-compatible aliases
        "correct_abstentions": tp_abstain,
        "hallucinated_answers": fn_hallucination,
        "refused_answerable": fp_refusal,
    }

    acc_ci = wilson_score_interval(correct_decisions, total_evaluated)
    prec_ci = wilson_score_interval(tp_abstain, total_abstained) if not zero_abstain_denom else None
    rec_ci = (
        wilson_score_interval(tp_abstain, unanswerable_evaluated) if not zero_unans_denom else None
    )

    return {
        "qa_abstention_accuracy": MetricResult(
            name="qa_abstention_accuracy",
            status=MetricStatus.AVAILABLE,
            value=round(accuracy, 4),
            sample_size=total_evaluated,
            confidence_interval=acc_ci,
            ci_method="wilson_score" if acc_ci else None,
            metadata={**confusion_meta, "correct_decisions": correct_decisions},
        ),
        "qa_abstention_precision": MetricResult(
            name="qa_abstention_precision",
            status=MetricStatus.AVAILABLE,
            value=round(abstention_precision, 4),
            sample_size=total_abstained,
            confidence_interval=prec_ci,
            ci_method="wilson_score" if prec_ci else None,
            metadata={
                **confusion_meta,
                "zero_denominator": zero_abstain_denom,
                "undefined_reason": (
                    "No questions were abstained on" if zero_abstain_denom else None
                ),
            },
        ),
        "qa_abstention_recall": MetricResult(
            name="qa_abstention_recall",
            status=MetricStatus.AVAILABLE if not zero_unans_denom else MetricStatus.NOT_APPLICABLE,
            value=round(abstention_recall, 4) if not zero_unans_denom else None,
            sample_size=unanswerable_evaluated,
            confidence_interval=rec_ci,
            ci_method="wilson_score" if rec_ci else None,
            metadata=confusion_meta,
            reason_if_unavailable=(
                "No unanswerable questions evaluated" if zero_unans_denom else None
            ),
        ),
        "qa_abstention_f1": MetricResult(
            name="qa_abstention_f1",
            status=MetricStatus.AVAILABLE,
            value=round(abstention_f1, 4),
            sample_size=total_evaluated,
            metadata={
                "precision": abstention_precision,
                "recall": abstention_recall,
                "confusion_matrix": confusion_meta,
            },
        ),
    }


# ── Structural Evidence-Reference Integrity ─────────────────────────────────


def evaluate_evidence_reference_integrity(
    cited_evidence_ids: Sequence[str],
    canonical_store_ids: set[str],
    target_video_id: str,
    evidence_video_map: dict[str, str],
) -> MetricResult:
    """Evaluates structural reference integrity of cited evidence.

    Checks:
    1. Does the cited ID exist in the canonical store?
    2. Does the cited ID belong to target_video_id (cross-video isolation)?

    NOTE: This is strictly a structural metric. It does NOT claim that the
    cited evidence semantically entails the claim.
    """
    total = len(cited_evidence_ids)
    if total == 0:
        return MetricResult(
            name="evidence_reference_integrity",
            status=MetricStatus.NOT_APPLICABLE,
            reason_if_unavailable="No evidence IDs were cited to evaluate",
        )

    valid_refs = 0
    cross_video_violations = 0
    missing_ids = 0

    for eid in cited_evidence_ids:
        if eid not in canonical_store_ids:
            missing_ids += 1
            continue
        owner_video = evidence_video_map.get(eid)
        if owner_video != target_video_id:
            cross_video_violations += 1
            continue
        valid_refs += 1

    rate = valid_refs / total
    return MetricResult(
        name="evidence_reference_integrity",
        status=MetricStatus.AVAILABLE,
        value=round(rate, 4),
        sample_size=total,
        confidence_interval=wilson_score_interval(valid_refs, total),
        ci_method="wilson_score",
        metadata={
            "total_cited": total,
            "valid_references": valid_refs,
            "missing_ids": missing_ids,
            "cross_video_violations": cross_video_violations,
        },
    )
