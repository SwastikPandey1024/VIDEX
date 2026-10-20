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
from dataclasses import dataclass
from typing import TypeVar

from videx.domain.schemas import BoundingBox
from videx.evaluation.schemas import (
    GroundTruthDetection,
    GroundTruthEvent,
    GroundTruthQA,
    MetricResult,
    MetricStatus,
)

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


# ── Bounding Box Detection Evaluation ───────────────────────────────────────


@dataclass(frozen=True)
class DetectionPrediction:
    """Prediction format consumed by the detection evaluator."""

    timestamp_pts: float
    class_label: str
    bbox: BoundingBox
    confidence: float = 1.0


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
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
) -> dict[str, MetricResult]:
    """Evaluates object detection using one-to-one greedy bipartite matching.

    Matching criteria:
    1. Exact class_label match.
    2. Temporal proximity within temporal_tolerance_seconds.
    3. Spatial IoU >= iou_threshold.
    4. One-to-one greedy assignment (highest IoU first). Unmatched predictions are False Positives.
       Unmatched ground truths are False Negatives.
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

    # Generate candidate pairs
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

    # Sort descending by IoU for greedy assignment
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0

    for _, g_idx, p_idx in candidates:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            tp += 1

    fp = total_pred - tp
    fn = total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    p_ci = wilson_score_interval(tp, tp + fp)
    r_ci = wilson_score_interval(tp, total_gt)

    return {
        "detection_precision": MetricResult(
            name="detection_precision",
            status=MetricStatus.AVAILABLE,
            value=round(precision, 4),
            sample_size=tp + fp,
            confidence_interval=p_ci,
            ci_method="wilson_score",
            metadata={"tp": tp, "fp": fp, "fn": fn, "iou_threshold": iou_threshold},
        ),
        "detection_recall": MetricResult(
            name="detection_recall",
            status=MetricStatus.AVAILABLE,
            value=round(recall, 4),
            sample_size=total_gt,
            confidence_interval=r_ci,
            ci_method="wilson_score",
            metadata={"tp": tp, "fp": fp, "fn": fn, "total_ground_truth": total_gt},
        ),
        "detection_f1": MetricResult(
            name="detection_f1",
            status=MetricStatus.AVAILABLE,
            value=round(f1, 4),
            sample_size=total_gt,
            metadata={"precision": precision, "recall": recall},
        ),
    }


# ── Temporal Event Matching ─────────────────────────────────────────────────


@dataclass(frozen=True)
class EventPrediction:
    """Prediction format consumed by the temporal event evaluator."""

    event_type: str
    start_seconds: float
    end_seconds: float
    participants: list[str] = ()  # type: ignore[assignment]
    confidence: float = 1.0


def compute_temporal_iou(
    interval_1: tuple[float, float],
    interval_2: tuple[float, float],
) -> float:
    """Computes temporal Intersection over Union (tIoU) between two time intervals."""
    s1, e1 = interval_1
    s2, e2 = interval_2

    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    inter_dur = max(0.0, inter_end - inter_start)

    union_start = min(s1, s2)
    union_end = max(e1, e2)
    union_dur = max(0.0, union_end - union_start)

    if union_dur <= 0.0:
        return 0.0
    return inter_dur / union_dur


def evaluate_temporal_events(
    ground_truth: Sequence[GroundTruthEvent],
    predictions: Sequence[EventPrediction],
    tiou_threshold: float = 0.5,
) -> dict[str, MetricResult]:
    """Evaluates temporal events using one-to-one greedy assignment over tIoU."""
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

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0
    boundary_errors: list[float] = []

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

    fp = total_pred - tp
    fn = total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_boundary_err = sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0.0

    return {
        "event_precision": MetricResult(
            name="event_precision",
            status=MetricStatus.AVAILABLE,
            value=round(precision, 4),
            sample_size=tp + fp,
            confidence_interval=wilson_score_interval(tp, tp + fp),
            ci_method="wilson_score",
            metadata={"tp": tp, "fp": fp, "fn": fn, "tiou_threshold": tiou_threshold},
        ),
        "event_recall": MetricResult(
            name="event_recall",
            status=MetricStatus.AVAILABLE,
            value=round(recall, 4),
            sample_size=total_gt,
            confidence_interval=wilson_score_interval(tp, total_gt),
            ci_method="wilson_score",
            metadata={"tp": tp, "fp": fp, "fn": fn, "total_ground_truth": total_gt},
        ),
        "event_f1": MetricResult(
            name="event_f1",
            status=MetricStatus.AVAILABLE,
            value=round(f1, 4),
            sample_size=total_gt,
            metadata={"mean_temporal_boundary_error_sec": round(mean_boundary_err, 4)},
        ),
    }


# ── QA Answerability & Abstention Evaluation ────────────────────────────────


@dataclass(frozen=True)
class QAPrediction:
    """Prediction format consumed by the QA answerability evaluator."""

    qa_id: str
    predicted_answer: str | None
    did_abstain: bool  # True if the model said "insufficient evidence" / abstained
    confidence: float = 1.0


def evaluate_qa_abstention(
    ground_truth: Sequence[GroundTruthQA],
    predictions: Sequence[QAPrediction],
) -> dict[str, MetricResult]:
    """Evaluates question answerability and abstention classification.

    Explicitly scores the system's ability to say 'insufficient evidence'
    when a question is unanswerable (is_answerable=False).
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

    correct_decisions = 0
    total_evaluated = 0
    unanswerable_total = 0
    correct_abstentions = 0
    hallucinated_answers = 0  # Answered an unanswerable question
    refused_answerable = 0  # Abstained on an answerable question

    for gt in ground_truth:
        pred = pred_map.get(str(gt.qa_id))
        if pred is None:
            continue

        total_evaluated += 1
        expected_unanswerable = not gt.is_answerable

        if expected_unanswerable:
            unanswerable_total += 1
            if pred.did_abstain:
                correct_abstentions += 1
                correct_decisions += 1
            else:
                hallucinated_answers += 1
        else:
            # Expected answerable
            if not pred.did_abstain:
                correct_decisions += 1
            else:
                refused_answerable += 1

    if total_evaluated == 0:
        return {
            "qa_abstention_accuracy": MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable="No matching predictions found for annotated QA items",
            )
        }

    accuracy = correct_decisions / total_evaluated
    abstention_recall = correct_abstentions / unanswerable_total if unanswerable_total > 0 else 0.0

    return {
        "qa_abstention_accuracy": MetricResult(
            name="qa_abstention_accuracy",
            status=MetricStatus.AVAILABLE,
            value=round(accuracy, 4),
            sample_size=total_evaluated,
            confidence_interval=wilson_score_interval(correct_decisions, total_evaluated),
            ci_method="wilson_score",
            metadata={
                "total_evaluated": total_evaluated,
                "correct_decisions": correct_decisions,
                "unanswerable_total": unanswerable_total,
                "correct_abstentions": correct_abstentions,
                "hallucinated_answers": hallucinated_answers,
                "refused_answerable": refused_answerable,
            },
        ),
        "qa_abstention_recall": MetricResult(
            name="qa_abstention_recall",
            status=MetricStatus.AVAILABLE
            if unanswerable_total > 0
            else MetricStatus.NOT_APPLICABLE,
            value=round(abstention_recall, 4) if unanswerable_total > 0 else None,
            sample_size=unanswerable_total,
            confidence_interval=wilson_score_interval(correct_abstentions, unanswerable_total)
            if unanswerable_total > 0
            else None,
            ci_method="wilson_score",
            metadata={
                "correct_abstentions": correct_abstentions,
                "unanswerable_total": unanswerable_total,
            },
            reason_if_unavailable="No unanswerable questions present in sample"
            if unanswerable_total == 0
            else None,
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
