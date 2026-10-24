"""Unit tests for VIDEX evaluation metrics and statistical estimators."""

from __future__ import annotations

from videx.domain.schemas import BoundingBox
from videx.evaluation.metrics import (
    DetectionPrediction,
    EventPrediction,
    QAPrediction,
    compute_cer,
    compute_temporal_iou,
    compute_wer,
    evaluate_detections,
    evaluate_evidence_reference_integrity,
    evaluate_qa_abstention,
    evaluate_temporal_events,
    levenshtein_edit_counts,
    normalize_text,
    wilson_score_interval,
)
from videx.evaluation.schemas import (
    GroundTruthDetection,
    GroundTruthEvent,
    GroundTruthQA,
    MetricStatus,
)


class TestStatisticalEstimators:
    """Verifies Wilson score confidence intervals and error estimators."""

    def test_wilson_interval_zero_trials(self) -> None:
        assert wilson_score_interval(successes=0, total=0) is None

    def test_wilson_interval_zero_successes(self) -> None:
        interval = wilson_score_interval(successes=0, total=10)
        assert interval is not None
        low, high = interval
        assert low == 0.0
        assert 0.0 < high < 0.35

    def test_wilson_interval_all_successes(self) -> None:
        interval = wilson_score_interval(successes=10, total=10)
        assert interval is not None
        low, high = interval
        assert 0.65 < low < 1.0
        assert high == 1.0

    def test_wilson_interval_fifty_percent(self) -> None:
        interval = wilson_score_interval(successes=50, total=100)
        assert interval is not None
        low, high = interval
        assert 0.40 <= low <= 0.42
        assert 0.58 <= high <= 0.60


class TestTextMetrics:
    """Verifies CER and WER Levenshtein edit distance and normalization."""

    def test_levenshtein_counts(self) -> None:
        # "cat" -> "hat": 1 substitution
        subs, dels, ins = levenshtein_edit_counts("cat", "hat")
        assert (subs, dels, ins) == (1, 0, 0)

        # "cat" -> "cats": 1 insertion
        subs, dels, ins = levenshtein_edit_counts("cat", "cats")
        assert (subs, dels, ins) == (0, 0, 1)

        # "cat" -> "at": 1 deletion
        subs, dels, ins = levenshtein_edit_counts("cat", "at")
        assert (subs, dels, ins) == (0, 1, 0)

    def test_compute_cer_exact_match(self) -> None:
        cer, meta = compute_cer("VIDEX Pipeline", "VIDEX Pipeline")
        assert cer == 0.0
        assert meta["substitutions"] == 0

    def test_compute_cer_with_errors(self) -> None:
        # ref="abc" (3 chars), hyp="axc" (1 sub) -> CER = 1/3 ≈ 0.3333
        cer, meta = compute_cer("abc", "axc")
        assert cer == 0.3333
        assert meta["substitutions"] == 1

    def test_compute_wer_exact_match(self) -> None:
        wer, meta = compute_wer("the worker wears a hard hat", "the worker wears a hard hat")
        assert wer == 0.0
        assert meta["reference_words"] == 6

    def test_compute_wer_substitution(self) -> None:
        # 6 words, 1 substitution ("yellow" instead of "hard")
        wer, meta = compute_wer("the worker wears a hard hat", "the worker wears a yellow hat")
        assert wer == round(1 / 6, 4)
        assert meta["substitutions"] == 1

    def test_text_normalization_unicode(self) -> None:
        # NFKC compatibility normalization
        raw = "नमस्ते   दुनिया!  "
        norm = normalize_text(raw, strip_punct=True)
        assert norm == "नमस्ते दुनिया"


class TestDetectionEvaluation:
    """Verifies greedy bipartite bounding-box matching and metrics."""

    def test_perfect_detection_matching(self) -> None:
        b1 = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="car", bbox=b1)]
        pred = [DetectionPrediction(timestamp_pts=1.0, class_label="car", bbox=b1)]

        results = evaluate_detections(gt, pred, iou_threshold=0.5)
        assert results["detection_precision"].value == 1.0
        assert results["detection_recall"].value == 1.0
        assert results["detection_f1"].value == 1.0

    def test_duplicate_prediction_penalized_as_fp(self) -> None:
        # 1 GT box, 2 identical predictions -> 1 TP, 1 FP -> Precision = 0.5, Recall = 1.0
        b1 = BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0)
        gt = [GroundTruthDetection(timestamp_pts=1.0, class_label="car", bbox=b1)]
        pred = [
            DetectionPrediction(timestamp_pts=1.0, class_label="car", bbox=b1),
            DetectionPrediction(timestamp_pts=1.0, class_label="car", bbox=b1),
        ]

        results = evaluate_detections(gt, pred, iou_threshold=0.5)
        assert results["detection_precision"].value == 0.5
        assert results["detection_recall"].value == 1.0
        assert results["detection_precision"].metadata["fp"] == 1

    def test_empty_ground_truth_returns_unavailable(self) -> None:
        pred = [
            DetectionPrediction(
                timestamp_pts=1.0,
                class_label="car",
                bbox=BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0),
            )
        ]
        results = evaluate_detections([], pred)
        assert results["detection_precision"].status == MetricStatus.UNAVAILABLE
        assert results["detection_recall"].status == MetricStatus.UNAVAILABLE


class TestTemporalEventEvaluation:
    """Verifies temporal IoU and event interval matching."""

    def test_temporal_iou_calculation(self) -> None:
        # [0, 4] and [2, 6] -> inter [2, 4] (len 2), union [0, 6] (len 6) -> IoU = 2/6 ≈ 0.3333
        iou = compute_temporal_iou((0.0, 4.0), (2.0, 6.0))
        assert round(iou, 4) == 0.3333

        # Disjoint intervals
        assert compute_temporal_iou((0.0, 2.0), (3.0, 5.0)) == 0.0

    def test_evaluate_temporal_events(self) -> None:
        gt = [GroundTruthEvent(event_type="loitering", start_seconds=2.0, end_seconds=8.0)]
        pred = [EventPrediction(event_type="loitering", start_seconds=2.5, end_seconds=7.5)]

        results = evaluate_temporal_events(gt, pred, tiou_threshold=0.5)
        assert results["event_precision"].value == 1.0
        assert results["event_recall"].value == 1.0
        assert results["event_f1"].value == 1.0


class TestQAAbstentionEvaluation:
    """Verifies question answering and abstention scoring."""

    def test_qa_abstention_evaluation(self) -> None:
        gt = [
            GroundTruthQA(
                question="Is there a car?",
                is_answerable=True,
                reference_answer="Yes, a car is present.",
            ),
            GroundTruthQA(
                question="What is the driver's phone number?",
                is_answerable=False,
                reference_answer=None,
            ),
        ]

        # Case 1: Ideal model (answers Q1, abstains on Q2)
        preds_ideal = [
            QAPrediction(
                qa_id=str(gt[0].qa_id), predicted_answer="A car is visible", did_abstain=False
            ),
            QAPrediction(qa_id=str(gt[1].qa_id), predicted_answer=None, did_abstain=True),
        ]
        res_ideal = evaluate_qa_abstention(gt, preds_ideal)
        assert res_ideal["qa_abstention_accuracy"].value == 1.0
        assert res_ideal["qa_abstention_recall"].value == 1.0

        # Case 2: Hallucinating model (answers both)
        preds_hallucinating = [
            QAPrediction(qa_id=str(gt[0].qa_id), predicted_answer="Yes", did_abstain=False),
            QAPrediction(qa_id=str(gt[1].qa_id), predicted_answer="555-0199", did_abstain=False),
        ]
        res_hall = evaluate_qa_abstention(gt, preds_hallucinating)
        assert res_hall["qa_abstention_accuracy"].value == 0.5
        assert res_hall["qa_abstention_recall"].value == 0.0
        assert res_hall["qa_abstention_accuracy"].metadata["hallucinated_answers"] == 1


class TestEvidenceReferenceIntegrity:
    """Verifies structural evidence ID and cross-video scoping checks."""

    def test_evidence_reference_integrity(self) -> None:
        canonical_ids = {"ev_1", "ev_2", "ev_cross"}
        evidence_video_map = {
            "ev_1": "vid_target",
            "ev_2": "vid_target",
            "ev_cross": "vid_other",
        }

        # 3 citations:
        # ev_1: valid
        # ev_cross: belongs to vid_other (cross-video violation)
        # ev_missing: missing from canonical store
        res = evaluate_evidence_reference_integrity(
            cited_evidence_ids=["ev_1", "ev_cross", "ev_missing"],
            canonical_store_ids=canonical_ids,
            target_video_id="vid_target",
            evidence_video_map=evidence_video_map,
        )

        assert res.status == MetricStatus.AVAILABLE
        assert res.value == round(1 / 3, 4)
        assert res.metadata["valid_references"] == 1
        assert res.metadata["cross_video_violations"] == 1
        assert res.metadata["missing_ids"] == 1
