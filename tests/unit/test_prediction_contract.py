"""Unit tests validating the VIDEX prediction ingestion contract and status distinctions.

Verifies that:
- Controlled prediction fixtures produce mathematically exact, independently calculated metrics.
- Missing prediction files cannot masquerade as zero model accuracy.
- Explicitly empty predictions are cleanly distinguished from missing prediction files.
- Cross-video predictions cannot match across video boundaries.
- Missing modality predictions return explicit UNAVAILABLE statuses.
"""

from __future__ import annotations

from pathlib import Path

from videx.evaluation.runner import (
    EvaluationHarness,
    load_annotation_dataset,
    load_predictions_dataset,
)
from videx.evaluation.schemas import MetricStatus


class TestPredictionIngestionContract:
    """Verifies all controlled prediction fixtures and edge cases."""

    def setup_method(self) -> None:
        self.annotations_path = Path("tests/fixtures/evaluation/sample_annotations.json")
        self.dataset = load_annotation_dataset(self.annotations_path)
        self.fixtures_dir = Path("tests/fixtures/evaluation/predictions")

    def test_all_predictions_correct(self) -> None:
        """Scenario 1: All predictions correct -> TP=2, FP=0, FN=0, P=1.0, R=1.0, F1=1.0."""
        preds = load_predictions_dataset(self.fixtures_dir / "all_correct.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        # Detections
        p = v.metrics["detection_precision"]
        r = v.metrics["detection_recall"]
        f1 = v.metrics["detection_f1"]
        assert p.status == MetricStatus.AVAILABLE and p.value == 1.0
        assert r.status == MetricStatus.AVAILABLE and r.value == 1.0
        assert f1.status == MetricStatus.AVAILABLE and f1.value == 1.0
        assert p.metadata["tp"] == 2
        assert p.metadata["fp"] == 0
        assert p.metadata["fn"] == 0

        # Events
        ev_p = v.metrics["event_precision"]
        ev_r = v.metrics["event_recall"]
        ev_f1 = v.metrics["event_f1"]
        assert ev_p.status == MetricStatus.AVAILABLE and ev_p.value == 1.0
        assert ev_r.status == MetricStatus.AVAILABLE and ev_r.value == 1.0
        assert ev_f1.status == MetricStatus.AVAILABLE and ev_f1.value == 1.0

        # QA
        qa_acc = v.metrics["qa_abstention_accuracy"]
        assert qa_acc.status == MetricStatus.AVAILABLE and qa_acc.value == 1.0
        assert qa_acc.metadata["correct_decisions"] == 2
        assert qa_acc.metadata["tp_abstain"] == 1
        assert qa_acc.metadata["tn_answer"] == 1

    def test_one_false_positive(self) -> None:
        """Scenario 2: One false positive -> TP=2, FP=1, FN=0, P=2/3=0.6667, R=1.0, F1=0.8000."""
        preds = load_predictions_dataset(self.fixtures_dir / "one_false_positive.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        r = v.metrics["detection_recall"]
        f1 = v.metrics["detection_f1"]
        assert p.value == 0.6667
        assert r.value == 1.0
        assert f1.value == 0.8
        assert p.metadata["tp"] == 2
        assert p.metadata["fp"] == 1
        assert p.metadata["fn"] == 0

    def test_one_false_negative(self) -> None:
        """Scenario 3: One false negative -> TP=1, FP=0, FN=1, P=1.0, R=1/2=0.5000, F1=0.6667."""
        preds = load_predictions_dataset(self.fixtures_dir / "one_false_negative.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        r = v.metrics["detection_recall"]
        f1 = v.metrics["detection_f1"]
        assert p.value == 1.0
        assert r.value == 0.5
        assert f1.value == 0.6667
        assert p.metadata["tp"] == 1
        assert p.metadata["fp"] == 0
        assert p.metadata["fn"] == 1

    def test_duplicate_prediction_penalized(self) -> None:
        """Scenario 4: Duplicate predictions for one ground truth object.
        GT 0 is predicted twice. Exactly one prediction matches; the duplicate is penalized as FP.
        TP=2, FP=1, FN=0 -> P=0.6667, R=1.0, F1=0.8000.
        """
        preds = load_predictions_dataset(self.fixtures_dir / "duplicate_predictions.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        assert p.metadata["tp"] == 2
        assert p.metadata["fp"] == 1
        assert p.metadata["fn"] == 0
        assert p.value == 0.6667

    def test_wrong_class_overlapping_box(self) -> None:
        """Scenario 5: Wrong class with spatial overlap -> does not match.
        GT 0 has box matching Pred 0, but class is truck instead of car.
        TP=1, FP=1, FN=1 -> P=0.5000, R=0.5000, F1=0.5000.
        """
        preds = load_predictions_dataset(self.fixtures_dir / "wrong_class_overlapping.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        r = v.metrics["detection_recall"]
        f1 = v.metrics["detection_f1"]
        assert p.metadata["tp"] == 1
        assert p.metadata["fp"] == 1
        assert p.metadata["fn"] == 1
        assert p.value == 0.5
        assert r.value == 0.5
        assert f1.value == 0.5

    def test_cross_video_isolation(self) -> None:
        """Scenario 6: Predictions belonging to another video -> UNAVAILABLE."""
        preds = load_predictions_dataset(self.fixtures_dir / "cross_video_predictions.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        assert p.status == MetricStatus.UNAVAILABLE
        assert "not found in predictions dataset" in (p.reason_if_unavailable or "")

    def test_missing_prediction_file(self) -> None:
        """Scenario 7: No prediction file supplied -> UNAVAILABLE with explicit reason."""
        harness = EvaluationHarness(dataset=self.dataset, predictions=None)
        report = harness.run()

        assert report.provider_info["predictions_supplied"] is False
        assert len(report.aggregate_metrics) == 0

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        assert p.status == MetricStatus.UNAVAILABLE
        assert "No prediction file supplied" in (p.reason_if_unavailable or "")

    def test_explicitly_empty_prediction_set(self) -> None:
        """Scenario 8: Explicitly empty predictions -> AVAILABLE with zero performance."""
        preds = load_predictions_dataset(self.fixtures_dir / "empty_predictions.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        r = v.metrics["detection_recall"]
        f1 = v.metrics["detection_f1"]

        assert p.status == MetricStatus.AVAILABLE
        assert p.value == 0.0
        assert p.metadata["zero_denominator"] is True
        assert "Zero positive predictions made" in p.metadata["undefined_reason"]

        assert r.status == MetricStatus.AVAILABLE
        assert r.value == 0.0
        assert r.metadata["tp"] == 0
        assert r.metadata["fn"] == 2

        assert f1.status == MetricStatus.AVAILABLE
        assert f1.value == 0.0

    def test_missing_modality_predictions(self) -> None:
        """Scenario 9: Missing modality-specific predictions (field is None) -> UNAVAILABLE."""
        preds = load_predictions_dataset(self.fixtures_dir / "missing_modality_predictions.json")
        harness = EvaluationHarness(dataset=self.dataset, predictions=preds)
        report = harness.run()

        v = report.per_video_results[0]
        p = v.metrics["detection_precision"]
        assert p.status == MetricStatus.UNAVAILABLE
        assert "Modality 'detections' not supplied in predictions file" in (
            p.reason_if_unavailable or ""
        )

        ev = v.metrics["event_f1"]
        assert ev.status == MetricStatus.UNAVAILABLE
        assert "Modality 'events' not supplied in predictions file" in (
            ev.reason_if_unavailable or ""
        )
