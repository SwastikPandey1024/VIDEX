"""Unified, reproducible evaluation harness runner for VIDEX.

Loads gold-standard annotations, ingests typed predictions, enforces explicit
missing/incomplete semantics, computes mathematically defined micro and macro
metrics, and outputs machine-readable EvaluationResult JSON along with
formatted CLI summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from videx.domain.schemas import BoundingBox
from videx.evaluation.metrics import (
    compute_cer,
    compute_wer,
    evaluate_detections,
    evaluate_qa_abstention,
    evaluate_temporal_events,
    wilson_score_interval,
)
from videx.evaluation.schemas import (
    AnnotationDataset,
    AnnotationStatus,
    DetectionPrediction,
    EvaluationResult,
    EventPrediction,
    MetricResult,
    MetricStatus,
    PredictionDataset,
    QAPrediction,
    VideoAnnotation,
    VideoEvaluationSummary,
    VideoPredictions,
)

logger = logging.getLogger("videx.evaluation")


def get_git_commit() -> str | None:
    """Safely retrieves the current git commit hash if in a git repository."""
    git_bin = shutil.which("git")
    if not git_bin:
        return None
    try:
        res = subprocess.run(  # noqa: S603
            [git_bin, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except OSError as err:
        logger.debug("Failed to retrieve git commit: %s", err)
    return None


def load_annotation_dataset(path: Path) -> AnnotationDataset:
    """Loads and strictly validates an AnnotationDataset from a JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return AnnotationDataset.model_validate(data)


def load_predictions_dataset(path: Path) -> PredictionDataset:
    """Loads and strictly validates a PredictionDataset from a JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "videos" in data and isinstance(data["videos"], list):
        return PredictionDataset.model_validate(data)
    elif isinstance(data, dict):
        # Raw dictionary format {video_id: {modality: [...]}}
        videos_list: list[VideoPredictions] = []
        for vid, v_data in data.items():
            if isinstance(v_data, dict):
                # Normalize detections if raw dicts
                norm_dets: list[DetectionPrediction] | None = None
                if "detections" in v_data and v_data["detections"] is not None:
                    norm_dets = []
                    for d in v_data["detections"]:
                        if isinstance(d, DetectionPrediction):
                            norm_dets.append(d)
                        elif isinstance(d, dict):
                            b = d["bbox"]
                            bbox_obj = b if isinstance(b, BoundingBox) else BoundingBox(**b)
                            norm_dets.append(
                                DetectionPrediction(
                                    timestamp_pts=float(d["timestamp_pts"]),
                                    class_label=str(d["class_label"]),
                                    bbox=bbox_obj,
                                    confidence=float(d.get("confidence", 1.0)),
                                )
                            )

                norm_events: list[EventPrediction] | None = None
                if "events" in v_data and v_data["events"] is not None:
                    norm_events = [
                        ev if isinstance(ev, EventPrediction) else EventPrediction(**ev)
                        for ev in v_data["events"]
                    ]

                norm_qa: list[QAPrediction] | None = None
                if "qa" in v_data and v_data["qa"] is not None:
                    norm_qa = [
                        q if isinstance(q, QAPrediction) else QAPrediction(**q)
                        for q in v_data["qa"]
                    ]

                videos_list.append(
                    VideoPredictions(
                        video_id=str(vid),
                        detections=norm_dets,
                        events=norm_events,
                        qa=norm_qa,
                        ocr=v_data.get("ocr"),
                        transcripts=v_data.get("transcripts"),
                    )
                )
        return PredictionDataset(dataset_id="raw_predictions_dataset", videos=videos_list)
    else:
        raise ValueError("Invalid prediction dataset JSON format")


class EvaluationHarness:
    """Executes evaluation across an AnnotationDataset using provided predictions."""

    def __init__(
        self,
        dataset: AnnotationDataset,
        predictions: PredictionDataset | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        predictions_by_video: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config or {
            "iou_threshold": 0.5,
            "tiou_threshold": 0.5,
            "temporal_tolerance_seconds": 0.1,
            "case_sensitive_ocr": False,
            "matching_strategy": "optimal",
        }

        preds_input = predictions if predictions is not None else predictions_by_video

        if preds_input is None:
            self.predictions_supplied = False
            self.predictions_dataset: PredictionDataset | None = None
        elif isinstance(preds_input, PredictionDataset):
            self.predictions_supplied = True
            self.predictions_dataset = preds_input
        elif isinstance(preds_input, dict):
            self.predictions_supplied = True
            # Wrap dictionary into PredictionDataset
            videos_list: list[VideoPredictions] = []
            for vid, v_data in preds_input.items():
                if isinstance(v_data, dict):
                    norm_dets = None
                    if "detections" in v_data and v_data["detections"] is not None:
                        norm_dets = []
                        for d in v_data["detections"]:
                            if isinstance(d, DetectionPrediction):
                                norm_dets.append(d)
                            elif isinstance(d, dict):
                                b = d["bbox"]
                                bbox_obj = b if isinstance(b, BoundingBox) else BoundingBox(**b)
                                norm_dets.append(
                                    DetectionPrediction(
                                        timestamp_pts=float(d["timestamp_pts"]),
                                        class_label=str(d["class_label"]),
                                        bbox=bbox_obj,
                                        confidence=float(d.get("confidence", 1.0)),
                                    )
                                )

                    norm_events = None
                    if "events" in v_data and v_data["events"] is not None:
                        norm_events = [
                            ev if isinstance(ev, EventPrediction) else EventPrediction(**ev)
                            for ev in v_data["events"]
                        ]

                    norm_qa = None
                    if "qa" in v_data and v_data["qa"] is not None:
                        norm_qa = [
                            q if isinstance(q, QAPrediction) else QAPrediction(**q)
                            for q in v_data["qa"]
                        ]

                    videos_list.append(
                        VideoPredictions(
                            video_id=str(vid),
                            detections=norm_dets,
                            events=norm_events,
                            qa=norm_qa,
                            ocr=v_data.get("ocr"),
                            transcripts=v_data.get("transcripts"),
                        )
                    )
            self.predictions_dataset = PredictionDataset(
                dataset_id="dict_predictions",
                videos=videos_list,
            )
        else:
            raise TypeError(f"Unsupported predictions type: {type(predictions)}")

    def evaluate_video(self, video_ann: VideoAnnotation) -> VideoEvaluationSummary:
        """Evaluates all annotated modalities for a single video."""
        vid = str(video_ann.video_id)
        video_metrics: dict[str, MetricResult] = {}
        unavailable_modalities: list[str] = []
        errors: list[str] = []

        # Find predictions for this video
        vp: VideoPredictions | None = None
        if self.predictions_supplied and self.predictions_dataset is not None:
            vp = self.predictions_dataset.get_video_predictions(vid)

        # 1. Detection Evaluation
        if video_ann.detections.status == AnnotationStatus.ANNOTATED:
            if not self.predictions_supplied:
                unavailable_modalities.append("detections")
                video_metrics["detection_precision"] = MetricResult(
                    name="detection_precision",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
                video_metrics["detection_recall"] = MetricResult(
                    name="detection_recall",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
                video_metrics["detection_f1"] = MetricResult(
                    name="detection_f1",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
            elif vp is None:
                unavailable_modalities.append("detections")
                video_metrics["detection_precision"] = MetricResult(
                    name="detection_precision",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
                video_metrics["detection_recall"] = MetricResult(
                    name="detection_recall",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
                video_metrics["detection_f1"] = MetricResult(
                    name="detection_f1",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
            elif vp.detections is None:
                unavailable_modalities.append("detections")
                video_metrics["detection_precision"] = MetricResult(
                    name="detection_precision",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'detections' not supplied in predictions file",
                )
                video_metrics["detection_recall"] = MetricResult(
                    name="detection_recall",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'detections' not supplied in predictions file",
                )
                video_metrics["detection_f1"] = MetricResult(
                    name="detection_f1",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'detections' not supplied in predictions file",
                )
            else:
                det_metrics = evaluate_detections(
                    ground_truth=video_ann.detections.items,
                    predictions=vp.detections,
                    iou_threshold=self.config.get("iou_threshold", 0.5),
                    temporal_tolerance_seconds=self.config.get("temporal_tolerance_seconds", 0.1),
                    matching_strategy=self.config.get("matching_strategy", "optimal"),
                )
                video_metrics.update(det_metrics)
        elif video_ann.detections.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["detection_precision"] = MetricResult(
                name="detection_precision",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=(
                    video_ann.detections.rationale or "Modality marked not applicable"
                ),
            )
        else:
            unavailable_modalities.append("detections")
            video_metrics["detection_precision"] = MetricResult(
                name="detection_precision",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=(
                    f"Detections GT status is {video_ann.detections.status.value}"
                ),
            )

        # 2. OCR Evaluation
        if video_ann.ocr.status == AnnotationStatus.ANNOTATED:
            gt_ocr = video_ann.ocr.items
            if not self.predictions_supplied:
                unavailable_modalities.append("ocr")
                video_metrics["ocr_cer"] = MetricResult(
                    name="ocr_cer",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
            elif vp is None:
                unavailable_modalities.append("ocr")
                video_metrics["ocr_cer"] = MetricResult(
                    name="ocr_cer",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
            elif vp.ocr is None:
                unavailable_modalities.append("ocr")
                video_metrics["ocr_cer"] = MetricResult(
                    name="ocr_cer",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'ocr' not supplied in predictions file",
                )
            else:
                if not gt_ocr:
                    video_metrics["ocr_cer"] = MetricResult(
                        name="ocr_cer",
                        status=MetricStatus.AVAILABLE,
                        value=0.0 if not vp.ocr else 1.0,
                        sample_size=0,
                        metadata={"note": "Verified zero on-screen text in ground truth"},
                    )
                else:
                    ref_text = " ".join(o.expected_text for o in gt_ocr)
                    hyp_text = " ".join(vp.ocr)
                    cer_val, cer_meta = compute_cer(
                        ref_text,
                        hyp_text,
                        case_sensitive=self.config.get("case_sensitive_ocr", False),
                    )
                    video_metrics["ocr_cer"] = MetricResult(
                        name="ocr_cer",
                        status=MetricStatus.AVAILABLE,
                        value=cer_val,
                        sample_size=cer_meta["reference_chars"],
                        metadata=cer_meta,
                    )
        elif video_ann.ocr.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["ocr_cer"] = MetricResult(
                name="ocr_cer",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=video_ann.ocr.rationale or "OCR marked not applicable",
            )
        else:
            unavailable_modalities.append("ocr")
            video_metrics["ocr_cer"] = MetricResult(
                name="ocr_cer",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"OCR ground truth status is {video_ann.ocr.status.value}",
            )

        # 3. Audio / Transcript Evaluation
        if video_ann.transcripts.status == AnnotationStatus.ANNOTATED:
            gt_tr = video_ann.transcripts.items
            if not self.predictions_supplied:
                unavailable_modalities.append("transcripts")
                video_metrics["asr_wer"] = MetricResult(
                    name="asr_wer",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
            elif vp is None:
                unavailable_modalities.append("transcripts")
                video_metrics["asr_wer"] = MetricResult(
                    name="asr_wer",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
            elif vp.transcripts is None:
                unavailable_modalities.append("transcripts")
                video_metrics["asr_wer"] = MetricResult(
                    name="asr_wer",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'transcripts' not supplied in predictions file",
                )
            else:
                if not gt_tr:
                    video_metrics["asr_wer"] = MetricResult(
                        name="asr_wer",
                        status=MetricStatus.AVAILABLE,
                        value=0.0 if not vp.transcripts else 1.0,
                        sample_size=0,
                        metadata={"note": "Verified zero speech dialogue in ground truth"},
                    )
                else:
                    ref_text = " ".join(t.reference_text for t in gt_tr)
                    hyp_text = " ".join(vp.transcripts)
                    wer_val, wer_meta = compute_wer(ref_text, hyp_text)
                    video_metrics["asr_wer"] = MetricResult(
                        name="asr_wer",
                        status=MetricStatus.AVAILABLE,
                        value=wer_val,
                        sample_size=wer_meta["reference_words"],
                        metadata=wer_meta,
                    )
        elif video_ann.transcripts.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["asr_wer"] = MetricResult(
                name="asr_wer",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=(
                    video_ann.transcripts.rationale or "Audio marked not applicable"
                ),
            )
        else:
            unavailable_modalities.append("transcripts")
            video_metrics["asr_wer"] = MetricResult(
                name="asr_wer",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=(
                    f"Transcripts GT status is {video_ann.transcripts.status.value}"
                ),
            )

        # 4. Temporal Events Evaluation
        if video_ann.events.status == AnnotationStatus.ANNOTATED:
            gt_events = video_ann.events.items
            if not self.predictions_supplied:
                unavailable_modalities.append("events")
                video_metrics["event_f1"] = MetricResult(
                    name="event_f1",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
            elif vp is None:
                unavailable_modalities.append("events")
                video_metrics["event_f1"] = MetricResult(
                    name="event_f1",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
            elif vp.events is None:
                unavailable_modalities.append("events")
                video_metrics["event_f1"] = MetricResult(
                    name="event_f1",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'events' not supplied in predictions file",
                )
            else:
                ev_metrics = evaluate_temporal_events(
                    ground_truth=gt_events,
                    predictions=vp.events,
                    tiou_threshold=self.config.get("tiou_threshold", 0.5),
                    matching_strategy=self.config.get("matching_strategy", "optimal"),
                )
                video_metrics.update(ev_metrics)
        elif video_ann.events.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["event_f1"] = MetricResult(
                name="event_f1",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=video_ann.events.rationale or "Events marked not applicable",
            )
        else:
            unavailable_modalities.append("events")
            video_metrics["event_f1"] = MetricResult(
                name="event_f1",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=(
                    f"Events GT status is {video_ann.events.status.value}"
                ),
            )

        # 5. QA Abstention Evaluation
        if video_ann.qa.status == AnnotationStatus.ANNOTATED:
            gt_qa = video_ann.qa.items
            if not self.predictions_supplied:
                unavailable_modalities.append("qa")
                video_metrics["qa_abstention_accuracy"] = MetricResult(
                    name="qa_abstention_accuracy",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="No prediction file supplied; cannot evaluate accuracy",
                )
            elif vp is None:
                unavailable_modalities.append("qa")
                video_metrics["qa_abstention_accuracy"] = MetricResult(
                    name="qa_abstention_accuracy",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable=f"Video '{vid}' not found in predictions dataset",
                )
            elif vp.qa is None:
                unavailable_modalities.append("qa")
                video_metrics["qa_abstention_accuracy"] = MetricResult(
                    name="qa_abstention_accuracy",
                    status=MetricStatus.UNAVAILABLE,
                    reason_if_unavailable="Modality 'qa' not supplied in predictions file",
                )
            else:
                qa_metrics = evaluate_qa_abstention(ground_truth=gt_qa, predictions=vp.qa)
                video_metrics.update(qa_metrics)
        elif video_ann.qa.status == AnnotationStatus.NOT_APPLICABLE:
            video_metrics["qa_abstention_accuracy"] = MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.NOT_APPLICABLE,
                reason_if_unavailable=video_ann.qa.rationale or "QA marked not applicable",
            )
        else:
            unavailable_modalities.append("qa")
            video_metrics["qa_abstention_accuracy"] = MetricResult(
                name="qa_abstention_accuracy",
                status=MetricStatus.UNAVAILABLE,
                reason_if_unavailable=f"QA ground truth status is {video_ann.qa.status.value}",
            )

        return VideoEvaluationSummary(
            video_id=vid,
            source_identifier=video_ann.source_identifier,
            metrics=video_metrics,
            unavailable_modalities=unavailable_modalities,
            errors=errors,
        )

    def run(self) -> EvaluationResult:
        """Executes full evaluation over all videos in the dataset."""
        per_video_summaries: list[VideoEvaluationSummary] = []
        failures: list[str] = []

        for video_ann in self.dataset.videos:
            try:
                summary = self.evaluate_video(video_ann)
                per_video_summaries.append(summary)
            except Exception as ex:
                logger.exception("Error evaluating video %s", video_ann.video_id)
                failures.append(f"Video {video_ann.video_id}: {str(ex)}")

        aggregate_metrics = self._aggregate_metrics(per_video_summaries)

        return EvaluationResult(
            evaluation_id=uuid4(),
            timestamp=datetime.now(UTC),
            repo_commit=get_git_commit(),
            dataset_version=self.dataset.schema_version,
            configuration=self.config,
            provider_info={
                "runner_version": "1.1.0",
                "execution_mode": "prediction_ingestion"
                if self.predictions_supplied
                else "annotation_audit",
                "predictions_supplied": self.predictions_supplied,
            },
            aggregate_metrics=aggregate_metrics,
            per_video_results=per_video_summaries,
            failures=failures,
        )

    def _aggregate_metrics(
        self,
        summaries: list[VideoEvaluationSummary],
    ) -> dict[str, MetricResult]:
        """Mathematically aggregates TP/FP/FN counts across videos for micro averages,
        and computes explicit macro averages across videos.
        """
        aggregates: dict[str, MetricResult] = {}

        # ── 1. Detection Micro & Macro Aggregation ─────────────────────────
        det_res = [
            s.metrics["detection_precision"]
            for s in summaries
            if "detection_precision" in s.metrics
            and s.metrics["detection_precision"].status == MetricStatus.AVAILABLE
        ]
        if det_res:
            total_tp = sum(r.metadata.get("tp", 0) for r in det_res)
            total_fp = sum(r.metadata.get("fp", 0) for r in det_res)
            total_fn = sum(r.metadata.get("fn", 0) for r in det_res)

            # Micro-precision
            micro_p_denom = total_tp + total_fp
            micro_p = (total_tp / micro_p_denom) if micro_p_denom > 0 else 0.0
            p_ci = wilson_score_interval(total_tp, micro_p_denom) if micro_p_denom > 0 else None

            # Micro-recall
            micro_r_denom = total_tp + total_fn
            micro_r = (total_tp / micro_r_denom) if micro_r_denom > 0 else 0.0
            r_ci = wilson_score_interval(total_tp, micro_r_denom) if micro_r_denom > 0 else None

            # Micro-F1
            micro_f1 = (
                (2.0 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
            )

            # Macro averages across videos
            p_vals = [r.value for r in det_res if r.value is not None]
            r_vals = [
                s.metrics["detection_recall"].value
                for s in summaries
                if "detection_recall" in s.metrics
                and s.metrics["detection_recall"].value is not None
            ]
            f1_vals = [
                s.metrics["detection_f1"].value
                for s in summaries
                if "detection_f1" in s.metrics and s.metrics["detection_f1"].value is not None
            ]

            macro_p = sum(p_vals) / len(p_vals) if p_vals else 0.0
            macro_r = sum(r_vals) / len(r_vals) if r_vals else 0.0
            macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0

            det_counts = {
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "videos_evaluated": len(det_res),
            }

            aggregates["detection_micro_precision"] = MetricResult(
                name="detection_micro_precision",
                status=MetricStatus.AVAILABLE,
                value=round(micro_p, 4),
                sample_size=micro_p_denom,
                confidence_interval=p_ci,
                ci_method="wilson_score" if p_ci else None,
                metadata={**det_counts, "aggregation": "micro"},
            )
            aggregates["detection_micro_recall"] = MetricResult(
                name="detection_micro_recall",
                status=MetricStatus.AVAILABLE,
                value=round(micro_r, 4),
                sample_size=micro_r_denom,
                confidence_interval=r_ci,
                ci_method="wilson_score" if r_ci else None,
                metadata={**det_counts, "aggregation": "micro"},
            )
            aggregates["detection_micro_f1"] = MetricResult(
                name="detection_micro_f1",
                status=MetricStatus.AVAILABLE,
                value=round(micro_f1, 4),
                sample_size=micro_r_denom,
                metadata={**det_counts, "aggregation": "micro"},
            )
            aggregates["detection_macro_precision"] = MetricResult(
                name="detection_macro_precision",
                status=MetricStatus.AVAILABLE,
                value=round(macro_p, 4),
                sample_size=len(p_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(p_vals)},
            )
            aggregates["detection_macro_recall"] = MetricResult(
                name="detection_macro_recall",
                status=MetricStatus.AVAILABLE,
                value=round(macro_r, 4),
                sample_size=len(r_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(r_vals)},
            )
            aggregates["detection_macro_f1"] = MetricResult(
                name="detection_macro_f1",
                status=MetricStatus.AVAILABLE,
                value=round(macro_f1, 4),
                sample_size=len(f1_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(f1_vals)},
            )

        # ── 2. Temporal Events Micro & Macro Aggregation ───────────────────
        ev_res = [
            s.metrics["event_precision"]
            for s in summaries
            if "event_precision" in s.metrics
            and s.metrics["event_precision"].status == MetricStatus.AVAILABLE
        ]
        if ev_res:
            ev_tp = sum(r.metadata.get("tp", 0) for r in ev_res)
            ev_fp = sum(r.metadata.get("fp", 0) for r in ev_res)
            ev_fn = sum(r.metadata.get("fn", 0) for r in ev_res)

            ev_p_denom = ev_tp + ev_fp
            ev_micro_p = (ev_tp / ev_p_denom) if ev_p_denom > 0 else 0.0
            ev_p_ci = wilson_score_interval(ev_tp, ev_p_denom) if ev_p_denom > 0 else None

            ev_r_denom = ev_tp + ev_fn
            ev_micro_r = (ev_tp / ev_r_denom) if ev_r_denom > 0 else 0.0
            ev_r_ci = wilson_score_interval(ev_tp, ev_r_denom) if ev_r_denom > 0 else None

            ev_micro_f1 = (
                (2.0 * ev_micro_p * ev_micro_r / (ev_micro_p + ev_micro_r))
                if (ev_micro_p + ev_micro_r) > 0
                else 0.0
            )

            ev_p_vals = [r.value for r in ev_res if r.value is not None]
            ev_r_vals = [
                s.metrics["event_recall"].value
                for s in summaries
                if "event_recall" in s.metrics and s.metrics["event_recall"].value is not None
            ]
            ev_f1_vals = [
                s.metrics["event_f1"].value
                for s in summaries
                if "event_f1" in s.metrics and s.metrics["event_f1"].value is not None
            ]

            ev_macro_p = sum(ev_p_vals) / len(ev_p_vals) if ev_p_vals else 0.0
            ev_macro_r = sum(ev_r_vals) / len(ev_r_vals) if ev_r_vals else 0.0
            ev_macro_f1 = sum(ev_f1_vals) / len(ev_f1_vals) if ev_f1_vals else 0.0

            ev_counts = {
                "total_tp": ev_tp,
                "total_fp": ev_fp,
                "total_fn": ev_fn,
                "videos_evaluated": len(ev_res),
            }

            aggregates["event_micro_precision"] = MetricResult(
                name="event_micro_precision",
                status=MetricStatus.AVAILABLE,
                value=round(ev_micro_p, 4),
                sample_size=ev_p_denom,
                confidence_interval=ev_p_ci,
                ci_method="wilson_score" if ev_p_ci else None,
                metadata={**ev_counts, "aggregation": "micro"},
            )
            aggregates["event_micro_recall"] = MetricResult(
                name="event_micro_recall",
                status=MetricStatus.AVAILABLE,
                value=round(ev_micro_r, 4),
                sample_size=ev_r_denom,
                confidence_interval=ev_r_ci,
                ci_method="wilson_score" if ev_r_ci else None,
                metadata={**ev_counts, "aggregation": "micro"},
            )
            aggregates["event_micro_f1"] = MetricResult(
                name="event_micro_f1",
                status=MetricStatus.AVAILABLE,
                value=round(ev_micro_f1, 4),
                sample_size=ev_r_denom,
                metadata={**ev_counts, "aggregation": "micro"},
            )
            aggregates["event_macro_precision"] = MetricResult(
                name="event_macro_precision",
                status=MetricStatus.AVAILABLE,
                value=round(ev_macro_p, 4),
                sample_size=len(ev_p_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(ev_p_vals)},
            )
            aggregates["event_macro_recall"] = MetricResult(
                name="event_macro_recall",
                status=MetricStatus.AVAILABLE,
                value=round(ev_macro_r, 4),
                sample_size=len(ev_r_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(ev_r_vals)},
            )
            aggregates["event_macro_f1"] = MetricResult(
                name="event_macro_f1",
                status=MetricStatus.AVAILABLE,
                value=round(ev_macro_f1, 4),
                sample_size=len(ev_f1_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(ev_f1_vals)},
            )

        # ── 3. OCR Micro & Macro Aggregation ───────────────────────────────
        ocr_res = [
            s.metrics["ocr_cer"]
            for s in summaries
            if "ocr_cer" in s.metrics and s.metrics["ocr_cer"].status == MetricStatus.AVAILABLE
        ]
        if ocr_res:
            total_edits = sum(
                r.metadata.get("substitutions", 0)
                + r.metadata.get("deletions", 0)
                + r.metadata.get("insertions", 0)
                for r in ocr_res
            )
            total_ref_chars = sum(r.metadata.get("reference_chars", 0) for r in ocr_res)
            micro_cer = (total_edits / total_ref_chars) if total_ref_chars > 0 else 0.0
            ocr_vals = [r.value for r in ocr_res if r.value is not None]
            macro_cer = sum(ocr_vals) / len(ocr_vals) if ocr_vals else 0.0

            aggregates["ocr_micro_cer"] = MetricResult(
                name="ocr_micro_cer",
                status=MetricStatus.AVAILABLE,
                value=round(micro_cer, 4),
                sample_size=total_ref_chars,
                metadata={
                    "total_edits": total_edits,
                    "total_reference_chars": total_ref_chars,
                    "aggregation": "micro",
                },
            )
            aggregates["ocr_macro_cer"] = MetricResult(
                name="ocr_macro_cer",
                status=MetricStatus.AVAILABLE,
                value=round(macro_cer, 4),
                sample_size=len(ocr_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(ocr_vals)},
            )

        # ── 4. ASR Micro & Macro Aggregation ───────────────────────────────
        asr_res = [
            s.metrics["asr_wer"]
            for s in summaries
            if "asr_wer" in s.metrics and s.metrics["asr_wer"].status == MetricStatus.AVAILABLE
        ]
        if asr_res:
            total_wer_edits = sum(
                r.metadata.get("substitutions", 0)
                + r.metadata.get("deletions", 0)
                + r.metadata.get("insertions", 0)
                for r in asr_res
            )
            total_ref_words = sum(r.metadata.get("reference_words", 0) for r in asr_res)
            micro_wer = (total_wer_edits / total_ref_words) if total_ref_words > 0 else 0.0
            asr_vals = [r.value for r in asr_res if r.value is not None]
            macro_wer = sum(asr_vals) / len(asr_vals) if asr_vals else 0.0

            aggregates["asr_micro_wer"] = MetricResult(
                name="asr_micro_wer",
                status=MetricStatus.AVAILABLE,
                value=round(micro_wer, 4),
                sample_size=total_ref_words,
                metadata={
                    "total_edits": total_wer_edits,
                    "total_reference_words": total_ref_words,
                    "aggregation": "micro",
                },
            )
            aggregates["asr_macro_wer"] = MetricResult(
                name="asr_macro_wer",
                status=MetricStatus.AVAILABLE,
                value=round(macro_wer, 4),
                sample_size=len(asr_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(asr_vals)},
            )

        # ── 5. QA Abstention Micro & Macro Aggregation ─────────────────────
        qa_acc_res = [
            s.metrics["qa_abstention_accuracy"]
            for s in summaries
            if "qa_abstention_accuracy" in s.metrics
            and s.metrics["qa_abstention_accuracy"].status == MetricStatus.AVAILABLE
        ]
        if qa_acc_res:
            total_correct = sum(r.metadata.get("correct_decisions", 0) for r in qa_acc_res)
            total_eval = sum(r.sample_size for r in qa_acc_res)
            micro_qa_acc = (total_correct / total_eval) if total_eval > 0 else 0.0
            qa_acc_ci = wilson_score_interval(total_correct, total_eval) if total_eval > 0 else None

            qa_vals = [r.value for r in qa_acc_res if r.value is not None]
            macro_qa_acc = sum(qa_vals) / len(qa_vals) if qa_vals else 0.0

            aggregates["qa_micro_accuracy"] = MetricResult(
                name="qa_micro_accuracy",
                status=MetricStatus.AVAILABLE,
                value=round(micro_qa_acc, 4),
                sample_size=total_eval,
                confidence_interval=qa_acc_ci,
                ci_method="wilson_score" if qa_acc_ci else None,
                metadata={
                    "total_correct": total_correct,
                    "total_evaluated": total_eval,
                    "aggregation": "micro",
                },
            )
            aggregates["qa_macro_accuracy"] = MetricResult(
                name="qa_macro_accuracy",
                status=MetricStatus.AVAILABLE,
                value=round(macro_qa_acc, 4),
                sample_size=len(qa_vals),
                metadata={"aggregation": "macro", "videos_contributing": len(qa_vals)},
            )

        return aggregates


def format_cli_summary(report: EvaluationResult) -> str:
    """Formats an EvaluationResult into a clean, human-readable terminal table."""
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append(f"{'VIDEX ACCURACY & INTEGRITY EVALUATION REPORT':^96}")
    lines.append("=" * 96)
    lines.append(
        f"Evaluation ID: {report.evaluation_id}  |  Commit: {report.repo_commit or 'unknown'}"
    )
    lines.append(
        f"Dataset Version: {report.dataset_version}  |  Executed: {report.timestamp.isoformat()}"
    )

    preds_supplied = report.provider_info.get("predictions_supplied", False)
    if not preds_supplied:
        lines.append("-" * 96)
        lines.append(
            "[!] NOTICE: Evaluation executed WITHOUT model predictions file.\n"
            "    Ground-truth readiness verified; no model accuracy benchmark claimed."
        )

    lines.append("-" * 96)
    lines.append(f"{'Source Video':<18} | {'Metric Name':<30} | {'Status':<12} | {'Value / N'}")
    lines.append("-" * 96)

    for v in report.per_video_results:
        first = True
        for mname, mres in v.metrics.items():
            src_str = v.source_identifier if first else ""
            first = False
            if mres.status == MetricStatus.AVAILABLE and mres.value is not None:
                if mres.metadata.get("zero_denominator"):
                    val_str = f"0.0000 (undefined: 0/{mres.sample_size})"
                else:
                    val_str = f"{mres.value:.4f} (N={mres.sample_size})"
                    if mres.confidence_interval:
                        ci_l, ci_u = mres.confidence_interval
                        val_str += f" [CI: {ci_l:.2f}-{ci_u:.2f}]"
            elif mres.status == MetricStatus.NOT_APPLICABLE:
                val_str = f"N/A ({mres.reason_if_unavailable})"
            else:
                val_str = f"UNAVAILABLE ({mres.reason_if_unavailable})"

            lines.append(f"{src_str:<18} | {mname:<30} | {mres.status.value:<12} | {val_str}")
        lines.append("-" * 96)

    lines.append("CORPUS AGGREGATE METRICS SUMMARY:")
    if not report.aggregate_metrics:
        lines.append(
            "  (No aggregate accuracy metrics computed — no valid model predictions supplied)"
        )
    else:
        for mname, mres in sorted(report.aggregate_metrics.items()):
            ci_str = ""
            if mres.confidence_interval:
                ci_l, ci_u = mres.confidence_interval
                ci_str = f" [95% Wilson CI: {ci_l:.4f} - {ci_u:.4f}]"
            lines.append(f"  - {mname:<32}: {mres.value:.4f} (N={mres.sample_size}){ci_str}")

    if report.failures:
        lines.append("\nFAILURES / ALERTS:")
        for f in report.failures:
            lines.append(f"  [!] {f}")

    lines.append("=" * 96)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running VIDEX evaluation against annotation datasets."""
    parser = argparse.ArgumentParser(description="VIDEX Accuracy-First Evaluation Harness")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to annotation dataset JSON file",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions JSON file (optional)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Destination path for output EvaluationResult JSON",
    )
    parser.add_argument(
        "--matching-strategy",
        type=str,
        default="optimal",
        choices=["optimal", "greedy"],
        help="Bipartite matching algorithm for detections and events (default: optimal)",
    )
    args = parser.parse_args()

    dataset = load_annotation_dataset(args.annotations)
    pred_dataset: PredictionDataset | None = None
    if args.predictions:
        pred_dataset = load_predictions_dataset(args.predictions)

    config = {
        "iou_threshold": 0.5,
        "tiou_threshold": 0.5,
        "temporal_tolerance_seconds": 0.1,
        "case_sensitive_ocr": False,
        "matching_strategy": args.matching_strategy,
    }

    harness = EvaluationHarness(dataset=dataset, predictions=pred_dataset, config=config)
    result = harness.run()

    print(format_cli_summary(result))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"Machine-readable results saved to {args.output_json}")


if __name__ == "__main__":
    main()
