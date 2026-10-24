"""Generate complete per-frame forensic diagnostic of PyTorch vs ONNX Runtime parity.

Investigates every single unmatched detection, computing:
  - Frame-by-frame counts
  - Matched pairs (IoU, conf delta, coord delta)
  - Unmatched detections for both runtimes
  - Nearest candidate from other runtime (even below threshold)
  - Distance from threshold
  - Categorization of root cause (threshold boundary, low-confidence background artifact, etc.)
"""

import json
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO

VIDEO_PATH = Path("Sample_Videos/ppe-1.mp4")
PT_MODEL_PATH = Path("models/yolo26n.pt")
ONNX_MODEL_PATH = Path("models/yolo26n_fp32_bench.onnx")
OUTPUT_JSON = Path("outputs/profiling/onnx_parity_per_frame_diagnostic.json")


def run_forensic_diagnostic(conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> dict[str, Any]:
    pt_model = YOLO(str(PT_MODEL_PATH))
    ort_model = YOLO(str(ONNX_MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    per_frame_records: list[dict[str, Any]] = []
    
    total_pt = 0
    total_ort = 0
    total_matched = 0
    total_unmatched_pt = 0
    total_unmatched_ort = 0
    all_matched_ious: list[float] = []
    all_matched_conf_deltas: list[float] = []
    all_matched_l2_deltas: list[float] = []
    class_mismatches_count = 0

    while cap.isOpened() and frame_idx < 343:
        ret, frame = cap.read()
        if not ret:
            break

        # Run both through identical Ultralytics prediction pipeline
        pt_res = pt_model(frame, device="cpu", conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        ort_res = ort_model(frame, device="cpu", conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

        # Also run at conf=0.15 to capture sub-threshold candidate existence for unmatched diagnosis
        pt_res_sub = pt_model(frame, device="cpu", conf=0.15, iou=iou_thresh, verbose=False)[0]
        ort_res_sub = ort_model(frame, device="cpu", conf=0.15, iou=iou_thresh, verbose=False)[0]

        pt_b = pt_res.boxes.xyxy.cpu().numpy()
        pt_c = pt_res.boxes.conf.cpu().numpy()
        pt_cls = pt_res.boxes.cls.cpu().numpy().astype(int)

        ort_b = ort_res.boxes.xyxy.cpu().numpy()
        ort_c = ort_res.boxes.conf.cpu().numpy()
        ort_cls = ort_res.boxes.cls.cpu().numpy().astype(int)

        sub_pt_b = pt_res_sub.boxes.xyxy.cpu().numpy()
        sub_pt_c = pt_res_sub.boxes.conf.cpu().numpy()
        sub_pt_cls = pt_res_sub.boxes.cls.cpu().numpy().astype(int)

        sub_ort_b = ort_res_sub.boxes.xyxy.cpu().numpy()
        sub_ort_c = ort_res_sub.boxes.conf.cpu().numpy()
        sub_ort_cls = ort_res_sub.boxes.cls.cpu().numpy().astype(int)

        total_pt += len(pt_b)
        total_ort += len(ort_b)

        # Match post-threshold detections greedily 1-to-1
        used_ort: set[int] = set()
        matched_pairs: list[dict[str, Any]] = []
        unmatched_pt: list[dict[str, Any]] = []

        for i, (pb, pc, pcl) in enumerate(zip(pt_b, pt_c, pt_cls)):
            best_iou = 0.0
            best_j = -1
            for j, (ob, oc, ocl) in enumerate(zip(ort_b, ort_c, ort_cls)):
                if j in used_ort:
                    continue
                x1 = max(pb[0], ob[0])
                y1 = max(pb[1], ob[1])
                x2 = min(pb[2], ob[2])
                y2 = min(pb[3], ob[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= 0.50:
                used_ort.add(best_j)
                ob = ort_b[best_j]
                oc = ort_c[best_j]
                ocl = ort_cls[best_j]
                conf_d = abs(float(pc) - float(oc))
                l2_d = float(np.sqrt(np.sum((pb - ob) ** 2)))
                cls_match = bool(pcl == ocl)
                if not cls_match:
                    class_mismatches_count += 1

                all_matched_ious.append(float(best_iou))
                all_matched_conf_deltas.append(conf_d)
                all_matched_l2_deltas.append(l2_d)
                total_matched += 1

                matched_pairs.append({
                    "pt_box": [round(float(v), 2) for v in pb],
                    "onnx_box": [round(float(v), 2) for v in ob],
                    "class_id_pt": int(pcl),
                    "class_name_pt": pt_model.names.get(int(pcl), str(pcl)),
                    "class_id_onnx": int(ocl),
                    "class_name_onnx": ort_model.names.get(int(ocl), str(ocl)),
                    "class_match": cls_match,
                    "pt_conf": round(float(pc), 4),
                    "onnx_conf": round(float(oc), 4),
                    "conf_delta": round(conf_d, 4),
                    "iou": round(float(best_iou), 4),
                    "l2_delta_px": round(l2_d, 3),
                })
            else:
                # Find nearest sub-threshold ONNX candidate to diagnose cause
                nearest_sub_iou = 0.0
                nearest_sub_conf = None
                nearest_sub_cls = None
                for sob, soc, socl in zip(sub_ort_b, sub_ort_c, sub_ort_cls):
                    x1 = max(pb[0], sob[0])
                    y1 = max(pb[1], sob[1])
                    x2 = min(pb[2], sob[2])
                    y2 = min(pb[3], sob[3])
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    union = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (sob[2] - sob[0]) * (sob[3] - sob[1]) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > nearest_sub_iou:
                        nearest_sub_iou = iou
                        nearest_sub_conf = float(soc)
                        nearest_sub_cls = int(socl)

                cause = "unclassified"
                if nearest_sub_conf is not None and nearest_sub_iou >= 0.50:
                    if nearest_sub_conf < conf_thresh:
                        cause = f"sub_threshold_in_onnx (onnx_conf={nearest_sub_conf:.4f} < {conf_thresh})"
                    else:
                        cause = "suppressed_by_nms_in_onnx"
                else:
                    cause = "no_overlapping_onnx_candidate_found"

                total_unmatched_pt += 1
                unmatched_pt.append({
                    "pt_box": [round(float(v), 2) for v in pb],
                    "class_id": int(pcl),
                    "class_name": pt_model.names.get(int(pcl), str(pcl)),
                    "confidence": round(float(pc), 4),
                    "distance_from_threshold": round(float(pc) - conf_thresh, 4),
                    "nearest_onnx_candidate_iou": round(float(nearest_sub_iou), 4),
                    "nearest_onnx_candidate_conf": round(float(nearest_sub_conf), 4) if nearest_sub_conf else None,
                    "nearest_onnx_candidate_class": ort_model.names.get(nearest_sub_cls, str(nearest_sub_cls)) if nearest_sub_cls is not None else None,
                    "cause": cause,
                })

        # Process unmatched ONNX detections
        unmatched_ort: list[dict[str, Any]] = []
        for j, (ob, oc, ocl) in enumerate(zip(ort_b, ort_c, ort_cls)):
            if j in used_ort:
                continue
            # Find nearest sub-threshold PyTorch candidate
            nearest_sub_iou = 0.0
            nearest_sub_conf = None
            nearest_sub_cls = None
            for spb, spc, spcl in zip(sub_pt_b, sub_pt_c, sub_pt_cls):
                x1 = max(ob[0], spb[0])
                y1 = max(ob[1], spb[1])
                x2 = min(ob[2], spb[2])
                y2 = min(ob[3], spb[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = (ob[2] - ob[0]) * (ob[3] - ob[1]) + (spb[2] - spb[0]) * (spb[3] - spb[1]) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > nearest_sub_iou:
                    nearest_sub_iou = iou
                    nearest_sub_conf = float(spc)
                    nearest_sub_cls = int(spcl)

            cause = "unclassified"
            if nearest_sub_conf is not None and nearest_sub_iou >= 0.50:
                if nearest_sub_conf < conf_thresh:
                    cause = f"sub_threshold_in_pytorch (pytorch_conf={nearest_sub_conf:.4f} < {conf_thresh})"
                else:
                    cause = "suppressed_by_nms_in_pytorch"
            else:
                cause = "no_overlapping_pytorch_candidate_found"

            total_unmatched_ort += 1
            unmatched_ort.append({
                "onnx_box": [round(float(v), 2) for v in ob],
                "class_id": int(ocl),
                "class_name": ort_model.names.get(int(ocl), str(ocl)),
                "confidence": round(float(oc), 4),
                "distance_from_threshold": round(float(oc) - conf_thresh, 4),
                "nearest_pt_candidate_iou": round(float(nearest_sub_iou), 4),
                "nearest_pt_candidate_conf": round(float(nearest_sub_conf), 4) if nearest_sub_conf else None,
                "nearest_pt_candidate_class": pt_model.names.get(nearest_sub_cls, str(nearest_sub_cls)) if nearest_sub_cls is not None else None,
                "cause": cause,
            })

        per_frame_records.append({
            "frame_index": frame_idx,
            "pt_detection_count": len(pt_b),
            "onnx_detection_count": len(ort_b),
            "count_delta": len(ort_b) - len(pt_b),
            "matched_pairs_count": len(matched_pairs),
            "unmatched_pt_count": len(unmatched_pt),
            "unmatched_onnx_count": len(unmatched_ort),
            "matched_pairs": matched_pairs,
            "unmatched_pt": unmatched_pt,
            "unmatched_onnx": unmatched_ort,
        })

        frame_idx += 1

    cap.release()

    diagnostic_summary: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "FORENSIC_DIAGNOSTIC — PyTorch vs ONNX Runtime Detection Parity Investigation",
        "video": str(VIDEO_PATH),
        "total_frames_evaluated": len(per_frame_records),
        "confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "counts": {
            "total_pt_detections": total_pt,
            "total_onnx_detections": total_ort,
            "count_delta": total_ort - total_pt,
            "count_delta_percentage": round(abs(total_ort - total_pt) / max(total_pt, 1) * 100, 2),
            "matched_pairs": total_matched,
            "unmatched_pt": total_unmatched_pt,
            "unmatched_onnx": total_unmatched_ort,
            "class_mismatches_in_matched": class_mismatches_count,
            "exact_count_match_frames": sum(1 for r in per_frame_records if r["count_delta"] == 0),
        },
        "matched_metrics": {
            "mean_iou": round(float(np.mean(all_matched_ious)), 4) if all_matched_ious else 0.0,
            "median_iou": round(float(np.median(all_matched_ious)), 4) if all_matched_ious else 0.0,
            "p90_iou": round(float(np.percentile(all_matched_ious, 90)), 4) if all_matched_ious else 0.0,
            "min_iou": round(float(np.min(all_matched_ious)), 4) if all_matched_ious else 0.0,
            "mean_conf_delta": round(float(np.mean(all_matched_conf_deltas)), 4) if all_matched_conf_deltas else 0.0,
            "median_conf_delta": round(float(np.median(all_matched_conf_deltas)), 4) if all_matched_conf_deltas else 0.0,
            "p90_conf_delta": round(float(np.percentile(all_matched_conf_deltas, 90)), 4) if all_matched_conf_deltas else 0.0,
            "max_conf_delta": round(float(np.max(all_matched_conf_deltas)), 4) if all_matched_conf_deltas else 0.0,
            "mean_l2_delta_px": round(float(np.mean(all_matched_l2_deltas)), 3) if all_matched_l2_deltas else 0.0,
            "median_l2_delta_px": round(float(np.median(all_matched_l2_deltas)), 3) if all_matched_l2_deltas else 0.0,
        },
        "frames": per_frame_records,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, indent=2)

    print(f"\n[OK] Forensic diagnostic complete. Saved {len(per_frame_records)} frames to {OUTPUT_JSON}")
    print(f"Summary: PT={total_pt}, ONNX={total_ort}, Delta={total_ort - total_pt} ({diagnostic_summary['counts']['count_delta_percentage']}%)")
    print(f"Matched={total_matched}, Mean IoU={diagnostic_summary['matched_metrics']['mean_iou']}, Median Conf Delta={diagnostic_summary['matched_metrics']['median_conf_delta']}")
    return diagnostic_summary


if __name__ == "__main__":
    run_forensic_diagnostic()
