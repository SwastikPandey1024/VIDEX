"""Standardized benchmark runner for VIDEX (Priority 3).

Executes reproducible benchmarks with full system metadata, isolated warmup,
separated stage latencies, and descriptive statistics (mean, median, p95, stdev, throughput).
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videx.evaluation.benchmarking import (
    BenchmarkReport,
    BenchmarkRunMetadata,
    calculate_latency_statistics,
    collect_system_environment,
    sha256_file,
)


def run_benchmark(
    video_path: Path,
    model_path: Path,
    backend: str = "pytorch",
    scope: str = "both",
    max_frames: int = 50,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    output_dir: Path = Path("outputs/benchmarks"),
) -> BenchmarkReport:
    """Executes a standardized benchmark run with separated, consistent execution paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_id = f"bench_{backend}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    # Verify input paths
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clip_hash = sha256_file(video_path)
    model_hash = sha256_file(model_path)
    sys_env = collect_system_environment()

    # 1. Cold Initialization Timing
    t_cold0 = time.perf_counter()
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(model_path), task="detect")
    cold_init_ms = (time.perf_counter() - t_cold0) * 1000.0

    # Read video frames up to max_frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    actual_frames = len(frames)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    warmup_times: list[float] = []

    detection_latencies: list[float] = []
    tracking_latencies: list[float] = []
    errors = 0

    # 2. Standalone Detector Inference Pass (if requested)
    if scope in ("detector", "both"):
        # Isolated warmup pass
        t_w0 = time.perf_counter()
        model.predict(
            dummy,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=[0],
            verbose=False,
        )
        warmup_times.append((time.perf_counter() - t_w0) * 1000.0)

        for frame in frames:
            try:
                t0 = time.perf_counter()
                res_det = model.predict(
                    frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    classes=[0],
                    verbose=False,
                )
                _ = res_det[0] if isinstance(res_det, list) else next(iter(res_det))
                detection_latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                errors += 1

    # 3. Integrated Tracking Pipeline Pass (if requested)
    # Note: Ultralytics model.track executes single-pass detection + ByteTrack Kalman association
    if scope in ("tracker", "both"):
        # Reset and warm up tracker
        t_w1 = time.perf_counter()
        model.track(
            dummy,
            tracker="bytetrack.yaml",
            conf=conf_thresh,
            iou=iou_thresh,
            classes=[0],
            verbose=False,
            persist=False,
        )
        warmup_times.append((time.perf_counter() - t_w1) * 1000.0)

        for frame in frames:
            try:
                t0 = time.perf_counter()
                res_trk = model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    conf=conf_thresh,
                    iou=iou_thresh,
                    classes=[0],
                    verbose=False,
                    persist=True,
                )
                _ = res_trk[0] if isinstance(res_trk, list) else next(iter(res_trk))
                tracking_latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                errors += 1

    # 4. Compute Statistics & Compile Report
    stage_statistics: dict[str, Any] = {}
    stages_measured: list[str] = []
    notes: list[str] = [
        f"Evaluated on {actual_frames} consecutive frames of {video_path.name}",
        f"Total video length: {total_video_frames} frames",
        "Single-pass execution: eliminated duplicate detector inference within measured frames.",
    ]

    if detection_latencies:
        det_stats = calculate_latency_statistics(detection_latencies)
        stage_statistics["detection"] = det_stats
        stages_measured.append("detection")
        notes.append("Stage 'detection': Isolated YOLO detection inference.")

    if tracking_latencies:
        trk_stats = calculate_latency_statistics(tracking_latencies)
        stage_statistics["tracking_pipeline"] = trk_stats
        stages_measured.append("tracking_pipeline")
        notes.append(
            "Stage 'tracking_pipeline': Integrated single-pass perception (YOLO detection + ByteTrack association)."
        )

    # End-to-end represents the complete perception pipeline (or detector if detector-only)
    if tracking_latencies:
        e2e_stats = stage_statistics["tracking_pipeline"]
    elif detection_latencies:
        e2e_stats = stage_statistics["detection"]
    else:
        raise RuntimeError("No measurements recorded")

    total_warmup_ms = sum(warmup_times)

    meta = BenchmarkRunMetadata(
        benchmark_id=benchmark_id,
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        backend=backend,
        model_path=str(model_path),
        model_hash=model_hash,
        clip_path=str(video_path),
        clip_hash=clip_hash,
        resolution=(width, height),
        fps=round(fps, 2),
        frame_count=actual_frames,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        stages_measured=stages_measured,
        cold_init_ms=round(cold_init_ms, 2),
        warmup_ms=round(total_warmup_ms, 2),
        num_measured_runs=actual_frames,
        system_metadata=sys_env,
    )

    report = BenchmarkReport(
        metadata=meta,
        stage_statistics=stage_statistics,
        end_to_end_statistics=e2e_stats,
        error_count=errors,
        skipped_frames=0,
        notes=notes,
    )

    json_path = output_dir / f"{benchmark_id}.json"
    md_path = output_dir / f"{benchmark_id}.md"
    report.save_json(json_path)
    report.save_markdown(md_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standardized VIDEX benchmark.")
    parser.add_argument(
        "--video", type=Path, default=Path("Sample_Videos/ppe-1.mp4"), help="Video clip path"
    )
    parser.add_argument("--model", type=Path, default=None, help="Model weights path")
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "onnx"],
        help="Inference backend",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="both",
        choices=["detector", "tracker", "both"],
        help="Benchmark scope",
    )
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to benchmark")
    parser.add_argument(
        "--outdir", type=Path, default=Path("outputs/benchmarks"), help="Output directory"
    )

    args = parser.parse_args()

    # Resolve default model weights based on selected backend
    if args.model is not None:
        model_path = args.model
    elif args.backend == "onnx":
        model_path = Path("models/yolo26n.onnx")
    else:
        model_path = Path("models/yolo26n.pt")

    report = run_benchmark(
        video_path=args.video,
        model_path=model_path,
        backend=args.backend,
        scope=args.scope,
        max_frames=args.frames,
        output_dir=args.outdir,
    )
    print(f"Benchmark completed successfully: {report.metadata.benchmark_id}")
    print(f"Backend: {report.metadata.backend} | Scope: {args.scope}")
    print(
        f"Mean Latency: {report.end_to_end_statistics.mean_ms:.2f} ms | Throughput: {report.end_to_end_statistics.throughput_fps:.1f} FPS"
    )


if __name__ == "__main__":
    main()
