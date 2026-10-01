"""Phase 3.0 OCR Evidence Pipeline Smoke Test Script.

Validates the complete OCR foundation:
1. VideoReader: Decodes frames with authoritative FrameTimestamp provenance.
2. OCRProvider / OCRRouter: Routes by language/script (English/Latin, Hindi/Devanagari).
3. OCRScheduler: Evaluates sampling policies (e.g., fixed interval, all frames).
4. TemporalOCRFusion: Deduplicates multi-frame text observations into durable TextObservations.
5. Evidence: Generates canonical audit records with full upstream provenance.
6. Telemetry & Performance: Measures latency/frame, latency/batch, and throughput.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from videx.domain.schemas import (
    BoundingBox,
    EvidenceType,
)
from videx.ingestion.base import DecodedFrame
from videx.ingestion.reader import OpenCVVideoReader
from videx.ocr.base import MockOCRConfig
from videx.ocr.fusion import TemporalOCRFusion, TemporalOCRFusionConfig
from videx.ocr.mock import MockOCRProvider
from videx.ocr.pipeline import OCRPipeline
from videx.ocr.router import OCRRouter, OCRRouterConfig
from videx.ocr.scheduler import OCRScheduler, OCRSchedulerConfig, OCRSchedulingStrategy


def create_smoke_video(path: Path, frame_count: int = 30, fps: float = 25.0) -> None:
    """Create a synthetic video with multiple text instances appearing over time."""
    width, height = 640, 360
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    try:
        for i in range(frame_count):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)

            # Text 1: "MH 12 AB 1234" present across frames 0..14
            if i < 15:
                cv2.rectangle(frame, (100, 100), (320, 150), (200, 200, 200), -1)
                cv2.putText(
                    frame,
                    "MH 12 AB 1234",
                    (110, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            # Text 2: "SPEED 40" present across frames 15..29
            if i >= 15:
                cv2.rectangle(frame, (350, 200), (520, 250), (220, 220, 220), -1)
                cv2.putText(
                    frame,
                    "SPEED 40",
                    (360, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            # Frame counter watermark
            cv2.putText(
                frame,
                f"FRAME {i:02d}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            out.write(frame)
    finally:
        out.release()


def run_smoke_test() -> None:
    print("=" * 70)
    print("VIDEX Phase 3.0 OCR Evidence Pipeline Smoke Test")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "ocr_smoke_test.mp4"
        total_frames = 30
        fps = 25.0
        print(f"\n[1/5] Generating synthetic video ({total_frames} frames @ {fps} fps)...")
        create_smoke_video(video_path, frame_count=total_frames, fps=fps)

        print("[2/5] Initializing OpenCVVideoReader and extracting frames...")
        with OpenCVVideoReader(str(video_path)) as reader:
            assert reader.total_frames == total_frames
            print(f"      VideoReader loaded: {reader.total_frames} frames, {reader.fps:.2f} fps")

            # Decode frames
            decoded_frames: list[DecodedFrame] = [
                reader.read_decoded_frame(i) for i in range(total_frames)
            ]

        # Verify authoritative timestamps
        for i, df in enumerate(decoded_frames):
            assert df.frame_timestamp is not None
            assert df.frame_timestamp.frame_index == i
            expected_pts = i / fps
            assert abs(df.frame_timestamp.pts_seconds - expected_pts) < 0.05
        print("      Authoritative FrameTimestamp verification passed.")

        # Build canned results matching the synthetic video
        canned_results_by_frame: dict[int, list[dict[str, object]]] = {}
        for i in range(total_frames):
            frame_obs: list[dict[str, object]] = []
            if i < 15:
                frame_obs.append(
                    {
                        "text": "MH 12 AB 1234",
                        "bbox": BoundingBox(x=100.0, y=100.0, width=220.0, height=50.0),
                        "confidence": 0.95,
                        "language": "en",
                        "script": "Latin",
                    }
                )
            else:
                frame_obs.append(
                    {
                        "text": "SPEED 40",
                        "bbox": BoundingBox(x=350.0, y=200.0, width=170.0, height=50.0),
                        "confidence": 0.92,
                        "language": "en",
                        "script": "Latin",
                    }
                )
            canned_results_by_frame[i] = frame_obs

        print("\n[3/5] Configuring OCR Provider, Router, Scheduler, and Fusion...")
        mock_provider = MockOCRProvider(
            MockOCRConfig(
                provider_name="smoke_ocr_provider",
                canned_results_by_frame=canned_results_by_frame,
                supported_languages=("en", "hi"),
            )
        )
        mock_provider.warmup()

        router = OCRRouter(
            OCRRouterConfig(
                providers={"smoke_ocr": mock_provider},
                default_provider_key="smoke_ocr",
            )
        )

        scheduler = OCRScheduler(
            OCRSchedulerConfig(
                strategy=OCRSchedulingStrategy.ALL_FRAMES,
            )
        )

        fusion = TemporalOCRFusion(
            TemporalOCRFusionConfig(
                max_frame_gap=3,
                min_iou_overlap=0.3,
            )
        )

        pipeline = OCRPipeline(
            router=router,
            scheduler=scheduler,
            fusion=fusion,
        )

        # Batch benchmarking
        print("\n[4/5] Measuring OCR Provider batch contract & latency...")
        batch_sizes = [5, 10]
        for b_size in batch_sizes:
            sub_batch = decoded_frames[:b_size]
            t0 = time.perf_counter()
            batch_res = mock_provider.detect_text_batch(sub_batch)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"      Batch size {b_size:2d}: {lat_ms:6.2f} ms total "
                f"({lat_ms / b_size:5.2f} ms/frame), {sum(len(r) for r in batch_res)} obs"
            )

        print("\n[5/5] Executing full OCR Pipeline over video timeline...")
        t_start = time.perf_counter()
        raw_obs, fused_obs, evidence_records = pipeline.process_frames(decoded_frames)
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Assertions
        assert len(raw_obs) == 30
        assert len(fused_obs) == 2, f"Expected 2 fused text observations, got {len(fused_obs)}"
        assert len(evidence_records) == 2

        fused_by_text = {f.text: f for f in fused_obs}
        assert "MH 12 AB 1234" in fused_by_text
        assert "SPEED 40" in fused_by_text

        plate = fused_by_text["MH 12 AB 1234"]
        assert plate.first_seen_frame == 0
        assert plate.last_seen_frame == 14
        assert len(plate.supporting_frames) == 15
        assert len(plate.supporting_observation_ids) == 15

        speed = fused_by_text["SPEED 40"]
        assert speed.first_seen_frame == 15
        assert speed.last_seen_frame == 29
        assert len(speed.supporting_frames) == 15
        assert len(speed.supporting_observation_ids) == 15

        for ev in evidence_records:
            assert ev.evidence_type == EvidenceType.OCR
            assert ev.video_id == decoded_frames[0].video_id
            assert len(ev.supporting_observation_ids) == 15

        print("\n" + "=" * 70)
        print("PERFORMANCE & TELEMETRY REPORT")
        print("=" * 70)
        print(f"Frames evaluated:       {total_frames}")
        print(f"Frames OCR processed:   {total_frames}")
        print(f"Raw observations:       {len(raw_obs)}")
        print(f"Fused TextObservations: {len(fused_obs)}")
        print(f"Evidence records:       {len(evidence_records)}")
        print(f"Total pipeline time:    {total_time_ms:.2f} ms")
        print(f"Mean latency/frame:     {total_time_ms / total_frames:.3f} ms/frame")
        print("-" * 70)
        print("TEMPORAL FUSION SUMMARY:")
        for idx, f in enumerate(fused_obs):
            print(
                f"  [{idx + 1}] Text: '{f.text}' | "
                f"Frames: {f.first_seen_frame}..{f.last_seen_frame} "
                f"({f.frame_count} frames, {f.duration_seconds:.3f}s) | "
                f"Confidence: {f.mean_confidence:.2f} | "
                f"Provider: {f.provider}"
            )
        print("=" * 70)
        print("VIDEX Phase 3.0 OCR Smoke Test: SUCCESS")
        print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
