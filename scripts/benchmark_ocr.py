"""VIDEX Phase 3.0R — OCR Performance & Telemetry Benchmark.

Benchmarks:
1. Real PaddleOCR General Route (PP-OCRv6, Latin/English, device=cpu)
2. Real PaddleOCR Indic Route (devanagari_PP-OCRv5_mobile_rec, Devanagari/Hindi, device=cpu)
3. Mock OCR Provider baseline (clearly labeled as mock)
"""

from __future__ import annotations

import time
from uuid import uuid4

import cv2
import numpy as np

from videx.domain.schemas import FrameTimestamp
from videx.ingestion.base import DecodedFrame, TimestampSource
from videx.ocr.base import MockOCRConfig
from videx.ocr.mock import MockOCRProvider
from videx.ocr.paddle import PaddleGeneralOCRProvider, PaddleIndicOCRProvider


def make_test_frame(text: str, frame_idx: int = 0) -> DecodedFrame:
    width, height = 640, 360
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(
        img,
        text,
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    ts = FrameTimestamp(
        frame_index=frame_idx,
        pts_seconds=frame_idx * 0.04,
        timestamp_source=TimestampSource.CONTAINER,
    )
    return DecodedFrame(
        frame_index=frame_idx,
        timestamp_seconds=ts.pts_seconds,
        width=width,
        height=height,
        frame_array=img,
        video_id=uuid4(),
        frame_timestamp=ts,
    )


def run_benchmark() -> None:
    print("=" * 70)
    print("VIDEX Phase 3.0R — OCR Performance & Telemetry Benchmark")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Real PaddleOCR General Provider Benchmark (PP-OCRv6)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[REAL BENCHMARK] PaddleGeneralOCRProvider (Latin / PP-OCRv6)")
    real_general = PaddleGeneralOCRProvider()
    t_warm = time.perf_counter()
    real_general.warmup()
    warmup_time = time.perf_counter() - t_warm
    print(f"  Warmup / Model Load time: {warmup_time:.2f} s")

    test_frames = [make_test_frame(f"VIDEX TEST {i:02d}", i) for i in range(5)]
    observations_count = 0
    latencies: list[float] = []

    for f in test_frames:
        t0 = time.perf_counter()
        obs = real_general.detect_text(f)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        observations_count += len(obs)

    mean_lat = sum(latencies) / len(latencies)

    # Batch measurement
    batch_frames = [make_test_frame(f"BATCH {i}", i) for i in range(3)]
    t_batch0 = time.perf_counter()
    _ = real_general.detect_text_batch(batch_frames)
    batch_lat = (time.perf_counter() - t_batch0) * 1000.0

    print("  Provider:               paddle_general (REAL)")
    print("  Model:                  PP-OCRv6_medium_det + PP-OCRv6_medium_rec")
    print("  Device:                 cpu (oneDNN disabled)")
    print(f"  Frames processed:       {len(test_frames)}")
    print(f"  Observations generated: {observations_count}")
    print(f"  Mean latency/frame:     {mean_lat:.2f} ms")
    print(f"  Batch latency (3 f):    {batch_lat:.2f} ms ({batch_lat / 3:.2f} ms/frame)")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Real PaddleOCR Indic Provider Benchmark (devanagari_PP-OCRv5_mobile_rec)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[REAL BENCHMARK] PaddleIndicOCRProvider (Devanagari / Indic)")
    real_indic = PaddleIndicOCRProvider()
    t_warm_indic = time.perf_counter()
    real_indic.warmup()
    warmup_indic_time = time.perf_counter() - t_warm_indic
    print(f"  Warmup / Model Load time: {warmup_indic_time:.2f} s")

    indic_frames = [make_test_frame(f"TEST {i:02d}", i) for i in range(3)]
    indic_obs_count = 0
    indic_latencies: list[float] = []

    for f in indic_frames:
        t0 = time.perf_counter()
        obs = real_indic.detect_text(f)
        lat = (time.perf_counter() - t0) * 1000.0
        indic_latencies.append(lat)
        indic_obs_count += len(obs)

    indic_mean_lat = sum(indic_latencies) / len(indic_latencies)

    print("  Provider:               paddle_indic (REAL)")
    print("  Model:                  PP-OCRv5_server_det + devanagari_PP-OCRv5_mobile_rec")
    print("  Device:                 cpu (oneDNN disabled)")
    print(f"  Frames processed:       {len(indic_frames)}")
    print(f"  Observations generated: {indic_obs_count}")
    print(f"  Mean latency/frame:     {indic_mean_lat:.2f} ms")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Mock OCR Baseline (Clearly Labeled as MOCK)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[MOCK BASELINE] MockOCRProvider (Simulation / Hermetic Baseline)")
    mock_canned: dict[int, list[dict[str, object]]] = {
        i: [{"text": "MOCK 123", "confidence": 0.99, "language": "en", "script": "Latin"}]
        for i in range(10)
    }
    mock_provider = MockOCRProvider(
        MockOCRConfig(
            provider_name="mock_baseline",
            canned_results_by_frame=mock_canned,
        )
    )
    mock_provider.warmup()

    mock_frames = [make_test_frame(f"MOCK {i}", i) for i in range(10)]
    t_mock0 = time.perf_counter()
    mock_obs = [mock_provider.detect_text(f) for f in mock_frames]
    mock_time = (time.perf_counter() - t_mock0) * 1000.0
    mock_obs_count = sum(len(o) for o in mock_obs)

    print("  Provider:               mock_baseline (MOCK / IN-MEMORY)")
    print("  Model:                  None (deterministic canned observations)")
    print("  Device:                 cpu / in-memory")
    print(f"  Frames processed:       {len(mock_frames)}")
    print(f"  Observations generated: {mock_obs_count}")
    print(f"  Mean latency/frame:     {mock_time / len(mock_frames):.4f} ms")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
