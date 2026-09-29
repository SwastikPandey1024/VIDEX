"""Integration and acceptance tests for VIDEX Phase 3.0 OCR pipeline.

Validates:
1. Real OCR Acceptance Test:
   - Evaluates real PaddleOCRProvider if paddleocr runtime is installed and configured.
   - Skips cleanly if paddleocr package or model weights are absent.
   - Never triggers automatic weight downloads during CI.

2. Temporal OCR Acceptance Test:
   - Synthetic sequence where identical text appears over multiple frames:
     N raw OCR observations -> 1 fused TextObservation
     Asserts: first_seen_frame, last_seen_frame, supporting_frames, supporting_observation_ids.
   - Changing text sequence ("1234" -> "1234" -> "5678" -> "5678"):
     Asserts: two separate fused TextObservations are created.

3. VideoReader End-to-End Pipeline Acceptance Test:
   - Generates an actual mp4 test video file.
   - Reads frames via OpenCVVideoReader into DecodedFrame.
   - Processes frames through OCRPipeline.
   - Asserts FrameTimestamp provenance, numeric synchronization, and Evidence records.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from videx.domain.schemas import (
    BoundingBox,
    EvidenceType,
    FrameTimestamp,
    OCRObservation,
    TextObservation,
)
from videx.ingestion.base import DecodedFrame, TimestampSource
from videx.ingestion.reader import OpenCVVideoReader
from videx.ocr.base import MockOCRConfig
from videx.ocr.fusion import TemporalOCRFusion, TemporalOCRFusionConfig
from videx.ocr.mock import MockOCRProvider
from videx.ocr.paddle import PaddleGeneralOCRProvider, PaddleIndicOCRProvider
from videx.ocr.pipeline import OCRPipeline
from videx.ocr.router import OCRRouter, OCRRouterConfig
from videx.ocr.scheduler import OCRScheduler, OCRSchedulerConfig, OCRSchedulingStrategy


def _is_paddleocr_available() -> bool:
    """Check if paddleocr package and local weights/artifacts are available."""
    if importlib.util.find_spec("paddleocr") is None:
        return False
    # If explicitly enabled or custom model directory set
    if os.environ.get("VIDEX_ENABLE_REAL_PADDLE_ACCEPTANCE") == "1":
        return True
    # By default, do not trigger download in hermetic runs: check local model cache
    paddlex_home = Path.home() / ".paddlex" / "official_models"
    if (paddlex_home / "PP-OCRv6_medium_rec").exists():
        return True
    return False


def _is_paddle_indic_available() -> bool:
    """Check if paddleocr package and local Devanagari model weights are available."""
    if importlib.util.find_spec("paddleocr") is None:
        return False
    if os.environ.get("VIDEX_ENABLE_REAL_PADDLE_ACCEPTANCE") == "1":
        return True
    paddlex_home = Path.home() / ".paddlex" / "official_models"
    if (paddlex_home / "devanagari_PP-OCRv5_mobile_rec").exists():
        return True
    return False


def _create_hindi_text_image(text: str, width: int = 400, height: int = 200) -> np.ndarray:
    """Create a BGR numpy image containing Hindi/Devanagari text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = None
    for font_path in [
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/mangal.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(font_path).exists():
            try:
                font = ImageFont.truetype(font_path, 36)
                break
            except Exception:
                continue
    draw.text((40, 70), text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 1. Real PaddleOCR Acceptance Test (Skipped when artifacts absent) ──────────


def test_real_paddleocr_acceptance(tmp_path: Path) -> None:
    """Acceptance gate for real PaddleOCR runtime.

    Must skip cleanly when runtime or artifacts are absent (zero CI network download).
    Executes complete chain:
        test video -> VideoReader -> DecodedFrame -> PaddleOCRProvider -> OCRObservation
    """
    if not _is_paddleocr_available():
        pytest.skip(
            "PaddleOCR runtime or model artifacts absent; skipping real OCR acceptance test."
        )

    # 1. Write synthetic video with English text
    video_path = tmp_path / "real_paddle_acceptance.mp4"
    frame_width, frame_height = 640, 360
    fps = 25.0
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
    img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    cv2.putText(
        img,
        "VIDEX OCR 2026",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    out.write(img)
    out.release()

    # 2. Ingest via VideoReader into DecodedFrame
    reader = OpenCVVideoReader(str(video_path))
    decoded = reader.read_decoded_frame(0)

    # 3. Process through real PaddleGeneralOCRProvider
    provider = PaddleGeneralOCRProvider()
    provider.warmup()
    observations = provider.detect_text(decoded)

    # 4. Verify OCR properties
    assert len(observations) >= 1
    obs = observations[0]
    assert obs.text != ""
    assert obs.confidence >= 0.0 and obs.confidence <= 1.0
    assert obs.bbox is not None
    assert obs.bbox.width > 0 and obs.bbox.height > 0
    assert obs.polygon is not None and len(obs.polygon) == 4
    assert obs.language == "en"
    assert obs.script == "Latin"
    assert obs.frame_number == 0
    assert obs.frame_timestamp == decoded.frame_timestamp
    assert (
        obs.attributes["timestamp_source"]
        == decoded.frame_timestamp.timestamp_source.value
    )
    assert obs.provider == "paddle_general"


# ── 2. Hindi / Devanagari Acceptance Test ──────────────────────────────────────


def test_real_paddleocr_hindi_acceptance(tmp_path: Path) -> None:
    """Acceptance gate for real PaddleOCR Hindi / Devanagari recognition route.

    Must skip cleanly when runtime or devanagari_PP-OCRv5_mobile_rec artifacts are absent.
    Executes complete chain:
        test video -> VideoReader -> DecodedFrame -> PaddleIndicOCRProvider -> OCRObservation
    """
    if not _is_paddle_indic_available():
        pytest.skip(
            "PaddleOCR runtime or devanagari_PP-OCRv5_mobile_rec artifacts absent; "
            "skipping Hindi acceptance test."
        )

    # 1. Write synthetic video with Hindi text
    video_path = tmp_path / "hindi_acceptance.mp4"
    frame_width, frame_height = 400, 200
    fps = 25.0
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
    img_bgr = _create_hindi_text_image("भारत 2026", frame_width, frame_height)
    out.write(img_bgr)
    out.release()

    # 2. Ingest via VideoReader into DecodedFrame
    reader = OpenCVVideoReader(str(video_path))
    decoded = reader.read_decoded_frame(0)

    # 3. Process through dedicated PaddleIndicOCRProvider
    provider = PaddleIndicOCRProvider()
    provider.warmup()
    observations = provider.detect_text(decoded)

    # 4. Verify Hindi / Devanagari properties and text extraction
    assert len(observations) >= 1
    obs = observations[0]
    assert obs.text != ""
    assert obs.language == "hi"
    assert obs.script == "Devanagari"
    assert obs.provider == "paddle_indic"
    assert obs.confidence >= 0.0 and obs.confidence <= 1.0
    assert obs.bbox is not None
    assert obs.bbox.width > 0 and obs.bbox.height > 0
    assert obs.polygon is not None and len(obs.polygon) == 4
    assert obs.frame_number == 0
    assert obs.frame_timestamp == decoded.frame_timestamp
    assert (
        obs.attributes["timestamp_source"]
        == decoded.frame_timestamp.timestamp_source.value
    )


# ── 3. Temporal OCR Acceptance Tests ───────────────────────────────────────────


def test_temporal_fusion_repeated_text_sequence() -> None:
    """Acceptance test: N raw observations across time fuse into 1 TextObservation.

    Verifies:
      first_seen_frame
      last_seen_frame
      supporting_frames
      supporting_observation_ids
      confidence summary
    """
    vid = uuid4()
    fusion = TemporalOCRFusion(
        TemporalOCRFusionConfig(
            max_frame_gap=3,
            min_iou_overlap=0.4,
        )
    )

    raw_observations: list[OCRObservation] = []
    num_frames = 6

    for i in range(num_frames):
        ts = FrameTimestamp(
            frame_index=i,
            pts_seconds=round(i * 0.04, 3),
            timestamp_source=TimestampSource.CONTAINER,
        )
        obs = OCRObservation(
            observation_id=uuid4(),
            frame_id=uuid4(),
            video_id=vid,
            frame_number=i,
            frame_timestamp=ts,
            timestamp_seconds=ts.pts_seconds,
            text="MH 12 AB 1234",
            normalized_text="MH 12 AB 1234",
            confidence=0.92 + (0.01 * (i % 3)),
            bbox=BoundingBox(x=120.0, y=240.0, width=160.0, height=45.0),
            language="en",
            script="Latin",
            provider="paddle_general",
        )
        raw_observations.append(obs)
        fusion.add_observation(obs)

    fused_list = fusion.finalize()

    # Must collapse 6 raw observations into exactly 1 TextObservation
    assert len(fused_list) == 1
    fused: TextObservation = fused_list[0]

    assert fused.text == "MH 12 AB 1234"
    assert fused.normalized_text == "MH 12 AB 1234"
    assert fused.first_seen_frame == 0
    assert fused.last_seen_frame == num_frames - 1
    assert fused.supporting_frames == list(range(num_frames))
    assert len(fused.supporting_observation_ids) == num_frames
    assert fused.supporting_observation_ids == [o.observation_id for o in raw_observations]
    assert fused.first_seen_timestamp.pts_seconds == 0.0
    assert fused.last_seen_timestamp.pts_seconds == pytest.approx(0.20)
    assert fused.duration_seconds == pytest.approx(0.20)
    assert fused.confidence_summary["count"] == num_frames
    assert fused.confidence_summary["mean"] >= 0.92


def test_temporal_fusion_text_change_sequence() -> None:
    """Acceptance test: Sequence with changing text in the same region.

    Frames 0, 1 -> "1234"
    Frames 2, 3 -> "5678"

    Must produce two distinct fused TextObservations.
    """
    vid = uuid4()
    fusion = TemporalOCRFusion()
    box = BoundingBox(x=200.0, y=100.0, width=120.0, height=40.0)

    # Sequence of observations
    sequence = [
        (0, 0.00, "1234"),
        (1, 0.04, "1234"),
        (2, 0.08, "5678"),
        (3, 0.12, "5678"),
    ]

    for frame_idx, pts, txt in sequence:
        ts = FrameTimestamp(
            frame_index=frame_idx,
            pts_seconds=pts,
            timestamp_source=TimestampSource.CONTAINER,
        )
        obs = OCRObservation(
            observation_id=uuid4(),
            frame_id=uuid4(),
            video_id=vid,
            frame_number=frame_idx,
            frame_timestamp=ts,
            text=txt,
            normalized_text=txt,
            confidence=0.95,
            bbox=box,
            language="en",
            script="Latin",
            provider="paddle_general",
        )
        fusion.add_observation(obs)

    fused_list = fusion.finalize()

    # Must produce exactly 2 fused observations
    assert len(fused_list) == 2
    by_text = {f.text: f for f in fused_list}
    assert set(by_text.keys()) == {"1234", "5678"}

    f1 = by_text["1234"]
    assert f1.first_seen_frame == 0
    assert f1.last_seen_frame == 1
    assert f1.supporting_frames == [0, 1]

    f2 = by_text["5678"]
    assert f2.first_seen_frame == 2
    assert f2.last_seen_frame == 3
    assert f2.supporting_frames == [2, 3]


# ── 3. VideoReader End-to-End Pipeline Acceptance Test ─────────────────────────


def test_videoreader_to_ocr_pipeline_acceptance(tmp_path: Path) -> None:
    """Acceptance test verifying complete chain:

    test video -> VideoReader -> DecodedFrame -> OCR Provider -> TextObservation -> Evidence
    """
    video_path = tmp_path / "acceptance_test_ocr.mp4"
    frame_width, frame_height = 320, 240
    fps = 25.0
    num_frames = 10

    # Write synthetic mp4 video
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
    for i in range(num_frames):
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 120
        cv2.putText(
            frame,
            f"F_{i}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        out.write(frame)
    out.release()

    # Ingest video with OpenCVVideoReader
    reader = OpenCVVideoReader(str(video_path))
    assert reader.total_frames == num_frames

    # Build mock provider returning deterministic text for first 5 frames
    canned_results = {}
    for i in range(5):
        canned_results[i] = [
            {
                "text": "MH 12 AB 1234",
                "bbox": BoundingBox(x=50.0, y=80.0, width=140.0, height=35.0),
                "confidence": 0.96,
                "language": "en",
                "script": "Latin",
            }
        ]

    mock_p = MockOCRProvider(
        MockOCRConfig(
            provider_name="acceptance_mock",
            canned_results_by_frame=canned_results,
            supported_languages=("en", "hi"),
        )
    )
    router = OCRRouter(
        OCRRouterConfig(
            providers={"acceptance_mock": mock_p},
            default_provider_key="acceptance_mock",
        )
    )
    pipeline = OCRPipeline(
        router=router,
        scheduler=OCRScheduler(OCRSchedulerConfig(strategy=OCRSchedulingStrategy.ALL_FRAMES)),
        fusion=TemporalOCRFusion(),
    )

    # Read all frames and process through pipeline
    decoded_frames: list[DecodedFrame] = [reader.read_decoded_frame(i) for i in range(num_frames)]

    raw_obs, fused_obs, evidence_records = pipeline.process_frames(decoded_frames, language="en")

    # Observations generated for frames 0..4
    assert len(raw_obs) == 5
    # Collapsed into 1 fused text observation
    assert len(fused_obs) == 1
    # Produced 1 canonical Evidence record
    assert len(evidence_records) == 1

    fused = fused_obs[0]
    assert fused.text == "MH 12 AB 1234"
    assert fused.first_seen_frame == 0
    assert fused.last_seen_frame == 4
    assert fused.supporting_frames == [0, 1, 2, 3, 4]
    assert fused.first_seen_timestamp == decoded_frames[0].frame_timestamp
    assert fused.last_seen_timestamp == decoded_frames[4].frame_timestamp

    ev = evidence_records[0]
    assert ev.evidence_type == EvidenceType.OCR
    assert ev.source_module == "acceptance_mock"
    assert ev.supporting_observation_ids == fused.supporting_observation_ids
    assert ev.raw_payload["text"] == "MH 12 AB 1234"
    assert ev.raw_payload["first_seen_frame"] == 0
    assert ev.raw_payload["last_seen_frame"] == 4
