"""Unit tests for VIDEX Phase 3.0 OCR Architecture, Schemas, Routing, Fusion, and Pipeline.

Deterministic and hermetic test suite covering:
- OCRProvider protocol compliance
- OCRObservation / TextObservation schema and validation
- FrameTimestamp provenance preservation (CONTAINER vs DERIVED)
- Conservative normalization (including Indic Devanagari ZWJ/ZWNJ preservation)
- OCRRouter language and script selection (English/Latin, Hindi/Devanagari, custom, fallback)
- TemporalOCRFusion:
  - Deduplication across continuous frames (N frames -> 1 TextObservation)
  - Spatial continuity (IoU and centroid distance)
  - Temporal continuity & frame gap handling
  - Disappearing text and short bursts
  - Text changes in the same spatial region ("1234" -> "5678")
  - Preservation of underlying raw OCRObservations
  - Empty OCR results handling
- OCRScheduler:
  - Sampling strategies (ALL_FRAMES, FIXED_INTERVAL, TEMPORAL_INTERVAL, SCENE_BOUNDARY)
  - Detection-guided ROI crop hook with FrameTimestamp inheritance
- OCRPipeline integration and Evidence record creation
- Mock and injected Paddle OCR execution
"""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from videx.domain.schemas import (
    BoundingBox,
    EvidenceType,
    OCRObservation,
    RawOCRObservation,
    TextObservation,
)
from videx.ingestion.base import (
    DecodedFrame,
    FrameTimestamp,
    TimestampSource,
)
from videx.ocr.base import MockOCRConfig, PaddleOCRConfig
from videx.ocr.fusion import TemporalOCRFusion, TemporalOCRFusionConfig
from videx.ocr.mock import MockOCRProvider
from videx.ocr.normalization import normalize_text
from videx.ocr.paddle import (
    PaddleGeneralOCRProvider,
    PaddleIndicOCRProvider,
)
from videx.ocr.pipeline import OCRPipeline
from videx.ocr.router import OCRRouter, OCRRouterConfig
from videx.ocr.scheduler import (
    OCRScheduler,
    OCRSchedulerConfig,
    OCRSchedulingStrategy,
)
from videx.providers.base import OCRProvider

# ── Helper Fixtures ────────────────────────────────────────────────────────────


def make_frame_timestamp(
    frame_index: int,
    pts_seconds: float,
    source: TimestampSource = TimestampSource.CONTAINER,
) -> FrameTimestamp:
    """Helper to construct an authoritative FrameTimestamp."""
    return FrameTimestamp(
        frame_index=frame_index,
        pts_seconds=pts_seconds,
        timestamp_source=source,
    )


def make_decoded_frame(
    frame_index: int,
    pts_seconds: float,
    video_id: UUID | None = None,
    source: TimestampSource = TimestampSource.CONTAINER,
    width: int = 1280,
    height: int = 720,
) -> DecodedFrame:
    """Helper to construct a DecodedFrame with synthetic numpy array."""
    vid = video_id or uuid4()
    ts = make_frame_timestamp(frame_index, pts_seconds, source)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    return DecodedFrame(
        frame_index=frame_index,
        timestamp_seconds=pts_seconds,
        width=width,
        height=height,
        frame_array=arr,
        video_id=vid,
        frame_timestamp=ts,
    )


# ── 1. OCRProvider Protocol Compliance ─────────────────────────────────────────


class TestOCRProviderProtocol:
    def test_mock_provider_satisfies_protocol(self) -> None:
        provider = MockOCRProvider()
        assert isinstance(provider, OCRProvider)
        assert provider.provider_name == "mock_ocr"
        assert "en" in provider.supported_languages
        assert "hi" in provider.supported_languages

    def test_paddle_general_provider_satisfies_protocol(self) -> None:
        provider = PaddleGeneralOCRProvider()
        assert isinstance(provider, OCRProvider)
        assert provider.provider_name == "paddle_general"
        assert "en" in provider.supported_languages

    def test_paddle_indic_provider_satisfies_protocol(self) -> None:
        provider = PaddleIndicOCRProvider()
        assert isinstance(provider, OCRProvider)
        assert provider.provider_name == "paddle_indic"
        assert "hi" in provider.supported_languages
        assert provider.config.script == "Devanagari"

    def test_mock_provider_warmup(self) -> None:
        provider = MockOCRProvider()
        assert not provider._warmed_up
        provider.warmup()
        assert provider._warmed_up

    def test_mock_batch_contract(self) -> None:
        provider = MockOCRProvider(
            config=MockOCRConfig(
                canned_texts=["Plate ABC"],
                canned_bboxes=[BoundingBox(x=10, y=20, width=100, height=30)],
            )
        )
        f1 = make_decoded_frame(0, 0.0)
        f2 = make_decoded_frame(1, 0.04)
        batch_results = provider.detect_text_batch([f1, f2])
        assert len(batch_results) == 2
        assert len(batch_results[0]) == 1
        assert len(batch_results[1]) == 1
        assert batch_results[0][0].text == "Plate ABC"
        assert batch_results[1][0].text == "Plate ABC"


# ── 2. OCRObservation / TextObservation Schema & Timestamps ────────────────────


class TestOCRObservationSchema:
    def test_valid_ocr_observation_with_timestamp_provenance(self) -> None:
        vid = uuid4()
        fid = uuid4()
        ts = make_frame_timestamp(42, 1.400, TimestampSource.CONTAINER)

        obs = OCRObservation(
            observation_id=uuid4(),
            frame_id=fid,
            video_id=vid,
            frame_number=42,
            frame_timestamp=ts,
            text="MH 12 AB 1234",
            normalized_text="MH 12 AB 1234",
            confidence=0.96,
            bbox=BoundingBox(x=100, y=200, width=150, height=40),
            language="en",
            script="Latin",
            provider="paddle_general",
        )
        assert obs.frame_number == 42
        assert obs.frame_timestamp.frame_index == 42
        assert obs.frame_timestamp.pts_seconds == 1.400
        assert obs.frame_timestamp.timestamp_source == TimestampSource.CONTAINER
        assert obs.timestamp_seconds == 1.400
        assert obs.ocr_id == obs.observation_id
        assert RawOCRObservation is OCRObservation

    def test_derived_timestamp_preservation(self) -> None:
        vid = uuid4()
        fid = uuid4()
        ts = make_frame_timestamp(10, 0.333, TimestampSource.DERIVED)

        obs = OCRObservation(
            frame_id=fid,
            video_id=vid,
            frame_number=10,
            frame_timestamp=ts,
            text="SPEED 40",
            confidence=0.88,
            provider="mock_ocr",
        )
        assert obs.frame_timestamp.timestamp_source == TimestampSource.DERIVED
        assert obs.timestamp_seconds == 0.333

    def test_confidence_validation(self) -> None:
        vid = uuid4()
        fid = uuid4()
        ts = make_frame_timestamp(0, 0.0)

        with pytest.raises(ValidationError):
            OCRObservation(
                frame_id=fid,
                video_id=vid,
                frame_timestamp=ts,
                text="TEST",
                confidence=1.5,  # Invalid: > 1.0
                provider="mock_ocr",
            )

        with pytest.raises(ValidationError):
            OCRObservation(
                frame_id=fid,
                video_id=vid,
                frame_timestamp=ts,
                text="TEST",
                confidence=-0.1,  # Invalid: < 0.0
                provider="mock_ocr",
            )

    def test_bbox_and_polygon_fields(self) -> None:
        vid = uuid4()
        fid = uuid4()
        ts = make_frame_timestamp(1, 0.04)

        box = BoundingBox(x=50.0, y=60.0, width=100.0, height=40.0)
        poly = [[50.0, 60.0], [150.0, 60.0], [150.0, 100.0], [50.0, 100.0]]

        obs = OCRObservation(
            frame_id=fid,
            video_id=vid,
            frame_timestamp=ts,
            text="EXIT",
            confidence=0.92,
            bbox=box,
            polygon=poly,
            provider="mock_ocr",
        )
        assert obs.bbox == box
        assert obs.polygon == poly

    def test_quality_signals(self) -> None:
        vid = uuid4()
        fid = uuid4()
        ts = make_frame_timestamp(5, 0.20)

        obs = OCRObservation(
            frame_id=fid,
            video_id=vid,
            frame_timestamp=ts,
            text="STOP",
            confidence=0.95,
            provider="paddle_general",
            recognition_confidence=0.98,
            detection_confidence=0.92,
            orientation=0.0,
            crop_quality=0.85,
            preprocessing_applied=["clahe", "dewarp"],
        )
        assert obs.recognition_confidence == 0.98
        assert obs.detection_confidence == 0.92
        assert obs.preprocessing_applied == ["clahe", "dewarp"]


class TestTextObservationSchema:
    def test_fused_text_observation_fields(self) -> None:
        vid = uuid4()
        t1 = make_frame_timestamp(10, 0.40)
        t2 = make_frame_timestamp(12, 0.48)
        o1 = uuid4()
        o2 = uuid4()

        fused = TextObservation(
            video_id=vid,
            text="IND 1234",
            normalized_text="IND 1234",
            first_seen_timestamp=t1,
            last_seen_timestamp=t2,
            first_seen_frame=10,
            last_seen_frame=12,
            supporting_frames=[10, 11, 12],
            supporting_observation_ids=[o1, o2],
            confidence_summary={"mean": 0.94, "max": 0.96, "min": 0.92},
            bbox_history={10: [100.0, 200.0, 80.0, 25.0], 12: [102.0, 201.0, 80.0, 25.0]},
            language="en",
            script="Latin",
            provider="paddle_general",
        )
        assert fused.text == "IND 1234"
        assert fused.first_seen_frame == 10
        assert fused.last_seen_frame == 12
        assert fused.first_seen_timestamp_seconds == 0.40
        assert fused.last_seen_timestamp_seconds == 0.48
        assert fused.duration_seconds == pytest.approx(0.08)
        assert len(fused.supporting_frames) == 3
        assert len(fused.supporting_observation_ids) == 2


# ── 3. Conservative Normalization ──────────────────────────────────────────────


class TestTextNormalization:
    def test_whitespace_and_unicode_nfkc(self) -> None:
        raw = "  Hello \t \n  World \u00a0 "
        norm = normalize_text(raw)
        assert norm == "Hello World"

    def test_case_folding_option(self) -> None:
        raw = "UPPER CASE"
        assert normalize_text(raw, lowercase=False) == "UPPER CASE"
        assert normalize_text(raw, lowercase=True) == "upper case"

    def test_indic_devanagari_zwj_zwnj_preservation(self) -> None:
        # Hindi word "क्या" (kya) uses half-ka formed with Virama + ZWJ or pure unicode
        # Let's test explicit string containing Devanagari Ka (\u0915), Virama (\u094d),
        # ZWJ (\u200d), and Ya (\u092f)
        indic_word = "\u0915\u094d\u200d\u092f\u093e"
        norm = normalize_text(indic_word)
        # ZWJ must be preserved for conjunct formation
        assert "\u200d" in norm
        assert norm == indic_word

    def test_indic_zwnj_preservation(self) -> None:
        # ZWNJ (\u200c) separates consonants preventing conjunct formation
        indic_word = "\u0915\u094d\u200c\u0915"
        norm = normalize_text(indic_word)
        assert "\u200c" in norm

    def test_strip_zero_width_space_and_bom(self) -> None:
        raw = "\ufeff\u200bHello\u200bWorld"
        norm = normalize_text(raw)
        assert norm == "HelloWorld"


# ── 4. OCRRouter Language / Script Selection ───────────────────────────────────


class TestOCRRouter:
    def test_default_routes(self) -> None:
        router = OCRRouter()
        general_p = router.get_provider(language="en")
        indic_p = router.get_provider(language="hi")

        assert general_p.provider_name == "paddle_general"
        assert indic_p.provider_name == "paddle_indic"

    def test_script_routing(self) -> None:
        router = OCRRouter()
        # Explicit script routing
        devanagari_p = router.get_provider(script="Devanagari")
        latin_p = router.get_provider(script="Latin")

        assert devanagari_p.provider_name == "paddle_indic"
        assert latin_p.provider_name == "paddle_general"

    def test_additional_indic_languages_routed_to_indic(self) -> None:
        router = OCRRouter()
        # Marathi (mr) and Sanskrit (sa) use Devanagari script
        marathi_p = router.get_provider(language="mr")
        sanskrit_p = router.get_provider(language="sa")

        assert marathi_p.provider_name == "paddle_indic"
        assert sanskrit_p.provider_name == "paddle_indic"

    def test_unknown_language_falls_back_to_general(self) -> None:
        router = OCRRouter()
        unknown_p = router.get_provider(language="unknown_xyz")
        assert unknown_p.provider_name == "paddle_general"

    def test_custom_provider_injection(self) -> None:
        mock_p = MockOCRProvider(MockOCRConfig(provider_name="custom_mock"))
        config = OCRRouterConfig(
            providers={"custom": mock_p},
            language_map={"custom_lang": "custom"},
            default_provider_key="custom",
        )
        router = OCRRouter(config)
        p = router.get_provider(language="custom_lang")
        assert p.provider_name == "custom_mock"


# ── 5. TemporalOCRFusion ───────────────────────────────────────────────────────


class TestTemporalOCRFusion:
    def test_deduplicate_continuous_frames(self) -> None:
        """Frame 100, 101, 102, 103 with same text and bbox should fuse into 1 TextObservation."""
        vid = uuid4()
        fusion = TemporalOCRFusion(
            TemporalOCRFusionConfig(
                spatial_iou_threshold=0.5,
                max_frame_gap=2,
            )
        )

        obs_list = []
        for i in range(4):
            frame_num = 100 + i
            ts = make_frame_timestamp(frame_num, 4.0 + i * 0.04)
            obs = OCRObservation(
                observation_id=uuid4(),
                frame_id=uuid4(),
                video_id=vid,
                frame_number=frame_num,
                frame_timestamp=ts,
                text="IND 1234",
                normalized_text="IND 1234",
                confidence=0.90 + i * 0.02,
                bbox=BoundingBox(x=100.0, y=200.0, width=80.0, height=25.0),
                language="en",
                script="Latin",
                provider="mock_ocr",
            )
            obs_list.append(obs)
            fusion.add_observation(obs)

        fused = fusion.finalize()
        assert len(fused) == 1
        f = fused[0]
        assert f.text == "IND 1234"
        assert f.first_seen_frame == 100
        assert f.last_seen_frame == 103
        assert f.supporting_frames == [100, 101, 102, 103]
        assert len(f.supporting_observation_ids) == 4
        assert f.confidence_summary["count"] == 4
        assert f.confidence_summary["max"] == pytest.approx(0.96)
        assert f.confidence_summary["min"] == pytest.approx(0.90)

    def test_raw_observations_not_destroyed(self) -> None:
        """Fusion returns fused TextObservations while raw OCRObservations remain intact."""
        vid = uuid4()
        fusion = TemporalOCRFusion()
        raw1 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=0,
            frame_timestamp=make_frame_timestamp(0, 0.0),
            text="RAW TEXT",
            confidence=0.95,
            provider="mock_ocr",
        )
        fusion.add_observation(raw1)
        fused = fusion.finalize()

        assert len(fused) == 1
        assert raw1.observation_id in fused[0].supporting_observation_ids
        # raw1 object is completely preserved and unaltered
        assert raw1.text == "RAW TEXT"

    def test_text_change_in_same_region(self) -> None:
        """Text changing from '1234' to '5678' in the same bbox must yield two observations."""
        vid = uuid4()
        fusion = TemporalOCRFusion()
        box = BoundingBox(x=100.0, y=200.0, width=80.0, height=25.0)

        # 2 frames of "1234"
        obs1 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=1,
            frame_timestamp=make_frame_timestamp(1, 0.04),
            text="1234",
            normalized_text="1234",
            confidence=0.95,
            bbox=box,
            provider="mock_ocr",
        )
        obs2 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=2,
            frame_timestamp=make_frame_timestamp(2, 0.08),
            text="1234",
            normalized_text="1234",
            confidence=0.95,
            bbox=box,
            provider="mock_ocr",
        )
        # 2 frames of "5678"
        obs3 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=3,
            frame_timestamp=make_frame_timestamp(3, 0.12),
            text="5678",
            normalized_text="5678",
            confidence=0.95,
            bbox=box,
            provider="mock_ocr",
        )
        obs4 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=4,
            frame_timestamp=make_frame_timestamp(4, 0.16),
            text="5678",
            normalized_text="5678",
            confidence=0.95,
            bbox=box,
            provider="mock_ocr",
        )

        for obs in [obs1, obs2, obs3, obs4]:
            fusion.add_observation(obs)

        fused = fusion.finalize()
        assert len(fused) == 2
        fused_texts = {f.text for f in fused}
        assert fused_texts == {"1234", "5678"}

    def test_spatial_discontinuity_creates_separate_observations(self) -> None:
        """Same text at completely different screen coordinates creates two observations."""
        vid = uuid4()
        fusion = TemporalOCRFusion(
            TemporalOCRFusionConfig(
                spatial_iou_threshold=0.3,
                max_centroid_distance_px=50.0,
            )
        )
        box_left = BoundingBox(x=10.0, y=10.0, width=50.0, height=20.0)
        box_right = BoundingBox(x=800.0, y=600.0, width=50.0, height=20.0)

        obs1 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=0,
            frame_timestamp=make_frame_timestamp(0, 0.0),
            text="EXIT",
            normalized_text="EXIT",
            confidence=0.9,
            bbox=box_left,
            provider="mock_ocr",
        )
        obs2 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=1,
            frame_timestamp=make_frame_timestamp(1, 0.04),
            text="EXIT",
            normalized_text="EXIT",
            confidence=0.9,
            bbox=box_right,
            provider="mock_ocr",
        )

        fusion.add_observation(obs1)
        fusion.add_observation(obs2)
        fused = fusion.finalize()
        assert len(fused) == 2

    def test_temporal_gap_timeout(self) -> None:
        """Frame gap exceeding max_frame_gap splits the observation into two."""
        vid = uuid4()
        fusion = TemporalOCRFusion(TemporalOCRFusionConfig(max_frame_gap=3))
        box = BoundingBox(x=100.0, y=100.0, width=80.0, height=25.0)

        obs1 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=1,
            frame_timestamp=make_frame_timestamp(1, 0.04),
            text="SIGN",
            normalized_text="SIGN",
            confidence=0.9,
            bbox=box,
            provider="mock_ocr",
        )
        # Gap of 10 frames (1 -> 12)
        obs2 = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=12,
            frame_timestamp=make_frame_timestamp(12, 0.48),
            text="SIGN",
            normalized_text="SIGN",
            confidence=0.9,
            bbox=box,
            provider="mock_ocr",
        )

        fusion.add_observation(obs1)
        fusion.add_observation(obs2)
        fused = fusion.finalize()
        assert len(fused) == 2
        assert fused[0].first_seen_frame == 1
        assert fused[1].first_seen_frame == 12

    def test_short_burst_and_disappearing_text(self) -> None:
        """A single frame detection should finalize into a valid single-frame TextObservation."""
        vid = uuid4()
        fusion = TemporalOCRFusion()
        obs = OCRObservation(
            frame_id=uuid4(),
            video_id=vid,
            frame_number=45,
            frame_timestamp=make_frame_timestamp(45, 1.80),
            text="FLASH",
            normalized_text="FLASH",
            confidence=0.85,
            provider="mock_ocr",
        )
        fusion.add_observation(obs)
        fused = fusion.finalize()
        assert len(fused) == 1
        assert fused[0].first_seen_frame == 45
        assert fused[0].last_seen_frame == 45
        assert fused[0].supporting_frames == [45]


# ── 6. OCRScheduler & Region Crop Hook ─────────────────────────────────────────


class TestOCRScheduler:
    def test_all_frames_strategy(self) -> None:
        scheduler = OCRScheduler(OCRSchedulerConfig(strategy=OCRSchedulingStrategy.ALL_FRAMES))
        for i in range(5):
            f = make_decoded_frame(i, i * 0.04)
            assert scheduler.should_ocr(f)

    def test_fixed_interval_strategy(self) -> None:
        scheduler = OCRScheduler(
            OCRSchedulerConfig(
                strategy=OCRSchedulingStrategy.FIXED_INTERVAL,
                frame_interval=3,
            )
        )
        selected_frames = []
        for i in range(10):
            f = make_decoded_frame(i, i * 0.04)
            if scheduler.should_ocr(f):
                selected_frames.append(i)
        assert selected_frames == [0, 3, 6, 9]

    def test_temporal_interval_strategy(self) -> None:
        scheduler = OCRScheduler(
            OCRSchedulerConfig(
                strategy=OCRSchedulingStrategy.TEMPORAL_INTERVAL,
                temporal_interval_seconds=0.5,
            )
        )
        selected_pts = []
        for i in range(25):
            pts = i * 0.1  # 0.0, 0.1, 0.2, ...
            f = make_decoded_frame(i, pts)
            if scheduler.should_ocr(f):
                selected_pts.append(pts)
        assert selected_pts == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_scene_boundary_strategy(self) -> None:
        scheduler = OCRScheduler(
            OCRSchedulerConfig(
                strategy=OCRSchedulingStrategy.SCENE_BOUNDARY,
                scene_boundary_frame_indices={0, 15, 30},
            )
        )
        selected = [i for i in range(35) if scheduler.should_ocr(make_decoded_frame(i, i * 0.04))]
        assert selected == [0, 15, 30]

    def test_detection_guided_region_crop_hook(self) -> None:
        orig_frame = make_decoded_frame(10, 0.40, width=640, height=480)
        # Put known pixel values inside crop region
        assert isinstance(orig_frame.frame_array, np.ndarray)
        orig_frame.frame_array[50:100, 100:200] = 255

        roi_bbox = BoundingBox(x=100.0, y=50.0, width=100.0, height=50.0)
        cropped = OCRScheduler.create_cropped_frame(orig_frame, roi_bbox, margin_px=0)

        assert cropped.frame_index == 10
        assert cropped.timestamp_seconds == 0.40
        assert cropped.frame_timestamp == orig_frame.frame_timestamp
        assert cropped.video_id == orig_frame.video_id
        assert cropped.width == 100
        assert cropped.height == 50
        assert isinstance(cropped.frame_array, np.ndarray)
        assert np.all(cropped.frame_array == 255)


# ── 7. Full OCR Pipeline & Evidence Linkage ───────────────────────────────────


class TestOCRPipeline:
    def test_pipeline_execution_and_evidence_creation(self) -> None:
        vid = uuid4()
        mock_provider = MockOCRProvider(
            config=MockOCRConfig(
                provider_name="test_mock",
                canned_texts=["MH 12 AB 1234"],
                canned_bboxes=[BoundingBox(x=100.0, y=200.0, width=150.0, height=40.0)],
                canned_confidence=0.95,
                language="en",
                script="Latin",
            )
        )
        router = OCRRouter(
            OCRRouterConfig(
                providers={"test_mock": mock_provider},
                default_provider_key="test_mock",
            )
        )
        pipeline = OCRPipeline(
            router=router,
            scheduler=OCRScheduler(OCRSchedulerConfig(strategy=OCRSchedulingStrategy.ALL_FRAMES)),
            fusion=TemporalOCRFusion(),
        )

        frames = [make_decoded_frame(i, i * 0.04, video_id=vid) for i in range(5)]
        raw_obs, fused_obs, evidence = pipeline.process_frames(frames, language="en")

        assert len(raw_obs) == 5
        assert len(fused_obs) == 1
        assert len(evidence) == 1

        ev = evidence[0]
        assert ev.evidence_type == EvidenceType.OCR
        assert ev.video_id == vid
        assert ev.timestamp_seconds == 0.0
        assert ev.confidence == pytest.approx(0.95)
        assert ev.source_module == "test_mock"
        assert ev.supporting_observation_ids == fused_obs[0].supporting_observation_ids
        assert ev.raw_payload["text"] == "MH 12 AB 1234"
        assert ev.raw_payload["first_seen_frame"] == 0
        assert ev.raw_payload["last_seen_frame"] == 4

    def test_pipeline_empty_frames(self) -> None:
        pipeline = OCRPipeline()
        raw_obs, fused_obs, evidence = pipeline.process_frames([])
        assert raw_obs == []
        assert fused_obs == []
        assert evidence == []


# ── 8. Injected Paddle OCR Adapter Execution ───────────────────────────────────


class TestPaddleOCRInjectedEngine:
    class MockPaddleEngine:
        def __init__(self) -> None:
            pass

        def ocr(
            self, img: np.ndarray, cls: bool = True
        ) -> list[list[tuple[list[list[float]], tuple[str, float]]]]:
            # Return paddle-style polygon and (text, confidence)
            poly = [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]]
            return [[(poly, ("नमस्ते VIDEX", 0.97))]]

    def test_paddle_indic_with_mock_engine(self) -> None:
        engine = self.MockPaddleEngine()
        config = PaddleOCRConfig(
            provider_name="paddle_indic_test",
            language="hi",
            script="Devanagari",
        )
        provider = PaddleIndicOCRProvider(config=config, injected_engine=engine)
        frame = make_decoded_frame(0, 0.0)

        obs_list = provider.detect_text(frame)
        assert len(obs_list) == 1
        obs = obs_list[0]
        assert obs.text == "नमस्ते VIDEX"
        assert obs.normalized_text == "नमस्ते VIDEX"
        assert obs.confidence == pytest.approx(0.97)
        assert obs.language == "hi"
        assert obs.script == "Devanagari"
        assert obs.provider == "paddle_indic_test"
        assert obs.bbox is not None
        assert obs.bbox.x == 10.0
        assert obs.bbox.y == 20.0
        assert obs.bbox.width == 90.0
        assert obs.bbox.height == 30.0
        assert obs.polygon is not None
        assert obs.polygon[0] == [10.0, 20.0]
