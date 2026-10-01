"""Acceptance and integration tests for VIDEX Temporal Event Intelligence Engine.

Verifies end-to-end event generation across perception, OCR, and audio pipelines,
evidence linking, chronological ordering, edge cases, and throughput benchmarks.
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from videx.audio.pipeline import AudioPipelineResult
from videx.domain.schemas import BoundingBox, EvidenceType
from videx.events.engine import EventEngine, EventTimeline
from videx.events.mock import (
    create_mock_text_observation,
    create_mock_track,
    create_mock_trajectory,
    create_mock_transcript_segment,
    create_mock_zone,
)
from videx.events.rules import CrossModalRuleEngine
from videx.events.schemas import EventEngineConfig
from videx.events.types import EventStatus, EventType
from videx.ocr.pipeline import OCRPipelineResult
from videx.perception.pipeline import PerceptionResult


class TestEventPipelineIntegration:
    """Tests end-to-end multimodal integration into the EventEngine."""

    @pytest.fixture
    def sample_multimodal_data(self) -> dict[str, object]:
        """Generate consistent synthetic multimodal pipeline outputs."""
        vid_id = uuid4()

        # 1. Perception: Vehicle track moving across scene into a zone
        trk = create_mock_track(
            video_id=vid_id,
            class_name="car",
            start_frame=0,
            end_frame=30,
            start_time=0.0,
            end_time=1.2,
        )
        traj = create_mock_trajectory(
            track_id=trk.track_id,
            start_x=50.0,
            start_y=150.0,
            end_x=250.0,
            end_y=150.0,
            frames=31,
            fps=25.0,
        )
        perception_res = PerceptionResult(
            video_id=vid_id,
            tracks=[trk],
            trajectories={trk.track_id: traj},
        )

        # 2. OCR: License plate text recognized on the car
        text_obs = create_mock_text_observation(
            video_id=vid_id,
            text="MH 12 AB 1234",
            start_frame=10,
            end_frame=25,
            start_time=0.4,
            end_time=1.0,
            bbox=BoundingBox(x=120.0, y=140.0, width=60.0, height=20.0),
        )
        ocr_res = OCRPipelineResult(
            video_id=vid_id,
            raw_observations=[],
            fused_observations=[text_obs],
            evidence_records=[],
            frames_evaluated=31,
            frames_processed=16,
            total_latency_ms=12.5,
        )

        # 3. Audio: Speech announcement near the same timestamp
        speech_seg = create_mock_transcript_segment(
            video_id=vid_id,
            text="Vehicle approaching security checkpoint",
            start_time=0.3,
            end_time=1.5,
            language="en",
        )
        audio_res = AudioPipelineResult(
            video_id=vid_id,
            audio_metadata=None,
            raw_segments=[speech_seg],
            fused_segments=[speech_seg],
            evidence=[],
            duration_seconds=1.5,
            language="en",
            processing_time_seconds=0.05,
        )

        # 4. Spatial Zone: Checkpoint gate
        zone = create_mock_zone(
            zone_id="zone_security_gate",
            zone_name="Security Gate",
            x_min=100.0,
            y_min=100.0,
            x_max=200.0,
            y_max=200.0,
        )

        return {
            "video_id": vid_id,
            "perception": perception_res,
            "ocr": ocr_res,
            "audio": audio_res,
            "zone": zone,
            "track": trk,
        }

    def test_multimodal_timeline_generation(
        self, sample_multimodal_data: dict[str, object]
    ) -> None:
        vid_id = sample_multimodal_data["video_id"]
        engine = EventEngine(
            zones=[sample_multimodal_data["zone"]],  # type: ignore[list-item]
        )

        timeline = engine.process_multimodal(
            video_id=vid_id,  # type: ignore[arg-type]
            perception_result=sample_multimodal_data["perception"],  # type: ignore[arg-type]
            ocr_result=sample_multimodal_data["ocr"],  # type: ignore[arg-type]
            audio_result=sample_multimodal_data["audio"],  # type: ignore[arg-type]
        )

        assert isinstance(timeline, EventTimeline)
        assert len(timeline) > 0

        # Verify presence of multiple event families
        types = {e.event_type for e in timeline.events}
        assert EventType.OBJECT_APPEARED in types
        assert EventType.OBJECT_PRESENT in types
        assert EventType.OBJECT_ENTERED_ZONE in types
        assert EventType.TEXT_APPEARED in types
        assert EventType.SPEECH_DETECTED in types

        # Verify strict chronological sorting
        for i in range(len(timeline.events) - 1):
            e1 = timeline.events[i]
            e2 = timeline.events[i + 1]
            assert e1.start_timestamp_seconds <= e2.start_timestamp_seconds

    def test_canonical_evidence_records_created(
        self, sample_multimodal_data: dict[str, object]
    ) -> None:
        vid_id = sample_multimodal_data["video_id"]
        engine = EventEngine(
            zones=[sample_multimodal_data["zone"]],  # type: ignore[list-item]
        )

        timeline = engine.process_multimodal(
            video_id=vid_id,  # type: ignore[arg-type]
            perception_result=sample_multimodal_data["perception"],  # type: ignore[arg-type]
            ocr_result=sample_multimodal_data["ocr"],  # type: ignore[arg-type]
            audio_result=sample_multimodal_data["audio"],  # type: ignore[arg-type]
        )

        evidence = timeline.evidence_records
        assert len(evidence) == len(timeline.events)

        for ev in evidence:
            assert ev.evidence_type == EvidenceType.EVENT
            assert ev.video_id == vid_id
            assert "event_type" in ev.raw_payload
            assert "event_id" in ev.raw_payload
            assert ev.confidence >= 0.0

    def test_explainability_chain_complete(
        self, sample_multimodal_data: dict[str, object]
    ) -> None:
        vid_id = sample_multimodal_data["video_id"]
        engine = EventEngine(
            zones=[sample_multimodal_data["zone"]],  # type: ignore[list-item]
        )

        timeline = engine.process_multimodal(
            video_id=vid_id,  # type: ignore[arg-type]
            perception_result=sample_multimodal_data["perception"],  # type: ignore[arg-type]
            ocr_result=sample_multimodal_data["ocr"],  # type: ignore[arg-type]
            audio_result=sample_multimodal_data["audio"],  # type: ignore[arg-type]
        )

        # Test explanation on every event in the timeline
        for event in timeline.events:
            explanation = timeline.explain_event(event.event_id)
            assert explanation["what"]["event_type"] == event.event_type.value
            assert explanation["what"]["status"] in [s.value for s in EventStatus]
            assert "start_seconds" in explanation["when"]
            assert len(explanation["which_entities"]) > 0
            assert "source_subsystem" in explanation

    def test_cross_modal_compound_rule_evaluation(
        self, sample_multimodal_data: dict[str, object]
    ) -> None:
        vid_id = sample_multimodal_data["video_id"]
        engine = EventEngine(
            zones=[sample_multimodal_data["zone"]],  # type: ignore[list-item]
        )

        timeline = engine.process_multimodal(
            video_id=vid_id,  # type: ignore[arg-type]
            perception_result=sample_multimodal_data["perception"],  # type: ignore[arg-type]
            ocr_result=sample_multimodal_data["ocr"],  # type: ignore[arg-type]
            audio_result=sample_multimodal_data["audio"],  # type: ignore[arg-type]
        )

        rule_engine = CrossModalRuleEngine(temporal_tolerance_seconds=1.5)
        compound = rule_engine.detect_speech_during_zone_presence(timeline.events, video_id=vid_id)  # type: ignore[arg-type]

        assert len(compound) >= 1
        comp = compound[0]
        assert comp.event_type == EventType.CUSTOM
        assert comp.attributes["rule_name"] == "speech_in_zone"
        assert len(comp.event_evidence) == 2


class TestEventEngineEdgeCases:
    """Tests robustness under edge conditions, empty streams, and unusual inputs."""

    def test_empty_multimodal_inputs(self) -> None:
        vid_id = uuid4()
        engine = EventEngine()
        timeline = engine.process_multimodal(video_id=vid_id)
        assert len(timeline) == 0
        assert timeline.evidence_records == []

    def test_out_of_order_evidence_sorted_deterministically(self) -> None:
        vid_id = uuid4()
        engine = EventEngine()

        t1 = create_mock_track(video_id=vid_id, start_time=5.0, end_time=6.0)
        t2 = create_mock_track(video_id=vid_id, start_time=1.0, end_time=2.0)
        t3 = create_mock_track(video_id=vid_id, start_time=3.0, end_time=4.0)

        # Feed out of order
        timeline = engine.process_multimodal(
            video_id=vid_id,
            perception_result=PerceptionResult(video_id=vid_id, tracks=[t1, t2, t3]),
        )

        timestamps = [e.start_timestamp_seconds for e in timeline.events]
        assert timestamps == sorted(timestamps)

    def test_boundary_jitter_suppression(self) -> None:
        # A track oscillating right at the boundary of a zone
        vid_id = uuid4()
        zone = create_mock_zone("gate", "Gate", x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0)
        engine = EventEngine(
            zones=[zone],
            config=EventEngineConfig(
                spatial_debounce_interval_seconds=0.5,
                spatial_boundary_tolerance_px=5.0,
            ),
        )

        trk = create_mock_track(video_id=vid_id, start_time=0.0, end_time=1.0)
        # Centroid jitters across 100.0 (99.5, 100.5, 99.8, 100.2...) rapidly
        pts = []
        for i in range(10):
            x = 99.5 if i % 2 == 0 else 100.5
            p = create_mock_trajectory(
                trk.track_id, start_x=x, start_y=150.0, end_x=x, end_y=150.0, frames=1
            )[0]
            pts.append(p.model_copy(update={"frame_number": i, "timestamp_seconds": i * 0.04}))

        timeline = engine.process_multimodal(
            video_id=vid_id,
            perception_result=PerceptionResult(
                video_id=vid_id,
                tracks=[trk],
                trajectories={trk.track_id: pts},
            ),
        )

        # Debounce and boundary tolerance should suppress multiple rapid enter/exit chatter
        entry_events = timeline.filter_by_type(EventType.OBJECT_ENTERED_ZONE)
        exit_events = timeline.filter_by_type(EventType.OBJECT_EXITED_ZONE)
        assert len(entry_events) <= 1
        assert len(exit_events) <= 1


class TestEventEnginePerformanceBenchmark:
    """Benchmark event generation throughput and latency."""

    def test_event_throughput_benchmark(self) -> None:
        vid_id = uuid4()
        zone = create_mock_zone()
        engine = EventEngine(zones=[zone])

        # Create 20 tracks, 10 text observations, 5 transcript segments
        tracks = []
        trajectories = {}
        for i in range(20):
            t = create_mock_track(video_id=vid_id, start_time=i * 0.5, end_time=(i + 2) * 0.5)
            tracks.append(t)
            trajectories[t.track_id] = create_mock_trajectory(t.track_id, frames=25)

        ocr_obs = [
            create_mock_text_observation(
                video_id=vid_id,
                text=f"PLATE {i}",
                start_time=i * 0.8,
                end_time=(i + 1) * 0.8,
            )
            for i in range(10)
        ]

        audio_segs = [
            create_mock_transcript_segment(
                video_id=vid_id,
                text=f"Announcement {i}",
                start_time=i * 1.5,
                end_time=(i + 1) * 1.5,
            )
            for i in range(5)
        ]

        t0 = time.perf_counter()
        timeline = engine.process_multimodal(
            video_id=vid_id,
            perception_result=PerceptionResult(
                video_id=vid_id, tracks=tracks, trajectories=trajectories
            ),
            ocr_result=OCRPipelineResult(
                video_id=vid_id,
                raw_observations=[],
                fused_observations=ocr_obs,
                evidence_records=[],
                frames_evaluated=500,
                frames_processed=100,
                total_latency_ms=10.0,
            ),
            audio_result=AudioPipelineResult(
                video_id=vid_id,
                audio_metadata=None,
                raw_segments=audio_segs,
                fused_segments=audio_segs,
                evidence=[],
                duration_seconds=10.0,
                language="en",
                processing_time_seconds=0.1,
            ),
        )
        elapsed_s = time.perf_counter() - t0

        total_events = len(timeline)
        throughput = total_events / max(1e-6, elapsed_s)

        # Deterministic CPU event engine should easily process thousands of events/sec
        assert total_events > 50
        assert elapsed_s < 0.25  # Sub-250ms for complex multimodal batch
        assert throughput > 200.0  # > 200 events/second
