"""VIDEX Phase 5.0 — Temporal Event Intelligence Engine Smoke Test.

Demonstrates deterministic event generation across all multimodal families:
1. Object lifecycle events (OBJECT_APPEARED, OBJECT_PRESENT, OBJECT_DISAPPEARED)
2. Movement events (OBJECT_STARTED_MOVING, OBJECT_STOPPED_MOVING, OBJECT_CHANGED_DIRECTION)
3. Spatial zone events (OBJECT_ENTERED_ZONE, OBJECT_EXITED_ZONE)
4. OCR text events (TEXT_APPEARED, TEXT_CHANGED, TEXT_DISAPPEARED)
5. Audio speech events (SPEECH_STARTED, SPEECH_DETECTED, SPEECH_ENDED)
6. Temporal relationships & Cross-modal compound events
7. 6-part grounded evidence explainability chain
8. EventTimeline chronological ordering & performance measurement
"""

from __future__ import annotations

import sys
import time
from uuid import uuid4

from videx.audio.pipeline import AudioPipelineResult
from videx.domain.schemas import BoundingBox
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
from videx.events.types import EventType
from videx.ocr.pipeline import OCRPipelineResult
from videx.perception.pipeline import PerceptionResult


def print_section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def main() -> int:
    print("\nVIDEX Phase 5.0 — Temporal Event Intelligence Engine Smoke Test")
    print("Testing Event Domain Models, Detectors, Evidence Linking & Timeline...")

    video_id = uuid4()
    t_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────────
    # [1/7] Prepare Multimodal Inputs & Spatial Zones
    # ─────────────────────────────────────────────────────────────
    print_section("[1/7] Initializing Spatial Zones and Multimodal Inputs")

    zone_gate = create_mock_zone(
        zone_id="zone_checkpoint",
        zone_name="Checkpoint Gate",
        x_min=100.0,
        y_min=100.0,
        x_max=220.0,
        y_max=220.0,
    )
    print(f"  [INFO] Configured SpatialZone '{zone_gate.zone_name}' (ID: {zone_gate.zone_id})")

    # Track 1: Car starts stationary, accelerates into checkpoint, turns, and stops
    trk_car = create_mock_track(
        video_id=video_id,
        class_name="car",
        start_frame=0,
        end_frame=40,
        start_time=0.0,
        end_time=1.6,
        confidence=0.96,
    )

    # 40 trajectory points:
    # 0.0s - 0.2s (frames 0-5): stationary at (50, 150)
    # 0.24s - 0.8s (frames 6-20): moving east (crosses zone at x=100 around frame 12, 0.48s)
    # 0.84s - 1.2s (frames 21-30): turns south (crosses exit at y=220 around frame 28)
    # 1.24s - 1.6s (frames 31-40): stationary at (180, 240)
    traj_points = []
    fps = 25.0

    for i in range(41):
        ts = i / fps
        if i <= 5:
            x, y = 50.0, 150.0
        elif i <= 20:
            prog = (i - 5) / 15.0
            x = 50.0 + prog * 130.0  # reaches 180
            y = 150.0
        elif i <= 30:
            prog = (i - 20) / 10.0
            x = 180.0
            y = 150.0 + prog * 90.0  # reaches 240
        else:
            x, y = 180.0, 240.0

        p = create_mock_trajectory(
            trk_car.track_id, start_x=x, start_y=y, end_x=x, end_y=y, frames=1
        )[0]
        traj_points.append(p.model_copy(update={"frame_number": i, "timestamp_seconds": ts}))

    perception_res = PerceptionResult(
        video_id=video_id,
        tracks=[trk_car],
        trajectories={trk_car.track_id: traj_points},
    )
    print(
        f"  [PASS] Perception input: Track '{trk_car.class_name}' "
        f"with {len(traj_points)} trajectory points"
    )

    # OCR: Number plate changes / becomes clear at checkpoint
    ocr_plate1 = create_mock_text_observation(
        video_id=video_id,
        text="MH 12 AB 1234",
        start_frame=12,
        end_frame=22,
        start_time=0.48,
        end_time=0.88,
        bbox=BoundingBox(x=120.0, y=140.0, width=70.0, height=25.0),
    )
    ocr_plate2 = create_mock_text_observation(
        video_id=video_id,
        text="MH 12 AB 1235",
        start_frame=23,
        end_frame=32,
        start_time=0.92,
        end_time=1.28,
        bbox=BoundingBox(x=122.0, y=142.0, width=70.0, height=25.0),
    )
    ocr_res = OCRPipelineResult(
        video_id=video_id,
        raw_observations=[],
        fused_observations=[ocr_plate1, ocr_plate2],
        evidence_records=[],
        frames_evaluated=41,
        frames_processed=20,
        total_latency_ms=15.0,
    )
    print(
        f"  [PASS] OCR input: 2 fused TextObservations "
        f"('{ocr_plate1.text}', '{ocr_plate2.text}')"
    )

    # Audio: Guard announcement
    speech_seg = create_mock_transcript_segment(
        video_id=video_id,
        text="Vehicle approaching security barrier",
        start_time=0.40,
        end_time=1.40,
        language="en",
    )
    audio_res = AudioPipelineResult(
        video_id=video_id,
        audio_metadata=None,
        raw_segments=[speech_seg],
        fused_segments=[speech_seg],
        evidence=[],
        duration_seconds=1.6,
        language="en",
        processing_time_seconds=0.08,
    )
    print(
        f"  [PASS] Audio input: TranscriptSegment ('{speech_seg.raw_text}') "
        f"at [{speech_seg.start_timestamp_seconds:.2f}s - "
        f"{speech_seg.end_timestamp_seconds:.2f}s]"
    )

    # ─────────────────────────────────────────────────────────────
    # [2/7] Execute EventEngine Multimodal Processing
    # ─────────────────────────────────────────────────────────────
    print_section("[2/7] Running Temporal Event Intelligence Engine")
    config = EventEngineConfig(
        lifecycle_confirmation_threshold=2,
        movement_velocity_threshold_px_s=10.0,
        direction_change_degrees_threshold=45.0,
        spatial_boundary_tolerance_px=5.0,
        spatial_debounce_interval_seconds=0.3,
    )
    engine = EventEngine(config=config, zones=[zone_gate])

    t_engine_start = time.perf_counter()
    timeline: EventTimeline = engine.process_multimodal(
        video_id=video_id,
        perception_result=perception_res,
        ocr_result=ocr_res,
        audio_result=audio_res,
    )
    engine_time = time.perf_counter() - t_engine_start

    print(f"  [PASS] EventEngine execution completed in {engine_time * 1000.0:.2f} ms")
    print(f"  [INFO] Total events generated: {len(timeline)}")
    print(f"  [INFO] Total evidence records: {len(timeline.evidence_records)}")

    # ─────────────────────────────────────────────────────────────
    # [3/7] Verify Event Families Detection
    # ─────────────────────────────────────────────────────────────
    print_section("[3/7] Verifying Event Families Detection")
    event_types = {e.event_type for e in timeline.events}

    # 1. Lifecycle
    assert EventType.OBJECT_APPEARED in event_types, "OBJECT_APPEARED missing"
    assert EventType.OBJECT_PRESENT in event_types, "OBJECT_PRESENT missing"
    assert EventType.OBJECT_DISAPPEARED in event_types, "OBJECT_DISAPPEARED missing"
    print("  [PASS] Lifecycle events: OBJECT_APPEARED, OBJECT_PRESENT, OBJECT_DISAPPEARED")

    # 2. Movement
    assert EventType.OBJECT_STARTED_MOVING in event_types, "OBJECT_STARTED_MOVING missing"
    assert EventType.OBJECT_STOPPED_MOVING in event_types, "OBJECT_STOPPED_MOVING missing"
    assert EventType.OBJECT_CHANGED_DIRECTION in event_types, "OBJECT_CHANGED_DIRECTION missing"
    print("  [PASS] Movement events: OBJECT_STARTED_MOVING, "
          "OBJECT_STOPPED_MOVING, OBJECT_CHANGED_DIRECTION")

    # 3. Spatial
    assert EventType.OBJECT_ENTERED_ZONE in event_types, "OBJECT_ENTERED_ZONE missing"
    assert EventType.OBJECT_EXITED_ZONE in event_types, "OBJECT_EXITED_ZONE missing"
    print("  [PASS] Spatial events: OBJECT_ENTERED_ZONE, OBJECT_EXITED_ZONE")

    # 4. OCR
    assert EventType.TEXT_APPEARED in event_types, "TEXT_APPEARED missing"
    assert EventType.TEXT_CHANGED in event_types, "TEXT_CHANGED missing"
    assert EventType.TEXT_DISAPPEARED in event_types, "TEXT_DISAPPEARED missing"
    print("  [PASS] OCR events: TEXT_APPEARED, TEXT_CHANGED, TEXT_DISAPPEARED")

    # 5. Audio
    assert EventType.SPEECH_STARTED in event_types, "SPEECH_STARTED missing"
    assert EventType.SPEECH_DETECTED in event_types, "SPEECH_DETECTED missing"
    assert EventType.SPEECH_ENDED in event_types, "SPEECH_ENDED missing"
    print("  [PASS] Audio events: SPEECH_STARTED, SPEECH_DETECTED, SPEECH_ENDED")

    # ─────────────────────────────────────────────────────────────
    # [4/7] Verify Temporal Ordering and Interval Queries
    # ─────────────────────────────────────────────────────────────
    print_section("[4/7] Verifying Chronological Ordering & Interval Queries")
    for i in range(len(timeline.events) - 1):
        e_curr = timeline.events[i]
        e_next = timeline.events[i + 1]
        assert e_curr.start_timestamp_seconds <= e_next.start_timestamp_seconds, (
            f"Event ordering violation at idx {i}: "
            f"{e_curr.start_timestamp_seconds} > {e_next.start_timestamp_seconds}"
        )
    print("  [PASS] Strict chronological sorting verified across entire timeline")

    # Interval query
    mid_events = timeline.events_between(0.4, 0.9)
    print(
        "  [PASS] Interval query events_between(0.4s, 0.9s) returned "
        f"{len(mid_events)} active events"
    )

    # ─────────────────────────────────────────────────────────────
    # [5/7] Verify Cross-Modal Compound Rules
    # ─────────────────────────────────────────────────────────────
    print_section("[5/7] Evaluating Cross-Modal Compound Rules")
    rule_engine = CrossModalRuleEngine(temporal_tolerance_seconds=1.0)
    compound_events = rule_engine.detect_speech_during_zone_presence(
        timeline.events, video_id=video_id
    )
    assert len(compound_events) >= 1, "Compound speech-in-zone event not detected"
    comp = compound_events[0]
    print(f"  [PASS] Compound Event Detected: {comp.description}")
    print(f"         Participants: {[p.label for p in comp.participants]}")
    print(f"         Supporting Evidence Count: {len(comp.event_evidence)}")

    # ─────────────────────────────────────────────────────────────
    # [6/7] Demonstrate 6-Part Evidence Explainability Chain
    # ─────────────────────────────────────────────────────────────
    print_section("[6/7] Demonstrating 6-Part Evidence Explainability Chain")
    entry_ev = next(e for e in timeline.events if e.event_type == EventType.OBJECT_ENTERED_ZONE)
    explanation = timeline.explain_event(entry_ev.event_id)

    print("  Structured Audit Trail:")
    print(f"    1. WHAT:                {explanation['what']['event_type']} - "
          f"'{explanation['what']['description']}'")
    print(f"    2. WHEN:                {explanation['when']['summary']}")
    print(f"    3. WHICH ENTITIES:      {explanation['which_entities']}")
    print(
        f"    4. WHAT EVIDENCE:       {len(explanation['supporting_evidence']['records'])} "
        f"linked evidence records (IDs: {explanation['supporting_evidence']['evidence_ids']})"
    )
    frames_str = str(explanation['temporal_anchors']['supporting_frames'])
    pts_val = explanation['temporal_anchors']['start_seconds']
    print(f"    5. WHICH ANCHORS:       Frames: {frames_str} at PTS {pts_val:.2f}s")
    print(
        f"    6. WHICH SUBSYSTEM:     Module: '{explanation['source_subsystem']['source_module']}' "
        f"with attributes: {explanation['source_subsystem']['attributes']}"
    )
    print("  [PASS] All 6 explainability questions successfully satisfied")

    # ─────────────────────────────────────────────────────────────
    # [7/7] Performance & Telemetry Summary
    # ─────────────────────────────────────────────────────────────
    print_section("[7/7] Performance & Telemetry Report")
    summary = timeline.summary()
    total_time = time.perf_counter() - t_start
    events_per_sec = len(timeline) / max(1e-6, engine_time)

    print(f"  Timeline Duration:       {summary['time_span_seconds']:.2f} s")
    print(f"  Total Events Emitted:    {summary['total_events']}")
    print(f"  Total Canonical Evidence:{summary['total_evidence_records']}")
    print(f"  Engine Latency:          {engine_time * 1000.0:.2f} ms")
    print(f"  Engine Throughput:       {events_per_sec:.0f} events/second")
    print(f"  Total Smoke Test Time:   {total_time * 1000.0:.2f} ms")
    print("  Event Distribution:")
    for ev_k, ev_v in summary["event_type_distribution"].items():
        print(f"    • {ev_k:<28}: {ev_v}")

    print_section("VIDEX Phase 5.0 Event Intelligence Smoke Test: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
