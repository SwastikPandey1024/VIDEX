"""Unit tests for VIDEX typed read-only tools and ToolRegistry."""

from __future__ import annotations

from videx.agent.mock import create_mock_context
from videx.agent.registry import ToolRegistry
from videx.agent.tools import register_investigation_tools
from videx.agent.types import ToolName


def test_tool_registry_registration_and_metadata() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    registered_names = registry.list_tools()
    assert len(registered_names) == 16

    for name in registered_names:
        tool_def = registry.get_definition(name)
        assert tool_def is not None
        assert tool_def.read_only is True
        assert tool_def.allowed_in_agent is True
        assert tool_def.provenance_required is True
        assert tool_def.name == name
        assert bool(tool_def.description)
        assert isinstance(tool_def.input_schema, dict)
        assert isinstance(tool_def.output_schema, dict)


def test_search_events_tool() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # Search without filter
    res = registry.execute(ToolName.SEARCH_EVENTS.value, {"video_id": ctx.video_id})
    assert res.success
    assert len(res.data["events"]) >= 3
    assert len(res.provenance) >= 3

    # Search with time window
    res_time = registry.execute(
        ToolName.SEARCH_EVENTS.value,
        {"video_id": ctx.video_id, "time_start": 2.0, "time_end": 4.0},
    )
    assert res_time.success
    for ev in res_time.data["events"]:
        assert ev["time_end"] >= 2.0
        assert ev["time_start"] <= 4.0


def test_get_event_tool() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # Existing event
    res = registry.execute(ToolName.GET_EVENT.value, {"event_id": "ev-1", "video_id": ctx.video_id})
    assert res.success
    assert res.data["event"]["event_id"] == "ev-1"
    assert len(res.provenance) == 1
    assert res.provenance[0]["source_id"] == str(ctx.events["ev-1"].event_id)

    # Missing event
    res_missing = registry.execute(
        ToolName.GET_EVENT.value,
        {"event_id": "ev-nonexistent", "video_id": ctx.video_id},
    )
    assert not res_missing.success


def test_get_track_and_get_object_tools() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # get_track
    res_track = registry.execute(
        ToolName.GET_TRACK.value,
        {"track_id": "7", "video_id": ctx.video_id},
    )
    assert res_track.success
    assert res_track.data["track"]["primary_class"] == "person"
    assert res_track.provenance[0]["source_id"] == str(ctx.tracks["7"].track_id)

    # get_object
    res_obj = registry.execute(
        ToolName.GET_OBJECT.value,
        {"class_name": "person", "video_id": ctx.video_id},
    )
    assert res_obj.success
    assert len(res_obj.data["detections"]) >= 2


def test_ocr_and_transcript_tools() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # get_ocr
    res_ocr = registry.execute(ToolName.GET_OCR.value, {"video_id": ctx.video_id})
    assert res_ocr.success
    assert len(res_ocr.data["ocr_observations"]) >= 1
    assert res_ocr.data["ocr_observations"][0]["text"] == "STOP"

    # get_transcript
    res_asr = registry.execute(ToolName.GET_TRANSCRIPT.value, {"video_id": ctx.video_id})
    assert res_asr.success
    assert len(res_asr.data["transcript_segments"]) >= 1
    assert "Driver waiting" in res_asr.data["transcript_segments"][0]["text"]


def test_graph_and_neighborhood_tools() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # query_graph
    res_q = registry.execute(
        ToolName.QUERY_GRAPH.value,
        {"video_id": ctx.video_id, "node_types": ["track"]},
    )
    assert res_q.success
    assert res_q.data["count"] >= 2

    # find_temporal_neighbors
    res_temp = registry.execute(
        ToolName.FIND_TEMPORAL_NEIGHBORS.value,
        {"event_id": "ev-1", "window_seconds": 3.0},
    )
    assert res_temp.success
    assert isinstance(res_temp.data["temporal_neighbors"], list)


def test_seek_and_compare_frames_tools() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # seek_video
    res_seek = registry.execute(
        ToolName.SEEK_VIDEO.value,
        {"video_id": ctx.video_id, "timestamp_seconds": 1.9},
    )
    assert res_seek.success
    assert res_seek.data["aligned_pts"] == 2.0

    # compare_frames
    # Find frame ids in ctx
    frame_ids = list(ctx.frames.keys())
    assert len(frame_ids) >= 2
    res_comp = registry.execute(
        ToolName.COMPARE_FRAMES.value,
        {"frame_id_1": frame_ids[0], "frame_id_2": frame_ids[1]},
    )
    assert res_comp.success
    assert "pts_gap_seconds" in res_comp.data


def test_reason_semantic_tool() -> None:
    ctx = create_mock_context()
    registry = ToolRegistry()
    register_investigation_tools(registry, ctx)

    # With semantic runner available
    res = registry.execute(
        ToolName.REASON_SEMANTIC.value,
        {"query": "Did the person interact with the car?"},
    )
    assert res.success
    assert "approached" in res.data["answer"]
    assert res.provenance[0]["epistemic_status"] == "VLM inferred"

    # Without runner available
    ctx_no_runner = create_mock_context()
    ctx_no_runner.semantic_runner = None
    reg_no_runner = ToolRegistry()
    register_investigation_tools(reg_no_runner, ctx_no_runner)

    res_no = reg_no_runner.execute(
        ToolName.REASON_SEMANTIC.value,
        {"query": "Did the person interact with the car?"},
    )
    assert not res_no.success
    assert res_no.error is not None
    assert "unavailable" in res_no.error.lower()
