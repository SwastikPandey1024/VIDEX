"""Typed, read-only tools for the VIDEX Agentic Investigator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from videx.agent.registry import ToolDefinition, ToolRegistry
from videx.agent.schemas import ToolResult
from videx.agent.types import ToolName
from videx.domain.schemas import (
    Evidence,
    Frame,
    OCRObservation,
    Scene,
    Track,
    TranscriptSegment,
)
from videx.events.schemas import Event
from videx.graph.nodes import GraphNode, make_node_id
from videx.graph.query import GraphQueryService
from videx.graph.schemas import NodeFilter
from videx.graph.types import GraphEdgeType, GraphNodeType

logger = logging.getLogger(__name__)


@dataclass
class AgentToolContext:
    """Read-only data context supplied to agent tools for an investigation."""

    video_id: str
    graph_query_service: GraphQueryService
    evidence_records: dict[str, Evidence] = field(default_factory=dict)
    events: dict[str, Event] = field(default_factory=dict)
    tracks: dict[str, Track] = field(default_factory=dict)
    scenes: dict[str, Scene] = field(default_factory=dict)
    frames: dict[str, Frame] = field(default_factory=dict)
    ocr_observations: dict[str, OCRObservation] = field(default_factory=dict)
    transcript_segments: dict[str, TranscriptSegment] = field(default_factory=dict)
    semantic_runner: Any | None = None  # Optional semantic reasoning callable / router


def _node_provenance(node: GraphNode) -> dict[str, Any]:
    """Extract standard provenance record from a GraphNode."""
    return {
        "source_id": node.source_id,
        "source_type": node.source_type,
        "video_id": node.video_id,
        "timestamp_start": node.timestamp_start,
        "timestamp_end": node.timestamp_end,
        "derivation": node.derivation.value,
        "epistemic_status": node.epistemic_status,
        "confidence": node.confidence,
        "evidence_ids": list(node.evidence_ids),
    }


def register_investigation_tools(registry: ToolRegistry, ctx: AgentToolContext) -> None:
    """Register all authorized read-only investigation tools into the registry."""

    # 1. search_events
    def _search_events(params: dict[str, Any]) -> ToolResult:
        vid = params.get("video_id", ctx.video_id)
        start_sec = float(params.get("start_sec", params.get("time_start", 0.0)))
        end_sec = float(params.get("end_sec", params.get("time_end", 1e9)))
        event_type_str = params.get("event_type")

        # Delegate directly to existing GraphQueryService.events_between
        nodes = ctx.graph_query_service.events_between(vid, start_sec, end_sec)
        if event_type_str:
            nodes = [
                n
                for n in nodes
                if n.attributes.get("event_type") == event_type_str
                or n.attributes.get("semantic_event_type") == event_type_str
            ]

        prov = [_node_provenance(n) for n in nodes]
        items = [
            {
                "event_id": n.source_id,
                "node_type": n.node_type.value,
                "label": n.label,
                "description": n.attributes.get("description", n.label),
                "start_pts": n.timestamp_start,
                "end_pts": n.timestamp_end,
                "time_start": n.timestamp_start,
                "time_end": n.timestamp_end,
                "attributes": n.attributes,
                "epistemic_status": n.epistemic_status,
                "confidence": n.confidence,
                "evidence_ids": list(n.evidence_ids),
                "participant_ids": n.attributes.get("participant_ids", []),
            }
            for n in nodes
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.SEARCH_EVENTS.value,
            success=True,
            data={"events": items, "count": len(items)},
            provenance=prov,
        )

    registry.register(
        ToolDefinition(
            name=ToolName.SEARCH_EVENTS.value,
            description="Search chronological events in a video within an optional PTS window.",
            input_schema={
                "video_id": "string",
                "start_sec": "float",
                "end_sec": "float",
                "event_type": "string (optional)",
            },
            output_schema={"events": "list[dict]", "count": "int"},
        ),
        _search_events,
    )

    # 2. get_event
    def _get_event(params: dict[str, Any]) -> ToolResult:
        ev_id = str(params["event_id"])
        nid = make_node_id(GraphNodeType.EVENT, ev_id)
        node = ctx.graph_query_service.get_node(nid)
        if node is None:
            # Check SEMANTIC_EVENT
            sem_nid = make_node_id(GraphNodeType.SEMANTIC_EVENT, ev_id)
            node = ctx.graph_query_service.get_node(sem_nid)

        if node is None and ev_id in ctx.events:
            ev_rec = ctx.events[ev_id]
            node = ctx.graph_query_service.get_node(
                make_node_id(GraphNodeType.EVENT, str(ev_rec.event_id))
            )

        if node is None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_EVENT.value,
                success=False,
                error=f"Event '{ev_id}' not found",
            )

        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_EVENT.value,
            success=True,
            data={
                "event": {
                    "event_id": ev_id,
                    "canonical_event_id": node.source_id,
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "label": node.label,
                    "description": node.attributes.get("description", node.label),
                    "start_pts": node.timestamp_start,
                    "end_pts": node.timestamp_end,
                    "time_start": node.timestamp_start,
                    "time_end": node.timestamp_end,
                    "attributes": node.attributes,
                    "epistemic_status": node.epistemic_status,
                    "confidence": node.confidence,
                    "evidence_ids": list(node.evidence_ids),
                    "participant_ids": node.attributes.get("participant_ids", []),
                }
            },
            provenance=[_node_provenance(node)],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_EVENT.value,
            description="Fetch a targeted Event or SemanticEvent by UUID with full provenance.",
            input_schema={"event_id": "string"},
            output_schema={"event": "dict"},
        ),
        _get_event,
    )

    # 3. get_track
    def _get_track(params: dict[str, Any]) -> ToolResult:
        track_id = str(params["track_id"])
        nid = make_node_id(GraphNodeType.TRACK, track_id)
        node = ctx.graph_query_service.get_node(nid)
        if node is None and track_id in ctx.tracks:
            trk_rec = ctx.tracks[track_id]
            node = ctx.graph_query_service.get_node(
                make_node_id(GraphNodeType.TRACK, str(trk_rec.track_id))
            )

        if node is None:
            trk = ctx.tracks.get(track_id)
            if trk:
                return ToolResult(
                    call_id="",
                    tool_name=ToolName.GET_TRACK.value,
                    success=True,
                    data={"track": trk.model_dump()},
                    provenance=[{
                        "source_id": str(trk.track_id),
                        "source_type": "Track",
                        "video_id": str(trk.video_id),
                        "confidence": trk.confidence,
                        "epistemic_status": "deterministically observed",
                    }],
                )
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_TRACK.value,
                success=False,
                error=f"Track '{track_id}' not found",
            )

        primary_class = (
            node.attributes.get("primary_class")
            or node.attributes.get("class_name")
            or node.label
        )
        t_start = node.timestamp_start or 0.0
        t_end = node.timestamp_end or t_start
        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_TRACK.value,
            success=True,
            data={
                "track": {
                    "track_id": track_id,
                    "canonical_track_id": node.source_id,
                    "node_id": node.node_id,
                    "primary_class": primary_class,
                    "attributes": node.attributes,
                    "epistemic_status": node.epistemic_status,
                    "evidence_ids": list(node.evidence_ids),
                    "start_pts": t_start,
                    "end_pts": t_end,
                    "duration_seconds": round(t_end - t_start, 2),
                }
            },
            provenance=[_node_provenance(node)],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_TRACK.value,
            description="Fetch tracked object trajectory, class name, and kinematics by track ID.",
            input_schema={"track_id": "string"},
            output_schema={"track": "dict"},
        ),
        _get_track,
    )

    # 4. get_object
    def _get_object(params: dict[str, Any]) -> ToolResult:
        if "object_id" in params:
            obj_id = str(params["object_id"])
            nid = make_node_id(GraphNodeType.OBJECT, obj_id)
            node = ctx.graph_query_service.get_node(nid)
            if node is None:
                return _get_track({"track_id": obj_id})
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_OBJECT.value,
                success=True,
                data={
                    "object": {
                        "object_id": node.source_id,
                        "label": node.label,
                        "attributes": node.attributes,
                        "epistemic_status": node.epistemic_status,
                    }
                },
                provenance=[_node_provenance(node)],
            )

        # Class-based query over detections
        class_name = str(params.get("class_name", ""))
        vid = params.get("video_id", ctx.video_id)
        det_nodes = ctx.graph_query_service.query_nodes(
            NodeFilter(video_id=vid, node_types=(GraphNodeType.DETECTION,))
        )
        if class_name:
            det_nodes = [
                n
                for n in det_nodes
                if class_name.lower() in str(n.attributes.get("class_name", "")).lower()
                or class_name.lower() in n.label.lower()
            ]
        items = [
            {
                "detection_id": n.source_id,
                "label": n.label,
                "confidence": n.confidence,
                "timestamp_pts": n.timestamp_start,
                "bbox": n.attributes.get("bbox"),
            }
            for n in det_nodes
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_OBJECT.value,
            success=True,
            data={"detections": items, "count": len(items)},
            provenance=[_node_provenance(n) for n in det_nodes],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_OBJECT.value,
            description="Fetch domain-level object abstraction or persistent tracked entity.",
            input_schema={"object_id": "string (optional)", "class_name": "string (optional)"},
            output_schema={"object": "dict (optional)", "detections": "list[dict] (optional)"},
        ),
        _get_object,
    )

    # 5. get_frames
    def _get_frames(params: dict[str, Any]) -> ToolResult:
        raw_fids = params.get("frame_ids", [])
        results: list[dict[str, Any]] = []
        provs: list[dict[str, Any]] = []
        for fid in raw_fids:
            str_fid = str(fid)
            nid = make_node_id(GraphNodeType.FRAME, str_fid)
            node = ctx.graph_query_service.get_node(nid)
            if node:
                results.append({
                    "frame_id": node.source_id,
                    "pts_seconds": node.timestamp_start,
                    "attributes": node.attributes,
                })
                provs.append(_node_provenance(node))
            elif str_fid in ctx.frames:
                f_rec = ctx.frames[str_fid]
                results.append({
                    "frame_id": str(f_rec.frame_id),
                    "frame_number": f_rec.frame_number,
                    "pts_seconds": f_rec.timestamp_seconds,
                    "frame_data_path": f_rec.frame_data_path,
                })
                provs.append({
                    "source_id": str(f_rec.frame_id),
                    "source_type": "Frame",
                    "video_id": str(f_rec.video_id),
                    "timestamp_start": f_rec.timestamp_seconds,
                    "epistemic_status": "deterministically observed",
                })

        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_FRAMES.value,
            success=True,
            data={"frames": results, "count": len(results)},
            provenance=provs,
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_FRAMES.value,
            description="Fetch discrete video frames by frame ID with PTS and storage metadata.",
            input_schema={"frame_ids": "list[string]"},
            output_schema={"frames": "list[dict]"},
        ),
        _get_frames,
    )

    # 6. get_evidence
    def _get_evidence(params: dict[str, Any]) -> ToolResult:
        if "event_id" in params and "evidence_id" not in params:
            event_id = str(params["event_id"])
            if event_id in ctx.events:
                event_id = str(ctx.events[event_id].event_id)
            ev_nodes = ctx.graph_query_service.evidence_for_event(event_id)
            items = [
                {
                    "evidence_id": n.source_id,
                    "attributes": n.attributes,
                    "epistemic_status": n.epistemic_status,
                    "confidence": n.confidence,
                }
                for n in ev_nodes
            ]
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_EVIDENCE.value,
                success=True,
                data={"evidence": items, "count": len(items)},
                provenance=[_node_provenance(n) for n in ev_nodes],
            )

        evid_id = str(params.get("evidence_id", ""))
        nid = make_node_id(GraphNodeType.EVIDENCE, evid_id)
        node = ctx.graph_query_service.get_node(nid)
        if node is not None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_EVIDENCE.value,
                success=True,
                data={
                    "evidence_id": node.source_id,
                    "attributes": node.attributes,
                    "epistemic_status": node.epistemic_status,
                    "confidence": node.confidence,
                },
                provenance=[_node_provenance(node)],
            )

        ev_rec = ctx.evidence_records.get(evid_id)
        if ev_rec is not None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_EVIDENCE.value,
                success=True,
                data=ev_rec.model_dump(),
                provenance=[{
                    "source_id": str(ev_rec.evidence_id),
                    "source_type": "Evidence",
                    "video_id": str(ev_rec.video_id),
                    "timestamp_start": ev_rec.timestamp_seconds,
                    "confidence": ev_rec.confidence,
                    "epistemic_status": "deterministically observed",
                }],
            )

        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_EVIDENCE.value,
            success=False,
            error=f"Evidence '{evid_id}' not found",
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_EVIDENCE.value,
            description="Fetch a canonical Evidence grounding record by evidence UUID or event ID.",
            input_schema={"evidence_id": "string (optional)", "event_id": "string (optional)"},
            output_schema={"evidence": "dict or list[dict]"},
        ),
        _get_evidence,
    )

    # 7. get_ocr
    def _get_ocr(params: dict[str, Any]) -> ToolResult:
        vid = params.get("video_id", ctx.video_id)
        text_query = params.get("text_query", "").lower()
        start_sec = float(params.get("start_sec", 0.0))
        end_sec = float(params.get("end_sec", 1e9))

        nodes = ctx.graph_query_service.query_nodes(
            NodeFilter(
                video_id=vid,
                node_types=(GraphNodeType.OCR_OBSERVATION,),
                time_start=start_sec,
                time_end=end_sec,
            )
        )
        if text_query:
            nodes = [n for n in nodes if text_query in n.attributes.get("text", "").lower()]

        items = [
            {
                "observation_id": n.source_id,
                "text": n.attributes.get("text"),
                "pts_seconds": n.timestamp_start,
                "confidence": n.confidence,
                "bbox": n.attributes.get("bbox"),
                "epistemic_status": n.epistemic_status,
                "evidence_ids": list(n.evidence_ids) or [n.source_id],
            }
            for n in nodes
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_OCR.value,
            success=True,
            data={"observations": items, "ocr_observations": items, "count": len(items)},
            provenance=[_node_provenance(n) for n in nodes],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_OCR.value,
            description=(
                "Query text snippets recognized in visual frames by text query or temporal window."
            ),
            input_schema={
                "video_id": "string",
                "text_query": "string (optional)",
                "start_sec": "float",
                "end_sec": "float",
            },
            output_schema={"observations": "list[dict]"},
        ),
        _get_ocr,
    )

    # 8. get_transcript
    def _get_transcript(params: dict[str, Any]) -> ToolResult:
        vid = params.get("video_id", ctx.video_id)
        start_sec = float(params.get("start_sec", 0.0))
        end_sec = float(params.get("end_sec", 1e9))

        nodes = ctx.graph_query_service.query_nodes(
            NodeFilter(
                video_id=vid,
                node_types=(GraphNodeType.TRANSCRIPT_SEGMENT,),
                time_start=start_sec,
                time_end=end_sec,
            )
        )
        items = [
            {
                "segment_id": n.source_id,
                "text": n.attributes.get("text"),
                "start_pts": n.timestamp_start,
                "end_pts": n.timestamp_end,
                "language": n.attributes.get("language"),
                "speaker": n.attributes.get("speaker_id"),
                "confidence": n.confidence,
                "epistemic_status": n.epistemic_status,
                "evidence_ids": list(n.evidence_ids) or [n.source_id],
            }
            for n in nodes
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_TRANSCRIPT.value,
            success=True,
            data={"segments": items, "transcript_segments": items, "count": len(items)},
            provenance=[_node_provenance(n) for n in nodes],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_TRANSCRIPT.value,
            description="Query transcribed speech audio segments within a temporal window.",
            input_schema={"video_id": "string", "start_sec": "float", "end_sec": "float"},
            output_schema={"segments": "list[dict]"},
        ),
        _get_transcript,
    )

    # 9. get_scene
    def _get_scene(params: dict[str, Any]) -> ToolResult:
        scene_id = str(params["scene_id"])
        nid = make_node_id(GraphNodeType.SCENE, scene_id)
        node = ctx.graph_query_service.get_node(nid)
        if node is not None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_SCENE.value,
                success=True,
                data={
                    "scene_id": node.source_id,
                    "label": node.label,
                    "start_pts": node.timestamp_start,
                    "end_pts": node.timestamp_end,
                    "attributes": node.attributes,
                    "epistemic_status": node.epistemic_status,
                },
                provenance=[_node_provenance(node)],
            )

        sc = ctx.scenes.get(scene_id)
        if sc is not None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.GET_SCENE.value,
                success=True,
                data=sc.model_dump(),
                provenance=[{
                    "source_id": str(sc.scene_id),
                    "source_type": "Scene",
                    "video_id": str(sc.video_id),
                    "timestamp_start": sc.start_timestamp_seconds,
                    "timestamp_end": sc.end_timestamp_seconds,
                    "epistemic_status": "deterministically observed",
                }],
            )

        return ToolResult(
            call_id="",
            tool_name=ToolName.GET_SCENE.value,
            success=False,
            error=f"Scene '{scene_id}' not found",
        )

    registry.register(
        ToolDefinition(
            name=ToolName.GET_SCENE.value,
            description="Fetch a visual scene boundary segment by scene ID.",
            input_schema={"scene_id": "string"},
            output_schema={"scene": "dict"},
        ),
        _get_scene,
    )

    # 10. query_graph
    def _query_graph(params: dict[str, Any]) -> ToolResult:
        vid = params.get("video_id", ctx.video_id)
        raw_types = params.get("node_types")
        types_tuple: tuple[GraphNodeType, ...] | None = None
        if raw_types:
            types_tuple = tuple(GraphNodeType(t) for t in raw_types)

        n_filter = NodeFilter(
            video_id=vid,
            node_types=types_tuple,
            time_start=params.get("time_start"),
            time_end=params.get("time_end"),
            label_contains=params.get("label_contains"),
            limit=params.get("limit", 50),
        )
        res = ctx.graph_query_service.query(node_filter=n_filter)
        return ToolResult(
            call_id="",
            tool_name=ToolName.QUERY_GRAPH.value,
            success=True,
            data={
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "source_id": n.source_id,
                        "node_type": n.node_type.value,
                        "label": n.label,
                        "epistemic_status": n.epistemic_status,
                    }
                    for n in res.nodes
                ],
                "count": len(res.nodes),
                "execution_time_ms": res.execution_time_ms,
            },
            provenance=[_node_provenance(n) for n in res.nodes],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.QUERY_GRAPH.value,
            description="Query graph nodes across entity types, labels, and intervals.",
            input_schema={
                "video_id": "string",
                "node_types": "list[string]",
                "time_start": "float",
                "time_end": "float",
                "label_contains": "string",
            },
            output_schema={"nodes": "list[dict]", "count": "int"},
        ),
        _query_graph,
    )

    # 11. find_related_events
    def _find_related_events(params: dict[str, Any]) -> ToolResult:
        ev_id = str(params["event_id"])
        if ev_id in ctx.events:
            ev_id = str(ctx.events[ev_id].event_id)
        rel_str = params.get("relationship")
        rel = GraphEdgeType(rel_str) if rel_str else None
        related = ctx.graph_query_service.related_events(ev_id, relationship=rel)
        items = [
            {
                "event_id": n.source_id,
                "label": n.label,
                "start_pts": n.timestamp_start,
                "end_pts": n.timestamp_end,
                "epistemic_status": n.epistemic_status,
                "evidence_ids": list(n.evidence_ids),
            }
            for n in related
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.FIND_RELATED_EVENTS.value,
            success=True,
            data={"related_events": items, "count": len(items)},
            provenance=[_node_provenance(n) for n in related],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.FIND_RELATED_EVENTS.value,
            description="Find events connected topologically or causally to a specific event.",
            input_schema={"event_id": "string", "relationship": "string (optional)"},
            output_schema={"related_events": "list[dict]"},
        ),
        _find_related_events,
    )

    # 12. find_temporal_neighbors
    def _find_temporal_neighbors(params: dict[str, Any]) -> ToolResult:
        ev_id = str(params["event_id"])
        if ev_id in ctx.events:
            ev_id = str(ctx.events[ev_id].event_id)
        window_sec = float(params.get("window_sec") or params.get("window_seconds") or 2.0)
        neighbors = ctx.graph_query_service.temporal_neighbors(ev_id, window_sec=window_sec)
        items = [
            {
                "event_id": n.source_id,
                "label": n.label,
                "start_pts": n.timestamp_start,
                "end_pts": n.timestamp_end,
                "epistemic_status": n.epistemic_status,
                "evidence_ids": list(n.evidence_ids),
            }
            for n in neighbors
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.FIND_TEMPORAL_NEIGHBORS.value,
            success=True,
            data={"temporal_neighbors": items, "count": len(items)},
            provenance=[_node_provenance(n) for n in neighbors],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.FIND_TEMPORAL_NEIGHBORS.value,
            description="Retrieve events co-occurring or occurring within window_sec of an event.",
            input_schema={"event_id": "string", "window_sec": "float"},
            output_schema={"temporal_neighbors": "list[dict]"},
        ),
        _find_temporal_neighbors,
    )

    # 13. find_spatial_neighbors
    def _find_spatial_neighbors(params: dict[str, Any]) -> ToolResult:
        node_id = str(params["node_id"])
        neighbors = ctx.graph_query_service.spatial_neighbors(node_id)
        items = [
            {
                "node_id": n.node_id,
                "source_id": n.source_id,
                "label": n.label,
                "epistemic_status": n.epistemic_status,
                "evidence_ids": list(n.evidence_ids),
            }
            for n in neighbors
        ]
        return ToolResult(
            call_id="",
            tool_name=ToolName.FIND_SPATIAL_NEIGHBORS.value,
            success=True,
            data={"spatial_neighbors": items, "count": len(items)},
            provenance=[_node_provenance(n) for n in neighbors],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.FIND_SPATIAL_NEIGHBORS.value,
            description="Retrieve spatial neighbors sharing bbox overlap or zone containment.",
            input_schema={"node_id": "string"},
            output_schema={"spatial_neighbors": "list[dict]"},
        ),
        _find_spatial_neighbors,
    )

    # 14. seek_video
    def _seek_video(params: dict[str, Any]) -> ToolResult:
        vid = params.get("video_id", ctx.video_id)
        target_pts = float(params["timestamp_seconds"])
        # Query nearest frame node
        frames = ctx.graph_query_service.query_nodes(
            NodeFilter(video_id=vid, node_types=(GraphNodeType.FRAME,))
        )
        if not frames:
            return ToolResult(
                call_id="",
                tool_name=ToolName.SEEK_VIDEO.value,
                success=False,
                error="No frames available in graph to seek",
            )

        nearest = min(frames, key=lambda f: abs((f.timestamp_start or 0.0) - target_pts))
        return ToolResult(
            call_id="",
            tool_name=ToolName.SEEK_VIDEO.value,
            success=True,
            data={
                "target_pts": target_pts,
                "aligned_frame_id": nearest.source_id,
                "aligned_pts": nearest.timestamp_start,
                "frame_attributes": nearest.attributes,
                "epistemic_status": nearest.epistemic_status,
            },
            provenance=[_node_provenance(nearest)],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.SEEK_VIDEO.value,
            description="Locate the closest canonical frame for a targeted PTS timestamp.",
            input_schema={"video_id": "string", "timestamp_seconds": "float"},
            output_schema={"aligned_frame_id": "string", "aligned_pts": "float"},
        ),
        _seek_video,
    )

    # 15. compare_frames
    def _compare_frames(params: dict[str, Any]) -> ToolResult:
        fid1 = str(params["frame_id_1"])
        fid2 = str(params["frame_id_2"])
        n1 = ctx.graph_query_service.get_node(make_node_id(GraphNodeType.FRAME, fid1))
        n2 = ctx.graph_query_service.get_node(make_node_id(GraphNodeType.FRAME, fid2))

        if n1 is None or n2 is None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.COMPARE_FRAMES.value,
                success=False,
                error=f"One or both frames not found: '{fid1}', '{fid2}'",
            )

        t1 = n1.timestamp_start or 0.0
        t2 = n2.timestamp_start or 0.0
        gap = abs(t2 - t1)
        f_num1 = n1.attributes.get("frame_number", 0)
        f_num2 = n2.attributes.get("frame_number", 0)
        is_seq = abs(f_num1 - f_num2) == 1

        return ToolResult(
            call_id="",
            tool_name=ToolName.COMPARE_FRAMES.value,
            success=True,
            data={
                "frame_1": {"frame_id": fid1, "pts": t1},
                "frame_2": {"frame_id": fid2, "pts": t2},
                "pts_gap_seconds": round(gap, 3),
                "is_sequential": is_seq,
            },
            provenance=[_node_provenance(n1), _node_provenance(n2)],
        )

    registry.register(
        ToolDefinition(
            name=ToolName.COMPARE_FRAMES.value,
            description="Compare metadata, PTS gap, and sequentiality between two frames.",
            input_schema={"frame_id_1": "string", "frame_id_2": "string"},
            output_schema={"pts_gap_seconds": "float", "is_sequential": "bool"},
        ),
        _compare_frames,
    )

    # 16. reason_semantic
    def _reason_semantic(params: dict[str, Any]) -> ToolResult:
        # Bounded semantic reasoning invoking runner if provided
        query = str(params["query"])
        cand_id = params.get("candidate_event_id")

        if ctx.semantic_runner is None:
            return ToolResult(
                call_id="",
                tool_name=ToolName.REASON_SEMANTIC.value,
                success=False,
                error="VLM semantic reasoning provider is unavailable in current profile.",
            )

        try:
            sem_result = ctx.semantic_runner(query, cand_id)
            return ToolResult(
                call_id="",
                tool_name=ToolName.REASON_SEMANTIC.value,
                success=True,
                data=sem_result,
                provenance=[{
                    "source_id": str(cand_id or "semantic_query"),
                    "source_type": "SemanticReasoning",
                    "video_id": ctx.video_id,
                    "epistemic_status": "VLM inferred",
                    "confidence": sem_result.get("confidence", 0.8),
                    "evidence_ids": sem_result.get("evidence_ids", []),
                }],
            )
        except Exception as exc:
            return ToolResult(
                call_id="",
                tool_name=ToolName.REASON_SEMANTIC.value,
                success=False,
                error=f"Semantic reasoning error: {exc}",
            )

    registry.register(
        ToolDefinition(
            name=ToolName.REASON_SEMANTIC.value,
            description=(
                "Invoke bounded VLM semantic reasoning over selected evidence when deterministic "
                "data is inconclusive."
            ),
            input_schema={"query": "string", "candidate_event_id": "string (optional)"},
            output_schema={"semantic_result": "dict"},
        ),
        _reason_semantic,
    )
