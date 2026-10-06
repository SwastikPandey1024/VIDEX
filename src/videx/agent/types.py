"""Type definitions and enumerations for VIDEX Agentic Investigation."""

from __future__ import annotations

from enum import StrEnum


class QuestionCategory(StrEnum):
    """Categorization of user questions to drive deterministic planning."""

    TEMPORAL = "temporal"
    OBJECT = "object"
    TRACK = "track"
    EVENT = "event"
    OCR = "ocr"
    AUDIO = "audio"
    SPATIAL = "spatial"
    RELATIONAL = "relational"
    SEMANTIC = "semantic"
    GENERAL = "general"


class ClaimStatus(StrEnum):
    """Epistemic validity status of an asserted claim."""

    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REJECTED = "rejected"


class InvestigationStatus(StrEnum):
    """Overall status of an investigation execution."""

    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"


class ToolName(StrEnum):
    """Enumeration of registered read-only investigation tools."""

    SEARCH_EVENTS = "search_events"
    GET_EVENT = "get_event"
    GET_TRACK = "get_track"
    GET_OBJECT = "get_object"
    GET_FRAMES = "get_frames"
    GET_EVIDENCE = "get_evidence"
    GET_OCR = "get_ocr"
    GET_TRANSCRIPT = "get_transcript"
    GET_SCENE = "get_scene"
    QUERY_GRAPH = "query_graph"
    FIND_RELATED_EVENTS = "find_related_events"
    FIND_TEMPORAL_NEIGHBORS = "find_temporal_neighbors"
    FIND_SPATIAL_NEIGHBORS = "find_spatial_neighbors"
    SEEK_VIDEO = "seek_video"
    COMPARE_FRAMES = "compare_frames"
    REASON_SEMANTIC = "reason_semantic"
