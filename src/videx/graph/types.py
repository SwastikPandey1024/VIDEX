"""Type definitions and enumerations for the VIDEX Evidence Graph."""

from __future__ import annotations

from enum import StrEnum


class GraphNodeType(StrEnum):
    """Categorization of entities represented as nodes in the Evidence Graph."""

    VIDEO = "video"
    SCENE = "scene"
    FRAME = "frame"
    DETECTION = "detection"
    TRACK = "track"
    OBJECT = "object"
    OCR_OBSERVATION = "ocr_observation"
    TRANSCRIPT_SEGMENT = "transcript_segment"
    ZONE = "zone"
    EVENT = "event"
    SEMANTIC_EVENT = "semantic_event"
    EVIDENCE = "evidence"


class GraphEdgeType(StrEnum):
    """Categorization of relationships connecting nodes in the Evidence Graph."""

    # Structural / Containment
    CONTAINS = "CONTAINS"
    BELONGS_TO = "BELONGS_TO"
    REPRESENTS = "REPRESENTS"

    # Multimodal & Grounding
    SUPPORTS = "SUPPORTS"
    SUPPORTED_BY = "SUPPORTED_BY"
    DESCRIBES = "DESCRIBES"
    REFERENCES = "REFERENCES"
    PARTICIPATES_IN = "PARTICIPATES_IN"

    # Spatiotemporal Topology
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    TEMPORALLY_NEAR = "TEMPORALLY_NEAR"
    SPATIALLY_NEAR = "SPATIALLY_NEAR"
    SPATIALLY_OVERLAPS = "SPATIALLY_OVERLAPS"
    OCCURS_IN = "OCCURS_IN"

    # Semantic Reasoning
    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTS_SEMANTIC_EVENT = "SUPPORTS_SEMANTIC_EVENT"


class DerivationType(StrEnum):
    """Provenance and derivation method behind graph elements."""

    DETERMINISTIC = "deterministic"
    STRUCTURAL = "structural"
    INFERRED = "inferred"
    TRANSITIVE = "transitive"
    MANUAL = "manual"


class Direction(StrEnum):
    """Direction for graph traversals."""

    OUTBOUND = "outbound"
    INBOUND = "inbound"
    BOTH = "both"
