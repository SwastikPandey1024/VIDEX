"""Query and filtering schemas for the VIDEX Evidence Graph."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from videx.graph.edges import GraphEdge
from videx.graph.nodes import GraphNode
from videx.graph.types import DerivationType, GraphEdgeType, GraphNodeType


class NodeFilter(BaseModel):
    """Criteria for filtering graph nodes."""

    model_config = ConfigDict(frozen=True)

    video_id: str | UUID | None = None
    node_types: tuple[GraphNodeType, ...] | None = None
    source_ids: tuple[str, ...] | None = None
    time_start: float | None = None
    time_end: float | None = None
    label_contains: str | None = None
    limit: int | None = None


class EdgeFilter(BaseModel):
    """Criteria for filtering graph edges."""

    model_config = ConfigDict(frozen=True)

    relationships: tuple[GraphEdgeType, ...] | None = None
    source_node_ids: tuple[str, ...] | None = None
    target_node_ids: tuple[str, ...] | None = None
    derivations: tuple[DerivationType, ...] | None = None
    min_confidence: float | None = None
    limit: int | None = None


class GraphPath(BaseModel):
    """Ordered sequence of nodes and edges representing a path traversal."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[GraphNode, ...] = Field(default_factory=tuple)
    edges: tuple[GraphEdge, ...] = Field(default_factory=tuple)

    @property
    def length(self) -> int:
        return len(self.edges)


class GraphQueryResult(BaseModel):
    """Container for graph query responses."""

    model_config = ConfigDict(frozen=True)

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    total_nodes_scanned: int = 0
    total_edges_scanned: int = 0


class GraphIntegrityReport(BaseModel):
    """Audit report from GraphIntegrityValidator."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    total_nodes: int
    total_edges: int
    total_videos: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
