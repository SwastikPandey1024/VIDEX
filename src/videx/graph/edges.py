"""GraphEdge definitions and typed factory constructors."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from videx.graph.provenance import EdgeProvenance
from videx.graph.types import DerivationType, GraphEdgeType


def make_edge_id(
    source_node_id: str,
    target_node_id: str,
    relationship: GraphEdgeType,
    qualifier: str | None = None,
) -> str:
    """Format canonical deterministic edge identifier."""
    base = f"{relationship.value}:{source_node_id}->{target_node_id}"
    if qualifier:
        return f"{base}#{qualifier}"
    return base


class GraphEdge(BaseModel):
    """A first-class directed relationship in the VIDEX Evidence Graph."""

    model_config = ConfigDict(frozen=True)

    edge_id: str = Field(..., description="Canonical deterministic edge identifier")
    source_node_id: str = Field(..., description="Tail node ID")
    target_node_id: str = Field(..., description="Head node ID")
    relationship: GraphEdgeType = Field(..., description="Categorization of relationship")
    provenance: EdgeProvenance = Field(..., description="Audit derivation and confidence metadata")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Relationship attributes (weights, metrics, roles)",
    )


def create_edge(
    source_node_id: str,
    target_node_id: str,
    relationship: GraphEdgeType,
    derivation: DerivationType = DerivationType.DETERMINISTIC,
    confidence: float = 1.0,
    reason: str = "",
    threshold: float | None = None,
    actual_value: float | None = None,
    qualifier: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> GraphEdge:
    """Convenience factory to create a fully validated GraphEdge."""
    edge_id = make_edge_id(source_node_id, target_node_id, relationship, qualifier)
    prov = EdgeProvenance(
        derivation=derivation,
        confidence=confidence,
        reason=reason,
        threshold=threshold,
        actual_value=actual_value,
        metadata=dict(attributes or {}),
    )
    return GraphEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relationship=relationship,
        provenance=prov,
        attributes=dict(attributes or {}),
    )
