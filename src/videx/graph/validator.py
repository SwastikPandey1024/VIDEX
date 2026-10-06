"""Graph integrity and spatiotemporal constraint validator."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from videx.graph.schemas import EdgeFilter, GraphIntegrityReport, NodeFilter
from videx.graph.store import GraphStore


class GraphIntegrityValidator:
    """Enforces spatiotemporal consistency, cross-video isolation, and provenance completeness."""

    def validate(
        self,
        store: GraphStore,
        video_id: str | UUID | None = None,
    ) -> GraphIntegrityReport:
        """Audit the GraphStore against all VIDEX epistemological integrity constraints."""
        errors: list[str] = []
        warnings: list[str] = []

        target_vid = str(video_id) if video_id is not None else None
        nodes = store.query_nodes(NodeFilter(video_id=target_vid))
        edges = store.query_edges(EdgeFilter())
        video_ids = {n.video_id for n in nodes}

        # 1. Node Integrity & Provenance Completeness
        for node in nodes:
            prov = node.provenance
            if not prov.source_id:
                errors.append(f"Node {node.node_id} has empty source_id")
            if not prov.source_type:
                errors.append(f"Node {node.node_id} has empty source_type")
            if not prov.video_id:
                errors.append(f"Node {node.node_id} has empty video_id")

            # Timestamp validity
            start = prov.timestamp_start
            end = prov.timestamp_end
            if start is not None and start < 0.0:
                errors.append(f"Node {node.node_id} has negative timestamp_start: {start}")
            if end is not None and end < 0.0:
                errors.append(f"Node {node.node_id} has negative timestamp_end: {end}")
            if start is not None and end is not None and start > end:
                errors.append(
                    f"Node {node.node_id} has inverted interval: {start:.3f}s > {end:.3f}s"
                )

        # 2. Edge Integrity & Endpoint Existence
        for edge in edges:
            src = store.get_node(edge.source_node_id)
            tgt = store.get_node(edge.target_node_id)

            if src is None:
                errors.append(
                    f"Orphan edge {edge.edge_id}: source '{edge.source_node_id}' does not exist"
                )
            if tgt is None:
                errors.append(
                    f"Orphan edge {edge.edge_id}: target '{edge.target_node_id}' does not exist"
                )

            # 3. Cross-Video Isolation
            if src is not None and tgt is not None:
                if src.video_id != tgt.video_id:
                    errors.append(
                        f"Cross-video isolation violation on edge {edge.edge_id}: "
                        f"source video {src.video_id} != target video {tgt.video_id}"
                    )

            # 4. Derivation and confidence validity
            if edge.provenance.confidence < 0.0 or edge.provenance.confidence > 1.0:
                errors.append(
                    f"Edge {edge.edge_id} has invalid confidence: {edge.provenance.confidence}"
                )

        is_valid = len(errors) == 0

        metrics: dict[str, Any] = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "videos_represented": len(video_ids),
            "node_types": {},
            "edge_types": {},
        }
        for n in nodes:
            t = n.node_type.value
            metrics["node_types"][t] = metrics["node_types"].get(t, 0) + 1
        for e in edges:
            r = e.relationship.value
            metrics["edge_types"][r] = metrics["edge_types"].get(r, 0) + 1

        return GraphIntegrityReport(
            is_valid=is_valid,
            total_nodes=len(nodes),
            total_edges=len(edges),
            total_videos=len(video_ids),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )
