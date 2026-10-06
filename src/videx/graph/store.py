"""Abstract GraphStore interface and high-performance InMemoryGraphStore implementation."""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol

from videx.graph.edges import GraphEdge
from videx.graph.nodes import GraphNode
from videx.graph.schemas import EdgeFilter, GraphPath, NodeFilter
from videx.graph.types import Direction, GraphEdgeType, GraphNodeType


class GraphStore(Protocol):
    """Protocol defining the core storage and traversal contracts for the Evidence Graph."""

    def upsert_node(self, node: GraphNode) -> None:
        """Insert or update a node."""
        ...

    def upsert_edge(self, edge: GraphEdge) -> None:
        """Insert or update a directed edge."""
        ...

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a node by its deterministic node_id."""
        ...

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Retrieve an edge by its deterministic edge_id."""
        ...

    def neighbors(
        self,
        node_id: str,
        direction: Direction = Direction.BOTH,
        relationship: GraphEdgeType | None = None,
    ) -> list[GraphNode]:
        """Retrieve adjacent nodes connected to node_id."""
        ...

    def edges_for(
        self,
        node_id: str,
        direction: Direction = Direction.BOTH,
        relationship: GraphEdgeType | None = None,
    ) -> list[GraphEdge]:
        """Retrieve edges connected to node_id."""
        ...

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[GraphPath]:
        """Find directed or undirected paths between source and target up to max_depth."""
        ...

    def query_nodes(self, node_filter: NodeFilter) -> list[GraphNode]:
        """Filter nodes matching criteria."""
        ...

    def query_edges(self, edge_filter: EdgeFilter) -> list[GraphEdge]:
        """Filter edges matching criteria."""
        ...

    def count_nodes(self, video_id: str | None = None) -> int:
        """Return total node count, optionally filtered by video_id."""
        ...

    def count_edges(self) -> int:
        """Return total edge count."""
        ...

    def clear(self) -> None:
        """Remove all nodes and edges."""
        ...


class InMemoryGraphStore:
    """Thread-safe, index-backed in-memory implementation of GraphStore.

    Uses adjacency indexing and multi-key secondary indices for O(1) lookups
    and sub-millisecond local neighborhood traversals.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

        # Adjacency indices: node_id -> set of edge_ids
        self._out_edges: dict[str, set[str]] = {}
        self._in_edges: dict[str, set[str]] = {}

        # Secondary indices
        self._video_nodes: dict[str, set[str]] = {}
        self._type_nodes: dict[GraphNodeType, set[str]] = {}
        self._rel_edges: dict[GraphEdgeType, set[str]] = {}

    def upsert_node(self, node: GraphNode) -> None:
        with self._lock:
            old_node = self._nodes.get(node.node_id)
            if old_node is not None:
                # Remove from old indices if changed
                if old_node.video_id != node.video_id:
                    self._video_nodes.get(old_node.video_id, set()).discard(node.node_id)
                if old_node.node_type != node.node_type:
                    self._type_nodes.get(old_node.node_type, set()).discard(node.node_id)

            self._nodes[node.node_id] = node
            self._video_nodes.setdefault(node.video_id, set()).add(node.node_id)
            self._type_nodes.setdefault(node.node_type, set()).add(node.node_id)

    def upsert_edge(self, edge: GraphEdge) -> None:
        with self._lock:
            old_edge = self._edges.get(edge.edge_id)
            if old_edge is not None:
                if old_edge.relationship != edge.relationship:
                    self._rel_edges.get(old_edge.relationship, set()).discard(edge.edge_id)

            self._edges[edge.edge_id] = edge
            self._out_edges.setdefault(edge.source_node_id, set()).add(edge.edge_id)
            self._in_edges.setdefault(edge.target_node_id, set()).add(edge.edge_id)
            self._rel_edges.setdefault(edge.relationship, set()).add(edge.edge_id)

    def get_node(self, node_id: str) -> GraphNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        with self._lock:
            return self._edges.get(edge_id)

    def edges_for(
        self,
        node_id: str,
        direction: Direction = Direction.BOTH,
        relationship: GraphEdgeType | None = None,
    ) -> list[GraphEdge]:
        with self._lock:
            edge_ids: set[str] = set()
            if direction in (Direction.OUTBOUND, Direction.BOTH):
                edge_ids.update(self._out_edges.get(node_id, set()))
            if direction in (Direction.INBOUND, Direction.BOTH):
                edge_ids.update(self._in_edges.get(node_id, set()))

            result: list[GraphEdge] = []
            for eid in sorted(edge_ids):
                e = self._edges.get(eid)
                if e is not None:
                    if relationship is None or e.relationship == relationship:
                        result.append(e)
            return result

    def neighbors(
        self,
        node_id: str,
        direction: Direction = Direction.BOTH,
        relationship: GraphEdgeType | None = None,
    ) -> list[GraphNode]:
        with self._lock:
            connected_edges = self.edges_for(node_id, direction, relationship)
            neighbor_ids: set[str] = set()
            for edge in connected_edges:
                if edge.source_node_id == node_id:
                    neighbor_ids.add(edge.target_node_id)
                else:
                    neighbor_ids.add(edge.source_node_id)

            result: list[GraphNode] = []
            for nid in sorted(neighbor_ids):
                n = self._nodes.get(nid)
                if n is not None:
                    result.append(n)
            return result

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[GraphPath]:
        """Find paths between source and target via breadth-first search (BFS)."""
        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return []
            if source_id == target_id:
                src_node = self._nodes[source_id]
                return [GraphPath(nodes=(src_node,), edges=())]

            queue: deque[tuple[str, list[GraphNode], list[GraphEdge]]] = deque()
            src_node = self._nodes[source_id]
            queue.append((source_id, [src_node], []))
            found_paths: list[GraphPath] = []

            while queue:
                curr_id, path_nodes, path_edges = queue.popleft()
                if len(path_edges) >= max_depth:
                    continue

                for edge in self.edges_for(curr_id, direction=Direction.OUTBOUND):
                    next_id = edge.target_node_id
                    next_node = self._nodes.get(next_id)
                    if next_node is None or any(n.node_id == next_id for n in path_nodes):
                        continue

                    new_nodes = [*path_nodes, next_node]
                    new_edges = [*path_edges, edge]

                    if next_id == target_id:
                        found_paths.append(
                            GraphPath(nodes=tuple(new_nodes), edges=tuple(new_edges))
                        )
                    else:
                        queue.append((next_id, new_nodes, new_edges))

            return found_paths

    def query_nodes(self, node_filter: NodeFilter) -> list[GraphNode]:
        with self._lock:
            candidate_ids: set[str] | None = None

            if node_filter.video_id is not None:
                candidate_ids = set(self._video_nodes.get(str(node_filter.video_id), set()))

            if node_filter.node_types is not None:
                type_matched: set[str] = set()
                for nt in node_filter.node_types:
                    type_matched.update(self._type_nodes.get(nt, set()))
                candidate_ids = (
                    type_matched if candidate_ids is None else candidate_ids & type_matched
                )

            if candidate_ids is not None:
                source_set = set(candidate_ids)
            else:
                source_set = set(self._nodes.keys())

            results: list[GraphNode] = []
            for nid in sorted(source_set):
                node = self._nodes.get(nid)
                if node is None:
                    continue

                if (
                    node_filter.source_ids is not None
                    and node.source_id not in node_filter.source_ids
                ):
                    continue

                if node_filter.time_start is not None:
                    end = node.timestamp_end
                    if end is not None and end < node_filter.time_start:
                        continue

                if node_filter.time_end is not None:
                    start = node.timestamp_start
                    if start is not None and start > node_filter.time_end:
                        continue

                if node_filter.label_contains is not None:
                    if node_filter.label_contains.lower() not in node.label.lower():
                        continue

                results.append(node)
                if node_filter.limit is not None and len(results) >= node_filter.limit:
                    break

            return results

    def query_edges(self, edge_filter: EdgeFilter) -> list[GraphEdge]:
        with self._lock:
            candidate_ids: set[str] | None = None

            if edge_filter.relationships is not None:
                rel_matched: set[str] = set()
                for rel in edge_filter.relationships:
                    rel_matched.update(self._rel_edges.get(rel, set()))
                candidate_ids = rel_matched

            if candidate_ids is not None:
                source_set = set(candidate_ids)
            else:
                source_set = set(self._edges.keys())

            results: list[GraphEdge] = []
            for eid in sorted(source_set):
                edge = self._edges.get(eid)
                if edge is None:
                    continue

                if (
                    edge_filter.source_node_ids is not None
                    and edge.source_node_id not in edge_filter.source_node_ids
                ):
                    continue

                if (
                    edge_filter.target_node_ids is not None
                    and edge.target_node_id not in edge_filter.target_node_ids
                ):
                    continue

                if (
                    edge_filter.derivations is not None
                    and edge.provenance.derivation not in edge_filter.derivations
                ):
                    continue

                if (
                    edge_filter.min_confidence is not None
                    and edge.provenance.confidence < edge_filter.min_confidence
                ):
                    continue

                results.append(edge)
                if edge_filter.limit is not None and len(results) >= edge_filter.limit:
                    break

            return results

    def count_nodes(self, video_id: str | None = None) -> int:
        with self._lock:
            if video_id is None:
                return len(self._nodes)
            return len(self._video_nodes.get(str(video_id), set()))

    def count_edges(self) -> int:
        with self._lock:
            return len(self._edges)

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._out_edges.clear()
            self._in_edges.clear()
            self._video_nodes.clear()
            self._type_nodes.clear()
            self._rel_edges.clear()
