"""Deterministic spatial relationship engine for the Evidence Graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from videx.domain.schemas import BoundingBox, Detection, Track
from videx.events.spatial import SpatialZone
from videx.graph.edges import GraphEdge, create_edge
from videx.graph.nodes import make_node_id
from videx.graph.types import DerivationType, GraphEdgeType, GraphNodeType


def compute_bbox_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box_a.x, box_b.x)
    y_top = max(box_a.y, box_b.y)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = box_a.area
    box_b_area = box_b.area
    union_area = box_a_area + box_b_area - intersection_area

    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


class SpatialGraphEngine:
    """Computes deterministic spatial relationships (SPATIALLY_OVERLAPS, OCCURS_IN)."""

    def __init__(self, iou_threshold: float = 0.1) -> None:
        self.iou_threshold = iou_threshold

    def compute_spatial_overlap_edges(
        self,
        detections: Sequence[Detection],
    ) -> list[GraphEdge]:
        """Compute co-temporal bounding box overlap edges between detections in the same frame."""
        edges: list[GraphEdge] = []
        by_frame: dict[str, list[Detection]] = defaultdict(list)
        for d in detections:
            by_frame[str(d.frame_id)].append(d)

        for frame_id in sorted(by_frame.keys()):
            frame_dets = sorted(by_frame[frame_id], key=lambda d: str(d.detection_id))
            n = len(frame_dets)
            for i in range(n):
                d_a = frame_dets[i]
                for j in range(i + 1, n):
                    d_b = frame_dets[j]
                    iou = compute_bbox_iou(d_a.bbox, d_b.bbox)
                    if iou >= self.iou_threshold:
                        nid_a = make_node_id(GraphNodeType.DETECTION, d_a.detection_id)
                        nid_b = make_node_id(GraphNodeType.DETECTION, d_b.detection_id)
                        edges.append(
                            create_edge(
                                source_node_id=nid_a,
                                target_node_id=nid_b,
                                relationship=GraphEdgeType.SPATIALLY_OVERLAPS,
                                derivation=DerivationType.DETERMINISTIC,
                                threshold=self.iou_threshold,
                                actual_value=round(iou, 4),
                                reason=(
                                    f"Frame {d_a.frame_number} IoU {iou:.3f} >= "
                                    f"threshold {self.iou_threshold:.2f}"
                                ),
                            )
                        )

        return edges

    def compute_zone_containment_edges(
        self,
        tracks: Sequence[Track],
        zones: Sequence[SpatialZone],
    ) -> list[GraphEdge]:
        """Compute OCCURS_IN edges when track centroids fall inside SpatialZones."""
        edges: list[GraphEdge] = []
        for trk in sorted(tracks, key=lambda t: str(t.track_id)):
            trk_nid = make_node_id(GraphNodeType.TRACK, trk.track_id)

            # Check appearance or disappearance bbox centroids
            sample_boxes: list[BoundingBox] = []
            if trk.start_bbox:
                sample_boxes.append(trk.start_bbox)
            if trk.end_bbox and trk.end_bbox != trk.start_bbox:
                sample_boxes.append(trk.end_bbox)

            for z in sorted(zones, key=lambda item: item.zone_id):
                zone_nid = make_node_id(GraphNodeType.ZONE, z.zone_id)
                is_contained = False
                for box in sample_boxes:
                    cx = box.center_x
                    cy = box.center_y
                    if z.contains_point(cx, cy):
                        is_contained = True
                        break

                if is_contained:
                    edges.append(
                        create_edge(
                            source_node_id=trk_nid,
                            target_node_id=zone_nid,
                            relationship=GraphEdgeType.OCCURS_IN,
                            derivation=DerivationType.DETERMINISTIC,
                            reason=f"Track centroid falls inside polygon of zone '{z.zone_name}'",
                        )
                    )

        return edges
