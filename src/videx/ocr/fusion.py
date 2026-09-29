"""Temporal OCR Fusion layer for aggregating repeated per-frame observations
into durable text evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from videx.domain.schemas import BoundingBox, FrameTimestamp, OCRObservation, TextObservation


@dataclass
class TemporalOCRFusionConfig:
    """Configuration for temporal text observation clustering and fusion."""

    max_frame_gap: int = 5
    """Maximum consecutive unobserved frames before closing a temporal text cluster."""

    min_iou_overlap: float = 0.2
    """Minimum IoU overlap between bounding boxes to consider them the same spatial region."""

    max_centroid_distance_px: float = 60.0
    """Maximum centroid displacement in pixels when IoU is low (handles text pan/motion)."""

    spatial_iou_threshold: float | None = None
    """Alias for min_iou_overlap."""

    exact_normalized_text: bool = True
    """Whether normalized text must match identically to be clustered."""

    def __post_init__(self) -> None:
        if self.spatial_iou_threshold is not None:
            self.min_iou_overlap = self.spatial_iou_threshold


def _compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute standard Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(b1.x, b2.x)
    y1 = max(b1.y, b2.y)
    x2 = min(b1.x + b1.width, b2.x + b2.width)
    y2 = min(b1.y + b1.height, b2.y + b2.height)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    union_area = (b1.width * b1.height) + (b2.width * b2.height) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _centroid_distance(b1: BoundingBox, b2: BoundingBox) -> float:
    """Calculate Euclidean distance between bounding box centroids."""
    c1_x = b1.x + b1.width / 2.0
    c1_y = b1.y + b1.height / 2.0
    c2_x = b2.x + b2.width / 2.0
    c2_y = b2.y + b2.height / 2.0
    return math.hypot(c2_x - c1_x, c2_y - c1_y)


@dataclass
class _ActiveTextCluster:
    """Internal mutable tracking cluster for temporal fusion."""

    cluster_id: UUID = field(default_factory=uuid4)
    video_id: UUID = field(default_factory=uuid4)
    best_text: str = ""
    normalized_text: str = ""
    best_confidence: float = 0.0
    language: str | None = "en"
    script: str = "Latin"
    provider: str = ""

    first_seen_timestamp: FrameTimestamp = field(default=None)  # type: ignore[assignment]
    last_seen_timestamp: FrameTimestamp = field(default=None)  # type: ignore[assignment]
    first_seen_frame: int = 0
    last_seen_frame: int = 0

    supporting_frames: list[int] = field(default_factory=list)
    supporting_observation_ids: list[UUID] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    bbox_history: list[BoundingBox] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def is_spatially_compatible(
        self,
        new_bbox: BoundingBox | None,
        min_iou: float,
        max_dist: float,
    ) -> bool:
        """Evaluate spatial proximity against recent bounding box history."""
        if not self.bbox_history or new_bbox is None:
            # If spatial data is missing on either side, fall back to pure text identity
            return True

        latest_bbox = self.bbox_history[-1]
        iou = _compute_iou(latest_bbox, new_bbox)
        if iou >= min_iou:
            return True

        dist = _centroid_distance(latest_bbox, new_bbox)
        return dist <= max_dist

    def add_observation(self, obs: OCRObservation) -> None:
        """Incorporate a new raw observation into this temporal cluster."""
        self.last_seen_frame = obs.frame_number
        self.last_seen_timestamp = obs.frame_timestamp
        self.supporting_frames.append(obs.frame_number)
        self.supporting_observation_ids.append(obs.observation_id)
        self.confidences.append(obs.confidence)

        if obs.bbox is not None:
            self.bbox_history.append(obs.bbox)

        # Update representative text if this observation has higher confidence
        if obs.confidence >= self.best_confidence:
            self.best_text = obs.text
            self.best_confidence = obs.confidence

    def to_text_observation(self) -> TextObservation:
        """Synthesize finalized immutable domain TextObservation."""
        confs = self.confidences or [self.best_confidence]
        conf_summary = {
            "mean": float(sum(confs) / len(confs)),
            "min": float(min(confs)),
            "max": float(max(confs)),
            "count": float(len(confs)),
        }
        return TextObservation(
            observation_id=self.cluster_id,
            video_id=self.video_id,
            text=self.best_text,
            normalized_text=self.normalized_text,
            first_seen_timestamp=self.first_seen_timestamp,
            last_seen_timestamp=self.last_seen_timestamp,
            first_seen_frame=self.first_seen_frame,
            last_seen_frame=self.last_seen_frame,
            first_seen_timestamp_seconds=self.first_seen_timestamp.pts_seconds,
            last_seen_timestamp_seconds=self.last_seen_timestamp.pts_seconds,
            supporting_frames=list(self.supporting_frames),
            supporting_observation_ids=list(self.supporting_observation_ids),
            confidence_summary=conf_summary,
            bbox_history=list(self.bbox_history),
            language=self.language,
            script=self.script,
            provider=self.provider,
            attributes=dict(self.attributes),
        )


class TemporalOCRFusion:
    """Fuses multi-frame OCR observations into unified, continuous TextObservations.

    Prevents raw observation explosion by aggregating continuous screen text occurrences
    while retaining explicit pointers to every supporting raw observation ID and frame.
    """

    def __init__(self, config: TemporalOCRFusionConfig | None = None) -> None:
        self.config = config or TemporalOCRFusionConfig()
        self._active_clusters: list[_ActiveTextCluster] = []
        self._completed_observations: list[TextObservation] = []
        self._current_frame: int = -1

    def add_observation(self, obs: OCRObservation) -> list[TextObservation]:
        """Convenience method to ingest a single observation."""
        return self.update([obs], current_frame=obs.frame_number)

    def update(
        self,
        observations: list[OCRObservation],
        current_frame: int | None = None,
    ) -> list[TextObservation]:
        """Ingest frame OCR observations, matching to active clusters or creating new ones.

        Args:
            observations: Raw OCR detections from current frame.
            current_frame: Frame index (inferred from observations if None).

        Returns:
            List of TextObservations that were closed and completed during this update.
        """
        if observations:
            frame_idx = observations[0].frame_number if current_frame is None else current_frame
        else:
            frame_idx = self._current_frame + 1 if current_frame is None else current_frame

        self._current_frame = frame_idx

        # 1. Retire clusters whose gap exceeds max_frame_gap before matching incoming observations
        still_active: list[_ActiveTextCluster] = []
        newly_closed: list[TextObservation] = []

        for cluster in self._active_clusters:
            gap = frame_idx - cluster.last_seen_frame
            if gap > self.config.max_frame_gap:
                finalized = cluster.to_text_observation()
                self._completed_observations.append(finalized)
                newly_closed.append(finalized)
            else:
                still_active.append(cluster)

        self._active_clusters = still_active

        # 2. Match each new observation to an existing active cluster
        unmatched_observations: list[OCRObservation] = []

        for obs in observations:
            matched_cluster: _ActiveTextCluster | None = None
            best_overlap = -1.0

            for cluster in self._active_clusters:
                # Check frame gap continuity
                if (obs.frame_number - cluster.last_seen_frame) > self.config.max_frame_gap:
                    continue

                # Check text match
                if self.config.exact_normalized_text:
                    if obs.normalized_text != cluster.normalized_text:
                        continue
                else:
                    if obs.normalized_text.lower() != cluster.normalized_text.lower():
                        continue

                # Check spatial compatibility
                if not cluster.is_spatially_compatible(
                    obs.bbox,
                    min_iou=self.config.min_iou_overlap,
                    max_dist=self.config.max_centroid_distance_px,
                ):
                    continue

                # Prefer cluster with highest spatial IoU or most recently seen
                overlap = (
                    _compute_iou(cluster.bbox_history[-1], obs.bbox)
                    if cluster.bbox_history and obs.bbox
                    else 1.0
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_cluster = cluster

            if matched_cluster is not None:
                matched_cluster.add_observation(obs)
            else:
                unmatched_observations.append(obs)

        # 2. Spawn new clusters for unmatched observations
        for obs in unmatched_observations:
            new_cluster = _ActiveTextCluster(
                cluster_id=uuid4(),
                video_id=obs.video_id,
                best_text=obs.text,
                normalized_text=obs.normalized_text,
                best_confidence=obs.confidence,
                language=obs.language,
                script=obs.script,
                provider=obs.provider,
                first_seen_timestamp=obs.frame_timestamp,
                last_seen_timestamp=obs.frame_timestamp,
                first_seen_frame=obs.frame_number,
                last_seen_frame=obs.frame_number,
                supporting_frames=[obs.frame_number],
                supporting_observation_ids=[obs.observation_id],
                confidences=[obs.confidence],
                bbox_history=[obs.bbox] if obs.bbox is not None else [],
            )
            self._active_clusters.append(new_cluster)

        return newly_closed

    def finalize(self) -> list[TextObservation]:
        """Flush all remaining active clusters and return complete list of TextObservations."""
        for cluster in self._active_clusters:
            self._completed_observations.append(cluster.to_text_observation())
        self._active_clusters.clear()
        return list(self._completed_observations)

    def all_fused_observations(self) -> list[TextObservation]:
        """Return all fused observations completed to date plus current active clusters."""
        active_snapshots = [c.to_text_observation() for c in self._active_clusters]
        return list(self._completed_observations) + active_snapshots

    def reset(self) -> None:
        """Reset internal fusion state."""
        self._active_clusters.clear()
        self._completed_observations.clear()
        self._current_frame = -1
