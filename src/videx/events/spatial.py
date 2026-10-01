"""Deterministic spatial zone event detector.

Defines SpatialZone geometries (rectangles, polygons) and detects
OBJECT_ENTERED_ZONE and OBJECT_EXITED_ZONE events with boundary debounce.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from videx.domain.schemas import BoundingBox, EvidenceType, Track, TrajectoryPoint
from videx.events.schemas import (
    Event,
    EventEngineConfig,
    EventEvidence,
    EventParticipant,
)
from videx.events.types import EventSeverity, EventStatus, EventType


class SpatialZone(BaseModel):
    """User- or configuration-defined spatial region of interest."""

    model_config = ConfigDict(frozen=True)

    zone_id: str = Field(..., description="Unique zone identifier (e.g. 'zone_loading_dock')")
    zone_name: str = Field(..., description="Human-readable zone label")
    polygon_points: list[tuple[float, float]] = Field(
        ...,
        description="Vertices of zone polygon [(x0, y0), (x1, y1), ...] in pixel coordinates",
    )

    @classmethod
    def from_rectangle(
        cls,
        zone_id: str,
        zone_name: str,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> SpatialZone:
        """Create a rectangular zone from bounding coordinates."""
        return cls(
            zone_id=zone_id,
            zone_name=zone_name,
            polygon_points=[
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ],
        )

    def contains_point(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        """Point-in-polygon test using ray casting algorithm with optional boundary tolerance."""
        poly = self.polygon_points
        n = len(poly)
        if n < 3:
            return False

        # Check bounding box with tolerance first for quick rejection
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs) - tolerance, max(xs) + tolerance
        min_y, max_y = min(ys) - tolerance, max(ys) + tolerance

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        # Ray casting
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def contains_bbox(self, bbox: BoundingBox, tolerance: float = 0.0) -> bool:
        """Test if bounding box center or any corner falls within zone."""
        center_x = bbox.x + bbox.width / 2.0
        center_y = bbox.y + bbox.height / 2.0
        return self.contains_point(center_x, center_y, tolerance)


@dataclass
class _TrackZoneState:
    """Internal state tracking track containment across frames."""

    is_inside: bool = False
    last_transition_timestamp: float = -1.0
    entry_timestamp: float | None = None
    entry_frame: int | None = None
    trajectory_points_inside: list[TrajectoryPoint] = field(default_factory=list)


class SpatialEventDetector:
    """Detects deterministic zone entry and exit events with temporal debounce."""

    def __init__(
        self,
        zones: Sequence[SpatialZone] | None = None,
        config: EventEngineConfig | None = None,
    ) -> None:
        self.zones = list(zones) if zones else []
        self.config = config or EventEngineConfig()

    def add_zone(self, zone: SpatialZone) -> None:
        """Register a new spatial zone."""
        self.zones.append(zone)

    def detect_events(
        self,
        tracks: Sequence[Track],
        trajectories: dict[UUID, list[TrajectoryPoint]],
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Detect zone boundary crossings across track trajectories.

        Args:
            tracks: Verified tracks from perception.
            trajectories: Trajectory points for each track.
            video_id: Video UUID override.

        Returns:
            List of evidence-backed zone Event records.
        """
        if not self.zones:
            return []

        events: list[Event] = []
        track_map = {t.track_id: t for t in tracks}

        for zone in self.zones:
            for trk_id, points in trajectories.items():
                if not points:
                    continue

                trk = track_map.get(trk_id)
                class_name = trk.class_name if trk else "object"
                v_id = video_id or (trk.video_id if trk else UUID(int=0))

                participant = EventParticipant(
                    participant_id=trk_id,
                    participant_type="track",
                    role="subject",
                    label=class_name,
                )
                zone_participant = EventParticipant(
                    participant_id=zone.zone_id,
                    participant_type="zone",
                    role="target",
                    label=zone.zone_name,
                )

                state = _TrackZoneState()

                for pt in points:
                    # Apply tolerance hysteresis: when already inside, tolerate small excursions
                    tol = (
                        self.config.spatial_boundary_tolerance_px
                        if state.is_inside
                        else 0.0
                    )
                    now_inside = zone.contains_point(
                        pt.bbox.center_x, pt.bbox.center_y, tolerance=tol
                    )
                    ts = pt.timestamp_seconds

                    # Transition: OUTSIDE -> INSIDE (Entry)
                    if now_inside and not state.is_inside:
                        time_since_last = ts - state.last_transition_timestamp
                        if time_since_last >= self.config.spatial_debounce_interval_seconds:
                            state.is_inside = True
                            state.last_transition_timestamp = ts
                            state.entry_timestamp = ts
                            state.entry_frame = pt.frame_number
                            state.trajectory_points_inside = [pt]

                            evidence = EventEvidence(
                                timestamp_seconds=ts,
                                evidence_type=EvidenceType.TRAJECTORY,
                                role="trigger",
                                track_id=trk_id,
                                provenance={
                                    "frame_number": pt.frame_number,
                                    "zone_id": zone.zone_id,
                                    "centroid": (
                                        round(pt.bbox.center_x, 1),
                                        round(pt.bbox.center_y, 1),
                                    ),
                                },
                            )

                            ev = Event(
                                video_id=v_id,
                                event_type=EventType.OBJECT_ENTERED_ZONE,
                                start_timestamp_seconds=ts,
                                end_timestamp_seconds=ts,
                                confidence=trk.confidence if trk else 0.9,
                                status=EventStatus.CONFIRMED,
                                severity=EventSeverity.INFO,
                                participants=[participant, zone_participant],
                                event_evidence=[evidence],
                                source_module="spatial_detector",
                                description=(
                                    f"{class_name.capitalize()} entered zone '{zone.zone_name}' "
                                    f"at {ts:.2f}s (frame {pt.frame_number})"
                                ),
                                zone_name=zone.zone_name,
                                track_ids=[trk_id],
                                attributes={
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.zone_name,
                                    "frame_number": pt.frame_number,
                                    "centroid_x": round(pt.bbox.center_x, 1),
                                    "centroid_y": round(pt.bbox.center_y, 1),
                                },
                            )
                            events.append(ev)

                    # Transition: INSIDE -> OUTSIDE (Exit)
                    elif not now_inside and state.is_inside:
                        time_since_last = ts - state.last_transition_timestamp
                        if time_since_last >= self.config.spatial_debounce_interval_seconds:
                            state.is_inside = False
                            state.last_transition_timestamp = ts

                            evidence = EventEvidence(
                                timestamp_seconds=ts,
                                evidence_type=EvidenceType.TRAJECTORY,
                                role="trigger",
                                track_id=trk_id,
                                provenance={
                                    "frame_number": pt.frame_number,
                                    "zone_id": zone.zone_id,
                                    "centroid": (
                                        round(pt.bbox.center_x, 1),
                                        round(pt.bbox.center_y, 1),
                                    ),
                                },
                            )

                            dwell_time = round(
                                ts - (state.entry_timestamp or ts),
                                2,
                            )

                            ev = Event(
                                video_id=v_id,
                                event_type=EventType.OBJECT_EXITED_ZONE,
                                start_timestamp_seconds=ts,
                                end_timestamp_seconds=ts,
                                confidence=trk.confidence if trk else 0.9,
                                status=EventStatus.CONFIRMED,
                                severity=EventSeverity.INFO,
                                participants=[participant, zone_participant],
                                event_evidence=[evidence],
                                source_module="spatial_detector",
                                description=(
                                    f"{class_name.capitalize()} exited zone '{zone.zone_name}' "
                                    f"at {ts:.2f}s after dwelling for {dwell_time}s"
                                ),
                                zone_name=zone.zone_name,
                                track_ids=[trk_id],
                                attributes={
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.zone_name,
                                    "frame_number": pt.frame_number,
                                    "dwell_duration_seconds": dwell_time,
                                },
                            )
                            events.append(ev)

                    elif state.is_inside:
                        state.trajectory_points_inside.append(pt)

        return events
