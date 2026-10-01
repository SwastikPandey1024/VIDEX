"""Deterministic movement event detector based on pixel-space trajectory kinematics.

Emits OBJECT_STARTED_MOVING, OBJECT_STOPPED_MOVING, and OBJECT_CHANGED_DIRECTION
events using strictly pixel-space displacement and velocity.
DO NOT claim physical speed (e.g. km/h) without external calibration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID

from videx.domain.schemas import EvidenceType, Track, TrajectoryPoint
from videx.events.schemas import (
    Event,
    EventEngineConfig,
    EventEvidence,
    EventParticipant,
)
from videx.events.types import EventSeverity, EventStatus, EventType


def _calculate_angle_degrees(dx: float, dy: float) -> float:
    """Calculate direction angle in degrees [0, 360) where 0 is right, 90 is down."""
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360.0) % 360.0


def _angular_difference(a1: float, a2: float) -> float:
    """Compute the shortest angular difference in degrees [0, 180]."""
    diff = abs(a1 - a2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def _cardinal_direction(degrees: float) -> str:
    """Map angle in degrees to 8-point compass direction."""
    points = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    idx = int((degrees + 22.5) / 45.0) % 8
    return points[idx]


class MovementEventDetector:
    """Detects deterministic kinematic transitions in pixel space from trajectory points."""

    def __init__(self, config: EventEngineConfig | None = None) -> None:
        self.config = config or EventEngineConfig()

    def detect_events(
        self,
        tracks: Sequence[Track],
        trajectories: dict[UUID, list[TrajectoryPoint]],
        video_id: UUID | None = None,
    ) -> list[Event]:
        """Characterize motion and emit movement events for each track trajectory.

        Args:
            tracks: Sequence of verified tracks.
            trajectories: Mapping from track_id to chronologically ordered TrajectoryPoints.
            video_id: Video UUID override if not derived from tracks.

        Returns:
            List of evidence-backed movement Event records.
        """
        events: list[Event] = []

        track_map = {t.track_id: t for t in tracks}

        for trk_id, points in trajectories.items():
            if len(points) < 2:
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

            # Analyze points window by window
            was_moving: bool | None = None
            last_heading: float | None = None
            last_change_ts = -1.0

            for i in range(1, len(points)):
                p_prev = points[i - 1]
                p_curr = points[i]

                dt = max(0.001, p_curr.timestamp_seconds - p_prev.timestamp_seconds)
                dx = p_curr.bbox.center_x - p_prev.bbox.center_x
                dy = p_curr.bbox.center_y - p_prev.bbox.center_y
                dist = math.hypot(dx, dy)
                speed = dist / dt

                is_moving = speed >= self.config.movement_velocity_threshold_px_s

                # ── 1. State Transitions: Started / Stopped Moving ─────────
                if was_moving is not None and is_moving != was_moving:
                    ev_type = (
                        EventType.OBJECT_STARTED_MOVING
                        if is_moving
                        else EventType.OBJECT_STOPPED_MOVING
                    )
                    desc_action = "started moving" if is_moving else "stopped moving"

                    evidence = EventEvidence(
                        timestamp_seconds=p_curr.timestamp_seconds,
                        evidence_type=EvidenceType.TRAJECTORY,
                        role="trigger",
                        track_id=trk_id,
                        provenance={
                            "frame_number": p_curr.frame_number,
                            "speed_px_per_sec": round(speed, 2),
                            "displacement_px": round(dist, 2),
                        },
                    )

                    ev = Event(
                        video_id=v_id,
                        event_type=ev_type,
                        start_timestamp_seconds=p_curr.timestamp_seconds,
                        end_timestamp_seconds=p_curr.timestamp_seconds,
                        confidence=trk.confidence if trk else 0.9,
                        status=EventStatus.CONFIRMED,
                        severity=EventSeverity.INFO,
                        participants=[participant],
                        event_evidence=[evidence],
                        source_module="movement_detector",
                        description=(
                            f"{class_name.capitalize()} {desc_action} at "
                            f"{p_curr.timestamp_seconds:.2f}s (velocity: {speed:.1f}px/s)"
                        ),
                        track_ids=[trk_id],
                        attributes={
                            "distance_pixels": round(dist, 2),
                            "mean_velocity_px_s": round(speed, 2),
                            "movement_state": "moving" if is_moving else "stationary",
                            "frame_number": p_curr.frame_number,
                        },
                    )
                    events.append(ev)

                was_moving = is_moving

                # ── 2. Direction Change ───────────────────────────────────
                if is_moving and dist >= 3.0:  # Minimum displacement to calculate reliable heading
                    current_heading = _calculate_angle_degrees(dx, dy)
                    if last_heading is not None:
                        angular_diff = _angular_difference(last_heading, current_heading)
                        # Check debounce: don't emit repeated direction changes on adjacent frames
                        if (
                            angular_diff >= self.config.direction_change_degrees_threshold
                            and (p_curr.timestamp_seconds - last_change_ts) >= 0.3
                        ):
                            last_change_ts = p_curr.timestamp_seconds
                            cardinal = _cardinal_direction(current_heading)
                            prev_cardinal = _cardinal_direction(last_heading)

                            evidence = EventEvidence(
                                timestamp_seconds=p_curr.timestamp_seconds,
                                evidence_type=EvidenceType.TRAJECTORY,
                                role="trigger",
                                track_id=trk_id,
                                provenance={
                                    "frame_number": p_curr.frame_number,
                                    "previous_heading_deg": round(last_heading, 1),
                                    "new_heading_deg": round(current_heading, 1),
                                    "angular_diff_deg": round(angular_diff, 1),
                                },
                            )

                            ev = Event(
                                video_id=v_id,
                                event_type=EventType.OBJECT_CHANGED_DIRECTION,
                                start_timestamp_seconds=p_curr.timestamp_seconds,
                                end_timestamp_seconds=p_curr.timestamp_seconds,
                                confidence=trk.confidence if trk else 0.85,
                                status=EventStatus.CONFIRMED,
                                severity=EventSeverity.INFO,
                                participants=[participant],
                                event_evidence=[evidence],
                                source_module="movement_detector",
                                description=(
                                    f"{class_name.capitalize()} changed direction from "
                                    f"{prev_cardinal} to {cardinal} "
                                    f"(deflection: {angular_diff:.1f}°) at "
                                    f"{p_curr.timestamp_seconds:.2f}s"
                                ),
                                track_ids=[trk_id],
                                attributes={
                                    "direction_degrees": round(current_heading, 1),
                                    "previous_direction_degrees": round(last_heading, 1),
                                    "angular_deflection_degrees": round(angular_diff, 1),
                                    "direction_cardinal": cardinal,
                                    "previous_cardinal": prev_cardinal,
                                    "mean_velocity_px_s": round(speed, 2),
                                    "frame_number": p_curr.frame_number,
                                },
                            )
                            events.append(ev)

                    last_heading = current_heading

        return events
