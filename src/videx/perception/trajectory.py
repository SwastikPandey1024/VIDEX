"""Spatial trajectory calculation and analysis.

Tracks pixel-space movements, centroids, displacements, and direction vectors
without physical speed claims.
"""

from __future__ import annotations

import math
from typing import Any

from videx.domain.schemas import BoundingBox, TrajectoryPoint


class TrajectoryAnalyzer:
    """Computes geometric and kinematic properties of tracked object paths in pixel space.

    Rule: Never claim physical speed (e.g. km/h, m/s). All calculations operate
    strictly in pixel displacement (pixels, pixels/sec, angle/direction).
    """

    @staticmethod
    def compute_centroid(bbox: BoundingBox) -> tuple[float, float]:
        """Return (cx, cy) pixel coordinates of the bounding box centroid."""
        return (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)

    @classmethod
    def analyze_trajectory(
        cls,
        points: list[TrajectoryPoint],
    ) -> dict[str, Any]:
        """Analyze a sequential list of TrajectoryPoints.

        Returns summary dictionary containing:
        - point_count: int
        - total_displacement_px: float (distance from first centroid to last)
        - path_length_px: float (cumulative sum of step-to-step distances)
        - duration_seconds: float
        - mean_pixel_velocity: float (path_length_px / duration_seconds)
        - direction_degrees: float | None (0 to 360 degrees where 0 = +X/right, 90 = +Y/down)
        - direction_cardinal: str (e.g., 'right', 'down-right', 'stationary')
        """
        if not points:
            return {
                "point_count": 0,
                "total_displacement_px": 0.0,
                "path_length_px": 0.0,
                "duration_seconds": 0.0,
                "mean_pixel_velocity": 0.0,
                "direction_degrees": None,
                "direction_cardinal": "unknown",
            }

        if len(points) == 1:
            return {
                "point_count": 1,
                "total_displacement_px": 0.0,
                "path_length_px": 0.0,
                "duration_seconds": 0.0,
                "mean_pixel_velocity": 0.0,
                "direction_degrees": None,
                "direction_cardinal": "stationary",
            }

        # Sort points by timestamp / frame number for sequential analysis
        sorted_pts = sorted(points, key=lambda p: (p.frame_number, p.timestamp_seconds))

        centroids = [cls.compute_centroid(p.bbox) for p in sorted_pts]
        path_length = 0.0

        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            path_length += math.hypot(dx, dy)

        first_c = centroids[0]
        last_c = centroids[-1]
        disp_x = last_c[0] - first_c[0]
        disp_y = last_c[1] - first_c[1]
        total_disp = math.hypot(disp_x, disp_y)

        duration = max(sorted_pts[-1].timestamp_seconds - sorted_pts[0].timestamp_seconds, 1e-6)
        mean_vel = path_length / duration

        # Direction calculation (image coordinates: +x right, +y down)
        deg: float | None = None
        cardinal = "stationary"

        if total_disp >= 2.0:  # Minimum 2 pixels movement threshold
            rad = math.atan2(disp_y, disp_x)
            deg = math.degrees(rad) % 360.0
            cardinal = cls._degrees_to_cardinal(deg)

        return {
            "point_count": len(sorted_pts),
            "total_displacement_px": round(total_disp, 2),
            "path_length_px": round(path_length, 2),
            "duration_seconds": round(duration, 4),
            "mean_pixel_velocity": round(mean_vel, 2),
            "direction_degrees": round(deg, 1) if deg is not None else None,
            "direction_cardinal": cardinal,
        }

    @staticmethod
    def _degrees_to_cardinal(degrees: float) -> str:
        """Map angle (0-360) in image space (+x right, +y down) to cardinal direction."""
        # 0 deg = right
        # 45 deg = down-right
        # 90 deg = down
        # 135 deg = down-left
        # 180 deg = left
        # 227 deg = up-left
        # 270 deg = up
        # 315 deg = up-right
        val = int((degrees + 22.5) / 45.0) % 8
        directions = [
            "right",
            "down-right",
            "down",
            "down-left",
            "left",
            "up-left",
            "up",
            "up-right",
        ]
        return directions[val]
