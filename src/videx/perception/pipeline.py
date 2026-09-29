"""Perception processing pipeline coordinating detection and tracking.

Consumes DecodedFrame stream from VideoReader, evaluates DetectionProvider,
updates TrackingProvider, and compiles evidence records and trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from videx.domain.schemas import (
    Detection,
    Evidence,
    EvidenceType,
    Track,
    TrajectoryPoint,
)
from videx.ingestion.base import DecodedFrame, VideoReader
from videx.perception.trajectory import TrajectoryAnalyzer
from videx.providers.base import DetectionProvider, TrackingProvider


@dataclass
class PerceptionResult:
    """Accumulated perception analysis output for a video or clip."""

    video_id: UUID
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    trajectories: dict[UUID, list[TrajectoryPoint]] = field(default_factory=dict)
    trajectory_summaries: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def total_tracks(self) -> int:
        return len(self.tracks)


class PerceptionPipeline:
    """Coordinates object detection and tracking across a video stream.

    Guarantees:
    - Never invents timestamps: uses DecodedFrame.frame_timestamp strictly.
    - Emits structured Evidence records for all verified detections and tracks.
    - Computes pixel-space trajectory properties without physical speed claims.
    """

    def __init__(
        self,
        detector: DetectionProvider,
        tracker: TrackingProvider,
        min_track_length: int = 2,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.min_track_length = min_track_length

    def process_reader(
        self,
        reader: VideoReader,
        start_frame: int = 0,
        end_frame: int | None = None,
        stride: int = 1,
    ) -> PerceptionResult:
        """Run detection and tracking sequentially across frames from a VideoReader."""
        self.tracker.reset()

        total = reader.total_frames
        end = min(total, end_frame) if end_frame is not None else total
        video_id = reader.video_id or uuid4()

        result = PerceptionResult(video_id=video_id)

        for frame_idx in range(start_frame, end, max(stride, 1)):
            decoded: DecodedFrame = reader.read_decoded_frame(frame_idx)

            # 1. Run detection on DecodedFrame
            frame_dets = self.detector.detect(decoded)
            result.detections.extend(frame_dets)

            # 2. Run tracker update
            active_points = self.tracker.update(frame_dets, decoded)

            for pt in active_points:
                if pt.track_id not in result.trajectories:
                    result.trajectories[pt.track_id] = []
                result.trajectories[pt.track_id].append(pt)

        # 3. Retrieve all consolidated tracks
        all_tracks = getattr(self.tracker, "all_tracks", self.tracker.active_tracks)()
        for trk in all_tracks:
            if trk.frame_count >= self.min_track_length:
                result.tracks.append(trk)

                # Compute trajectory analytics
                pts = result.trajectories.get(trk.track_id, [])
                summary = TrajectoryAnalyzer.analyze_trajectory(pts)
                result.trajectory_summaries[trk.track_id] = summary

                # Generate Track Evidence record
                result.evidence.append(
                    Evidence(
                        evidence_type=EvidenceType.TRACK,
                        source_module=f"{self.tracker.provider_name}_tracker",
                        video_id=video_id,
                        track_id=trk.track_id,
                        timestamp_seconds=trk.first_seen_timestamp_seconds,
                        bbox=trk.end_bbox or trk.start_bbox,
                        confidence=trk.confidence,
                        description=(
                            f"Tracked {trk.class_name} across {trk.frame_count} frames "
                            f"({trk.duration_seconds:.2f}s, "
                            f"displacement {summary['total_displacement_px']}px, "
                            f"direction {summary['direction_cardinal']})"
                        ),
                        raw_payload={
                            "first_seen_frame": trk.first_seen_frame_number,
                            "last_seen_frame": trk.last_seen_frame_number,
                            "first_seen_ts": trk.first_seen_timestamp_seconds,
                            "last_seen_ts": trk.last_seen_timestamp_seconds,
                            "trajectory": summary,
                        },
                        supporting_observation_ids=list(trk.detection_ids),
                    )
                )

        return result
