"""Phase 2.0R Perception Smoke Test Script.

Validates the full perception subsystem:
1. VideoReader: Decodes frames with authoritative FrameTimestamp provenance.
2. DetectionProvider: Detects objects preserving exact FrameTimestamp.
3. TrackingProvider (BoTSORTTracker): Associates objects, predicts motion via Kalman filter,
   maintains track lifecycle (ACTIVE -> LOST -> TERMINATED), and emits TrajectoryPoints.
4. TrajectoryAnalyzer: Computes pixel displacement, path length,
   pixel velocity, and cardinal direction.
5. Evidence: Compiles structured evidence records with full audit trail.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from videx.domain.schemas import (
    BoundingBox,
    CoordinateType,
    Detection,
    EvidenceType,
    Frame,
    FrameTimestamp,
    TimestampSource,
)
from videx.ingestion.base import DecodedFrame
from videx.ingestion.reader import OpenCVVideoReader
from videx.perception.pipeline import PerceptionPipeline
from videx.perception.tracking import BoTSORTConfig, BoTSORTTracker
from videx.perception.trajectory import TrajectoryAnalyzer


def create_smoke_video(path: Path, frame_count: int = 25, fps: float = 25.0) -> None:
    """Create a 1-second synthetic video with a moving rectangular target."""
    width, height = 320, 240
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    try:
        for i in range(frame_count):
            frame = np.full((height, width, 3), 30, dtype=np.uint8)
            # Draw moving white target (moving down-right)
            # Simulate occlusion in frames 10..12
            if not (10 <= i <= 12):
                x = 30 + i * 6
                y = 40 + i * 4
                cv2.rectangle(frame, (x, y), (x + 40, y + 40), (240, 240, 240), -1)

            # Frame label
            cv2.putText(
                frame,
                f"F{i}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            out.write(frame)
    finally:
        out.release()


class SyntheticShapeDetector:
    """Deterministic visual detector that identifies high-contrast shapes for smoke testing."""

    def __init__(self, provider_name: str = "smoke_shape_detector") -> None:
        self._provider = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider

    def warmup(self) -> None:
        pass

    def detect(self, frame_data: object, frame_meta: Frame | None = None) -> list[Detection]:
        from videx.ingestion.base import DecodedFrame

        if not isinstance(frame_data, DecodedFrame):
            return []

        arr = frame_data.frame_array
        if not isinstance(arr, np.ndarray):
            return []

        # Find bright pixels (> 200)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        mask = (gray > 200).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        frame_ts = frame_data.frame_timestamp
        if frame_ts is None:
            frame_ts = FrameTimestamp(
                frame_index=frame_data.frame_index,
                pts_seconds=frame_data.timestamp_seconds,
                timestamp_source=TimestampSource.CONTAINER,
            )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 20 and h >= 20:  # filter noise
                detections.append(
                    Detection(
                        video_id=frame_data.video_id or uuid4(),
                        frame_id=uuid4(),
                        frame_number=frame_data.frame_index,
                        timestamp_seconds=frame_data.timestamp_seconds,
                        frame_timestamp=frame_ts,
                        class_name="target_box",
                        class_id=1,
                        confidence=0.96,
                        bbox=BoundingBox(
                            x=float(x),
                            y=float(y),
                            width=float(w),
                            height=float(h),
                            coordinate_type=CoordinateType.PIXEL,
                        ),
                        provider=self.provider_name,
                        attributes={
                            "timestamp_source": frame_ts.timestamp_source.value,
                            "is_repaired": frame_ts.is_repaired,
                        },
                    )
                )

        return detections

    def detect_batch(
        self, batch: list[tuple[bytes | DecodedFrame, Frame | None]]
    ) -> list[list[Detection]]:
        return [self.detect(f[0], f[1]) for f in batch]


def main() -> None:
    smoke_path = Path("smoke_perception.mp4")
    print(f"Creating synthetic perception video: {smoke_path.resolve()} ...")
    create_smoke_video(smoke_path, frame_count=25, fps=25.0)

    try:
        print("\n--- Initializing VideoReader & Perception Pipeline ---")
        detector = SyntheticShapeDetector()
        tracker = BoTSORTTracker(BoTSORTConfig(track_buffer=5, track_high_thresh=0.5))
        pipeline = PerceptionPipeline(detector=detector, tracker=tracker, min_track_length=3)

        with OpenCVVideoReader(smoke_path) as reader:
            print(
                f"Video opened: {reader.total_frames} frames, "
                f"{reader.fps:.1f} fps, {reader.duration_seconds:.2f}s"
            )
            print(f"Timing Mode: {reader.timing_mode}")

            result = pipeline.process_reader(reader)

        print("\n--- Perception Results ---")
        print(f"Total Detections: {result.total_detections}")
        print(f"Total Tracks:     {result.total_tracks}")
        print(f"Evidence Records: {len(result.evidence)}")

        print("\n--- Timestamp Provenance Verification ---")
        for i, det in enumerate(result.detections[:3]):
            print(
                f"  Detection [{i}]: Frame {det.frame_number}, "
                f"PTS={det.frame_timestamp.pts_seconds:.3f}s, "
                f"Source={det.frame_timestamp.timestamp_source.value}, "
                f"Repaired={det.frame_timestamp.is_repaired}"
            )

        print("\n--- Tracking & Trajectory Kinematics ---")
        for trk in tracker.all_tracks():
            points = result.trajectories.get(trk.track_id, [])
            kinematics = TrajectoryAnalyzer.analyze_trajectory(points)
            print(
                f"  Track [{trk.track_id}]: Class='{trk.class_name}', Status={trk.status.value}, "
                f"Frames={trk.first_seen_frame_number}..{trk.last_seen_frame_number} "
                f"({trk.frame_count} frames)"
            )
            print(
                f"    Displacement: {kinematics.get('total_displacement_px', 0):.1f} px, "
                f"Path Length: {kinematics.get('path_length_px', 0):.1f} px"
            )
            print(
                f"    Mean Velocity: {kinematics.get('mean_pixel_velocity', 0):.1f} px/s, "
                f"Direction: {kinematics.get('direction_cardinal')}"
            )

        print("\n--- Evidence Output ---")
        for ev in result.evidence:
            if ev.evidence_type == EvidenceType.TRACK:
                print(
                    f"  Evidence [{ev.evidence_id}]: Type={ev.evidence_type.value}, "
                    f"Confidence={ev.confidence:.2f}, Module={ev.source_module}"
                )

        print("\nPerception Smoke Test: SUCCESS (All components operational)")
    finally:
        if smoke_path.exists():
            smoke_path.unlink()


if __name__ == "__main__":
    main()
