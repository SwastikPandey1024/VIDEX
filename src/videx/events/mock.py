"""Mock fixtures and synthetic data generators for event testing."""

from __future__ import annotations

from uuid import UUID, uuid4

from videx.domain.schemas import (
    BoundingBox,
    FrameTimestamp,
    TextObservation,
    TimestampSource,
    Track,
    TrajectoryPoint,
    TranscriptSegment,
    WordTimestamp,
)
from videx.events.spatial import SpatialZone


def create_mock_track(
    video_id: UUID | None = None,
    track_id: UUID | None = None,
    class_name: str = "car",
    start_frame: int = 0,
    end_frame: int = 24,
    start_time: float = 0.0,
    end_time: float = 1.0,
    confidence: float = 0.92,
) -> Track:
    """Generate a valid Track domain record."""
    t_id = track_id or uuid4()
    v_id = video_id or uuid4()
    return Track(
        track_id=t_id,
        video_id=v_id,
        class_id=2,
        class_name=class_name,
        confidence=confidence,
        first_seen_frame_number=start_frame,
        last_seen_frame_number=end_frame,
        first_seen_timestamp_seconds=start_time,
        last_seen_timestamp_seconds=end_time,
        frame_count=end_frame - start_frame + 1,
        detection_ids=[uuid4() for _ in range(end_frame - start_frame + 1)],
        start_bbox=BoundingBox(x=100.0, y=100.0, width=50.0, height=50.0),
        end_bbox=BoundingBox(x=200.0, y=200.0, width=50.0, height=50.0),
        provider="mock_tracker",
    )


def create_mock_trajectory(
    track_id: UUID,
    start_x: float = 100.0,
    start_y: float = 100.0,
    end_x: float = 200.0,
    end_y: float = 200.0,
    frames: int = 10,
    fps: float = 25.0,
) -> list[TrajectoryPoint]:
    """Generate a straight-line synthetic trajectory."""
    points: list[TrajectoryPoint] = []
    dx = (end_x - start_x) / max(1, frames - 1)
    dy = (end_y - start_y) / max(1, frames - 1)

    for i in range(frames):
        ts = i / fps
        x = start_x + i * dx
        y = start_y + i * dy
        pt = TrajectoryPoint(
            track_id=track_id,
            frame_id=uuid4(),
            frame_number=i,
            timestamp_seconds=ts,
            frame_timestamp=FrameTimestamp(
                frame_index=i,
                pts_seconds=ts,
                timestamp_source=TimestampSource.CONTAINER,
            ),
            bbox=BoundingBox(x=x - 20, y=y - 20, width=40, height=40),
            confidence=0.9,
        )
        points.append(pt)
    return points


def create_mock_text_observation(
    video_id: UUID | None = None,
    text: str = "MH12AB1234",
    start_frame: int = 10,
    end_frame: int = 20,
    start_time: float = 0.4,
    end_time: float = 0.8,
    bbox: BoundingBox | None = None,
) -> TextObservation:
    """Generate a temporally fused TextObservation record."""
    v_id = video_id or uuid4()
    box = bbox or BoundingBox(x=50.0, y=50.0, width=120.0, height=30.0)
    return TextObservation(
        observation_id=uuid4(),
        video_id=v_id,
        text=text,
        normalized_text=text.upper().strip(),
        language="en",
        script="Latn",
        first_seen_frame=start_frame,
        last_seen_frame=end_frame,
        first_seen_timestamp_seconds=start_time,
        last_seen_timestamp_seconds=end_time,
        first_seen_timestamp=FrameTimestamp(
            frame_index=start_frame,
            pts_seconds=start_time,
            timestamp_source=TimestampSource.CONTAINER,
        ),
        last_seen_timestamp=FrameTimestamp(
            frame_index=end_frame,
            pts_seconds=end_time,
            timestamp_source=TimestampSource.CONTAINER,
        ),
        mean_confidence=0.94,
        latest_bbox=box,
        supporting_frames=list(range(start_frame, end_frame + 1)),
        supporting_observation_ids=[uuid4() for _ in range(end_frame - start_frame + 1)],
        provider="mock_ocr",
    )


def create_mock_transcript_segment(
    video_id: UUID | None = None,
    text: str = "Speed limit 40 ahead",
    start_time: float = 1.0,
    end_time: float = 3.0,
    language: str = "en",
) -> TranscriptSegment:
    """Generate a time-aligned TranscriptSegment record."""
    v_id = video_id or uuid4()
    return TranscriptSegment(
        segment_id=uuid4(),
        video_id=v_id,
        start_timestamp_seconds=start_time,
        end_timestamp_seconds=end_time,
        raw_text=text,
        normalized_text=text.strip(),
        language=language,
        confidence=0.91,
        provider="mock_whisper",
        words=[
            WordTimestamp(
                word="Speed",
                start_timestamp_seconds=start_time,
                end_timestamp_seconds=start_time + 0.5,
                confidence=0.95,
            ),
            WordTimestamp(
                word="limit",
                start_timestamp_seconds=start_time + 0.5,
                end_timestamp_seconds=start_time + 1.0,
                confidence=0.94,
            ),
            WordTimestamp(
                word="40",
                start_timestamp_seconds=start_time + 1.0,
                end_timestamp_seconds=start_time + 1.5,
                confidence=0.90,
            ),
            WordTimestamp(
                word="ahead",
                start_timestamp_seconds=start_time + 1.5,
                end_timestamp_seconds=end_time,
                confidence=0.92,
            ),
        ],
    )


def create_mock_zone(
    zone_id: str = "zone_gate_a",
    zone_name: str = "Gate A Entrance",
    x_min: float = 150.0,
    y_min: float = 150.0,
    x_max: float = 300.0,
    y_max: float = 300.0,
) -> SpatialZone:
    """Generate a rectangular SpatialZone."""
    return SpatialZone.from_rectangle(
        zone_id=zone_id,
        zone_name=zone_name,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
