"""VIDEX domain schemas.

This module defines the complete typed domain model for the VIDEX pipeline.
All entities use Pydantic v2 models with UUID primary keys and UTC datetimes,
designed so that future database persistence (e.g., SQLAlchemy / SQLModel)
does not require structural changes to these schemas.

Entity hierarchy::

    Video
     └── Scene          (video_id FK)
     └── Frame          (video_id FK)
          └── Detection (frame_id FK)
          └── OCRObservation (frame_id FK)
    Track               (video_id FK, references Detection IDs)
     └── TrajectoryPoint (track_id FK, frame_id FK)
    AudioSegment        (video_id FK)
    Event               (video_id FK, references Track IDs)
    Evidence            (video_id FK, optional frame_id, track_id)

Design notes:
- UUIDs are used as PKs everywhere for database-friendliness.
- ``BoundingBox`` supports both pixel and normalised coordinate systems.
- ``Evidence`` is the canonical output record — all pipeline stages write Evidence.
- ``raw_payload`` on Evidence allows provider-specific fields without schema churn.
- ``model_config = ConfigDict(frozen=True)`` is used on leaf value types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Temporal Provenance ───────────────────────────────────────────────────


class TimestampSource(StrEnum):
    """Source provenance for an individual frame presentation timestamp."""

    CONTAINER = "container"  # Extracted directly from container/packet header
    DERIVED = "derived"  # Computed mathematically or repaired via interpolation/extrapolation


@dataclass(frozen=True)
class FrameTimestamp:
    """Detailed temporal provenance for an individual frame."""

    frame_index: int
    pts_seconds: float
    timestamp_source: TimestampSource
    is_repaired: bool = False
    repair_reason: str | None = None
    original_pts_seconds: float | None = None

    @property
    def source(self) -> TimestampSource:
        """Alias for timestamp_source."""
        return self.timestamp_source

    def __float__(self) -> float:
        return self.pts_seconds

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.pts_seconds == float(other)
        if isinstance(other, FrameTimestamp):
            return self.pts_seconds == other.pts_seconds and self.frame_index == other.frame_index
        return False

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (int, float, FrameTimestamp)):
            return self.pts_seconds < float(other)
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, (int, float, FrameTimestamp)):
            return self.pts_seconds <= float(other)
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (int, float, FrameTimestamp)):
            return self.pts_seconds > float(other)
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, (int, float, FrameTimestamp)):
            return self.pts_seconds >= float(other)
        return NotImplemented

    def __sub__(self, other: object) -> float:
        return self.pts_seconds - float(other)  # type: ignore[arg-type]

    def __rsub__(self, other: object) -> float:
        return float(other) - self.pts_seconds  # type: ignore[arg-type]

    def __add__(self, other: object) -> float:
        return self.pts_seconds + float(other)  # type: ignore[arg-type]

    def __radd__(self, other: object) -> float:
        return float(other) + self.pts_seconds  # type: ignore[arg-type]

# ── Coordinate system ──────────────────────────────────────────────────────


class CoordinateType(StrEnum):
    """Specifies how BoundingBox coordinates are expressed."""

    PIXEL = "pixel"
    """Coordinates are in absolute pixels relative to the frame dimensions."""

    NORMALIZED = "normalized"
    """Coordinates are in [0, 1] relative to frame width/height."""


class BoundingBox(BaseModel):
    """Axis-aligned bounding box.

    Top-left corner is (x, y); the box extends to (x + width, y + height).
    The coordinate system is indicated by ``coordinate_type``.
    """

    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="Top-left x coordinate")
    y: float = Field(..., description="Top-left y coordinate")
    width: float = Field(..., gt=0.0, description="Box width (must be positive)")
    height: float = Field(..., gt=0.0, description="Box height (must be positive)")
    coordinate_type: CoordinateType = Field(
        default=CoordinateType.PIXEL,
        description="Pixel or normalized coordinates",
    )

    @property
    def x2(self) -> float:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center_x(self) -> float:
        """Horizontal centre of the box."""
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        """Vertical centre of the box."""
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height


# ── Video ──────────────────────────────────────────────────────────────────


class Video(BaseModel):
    """A registered video file.

    Created when a video is submitted to the ingestion pipeline.
    All other entities reference this record via ``video_id``.
    """

    video_id: UUID = Field(default_factory=uuid4, description="Unique video identifier")
    source_path: str = Field(..., description="Original path or URL of the video file")
    duration_seconds: float | None = Field(
        default=None, ge=0.0, description="Total duration in seconds"
    )
    fps: float | None = Field(
        default=None, gt=0.0, description="Frames per second of the source video"
    )
    width: int | None = Field(default=None, gt=0, description="Frame width in pixels")
    height: int | None = Field(default=None, gt=0, description="Frame height in pixels")
    video_codec: str | None = Field(
        default=None, description="Video codec identifier (e.g., 'h264')"
    )
    total_frames: int | None = Field(default=None, ge=0, description="Total frame count")
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the video was registered",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (container format, audio tracks, etc.)",
    )


# ── Scene ──────────────────────────────────────────────────────────────────


class Scene(BaseModel):
    """A contiguous scene segment within a video.

    Produced by the scene-change detection stage. A video is partitioned
    into one or more non-overlapping scenes.
    """

    scene_id: UUID = Field(default_factory=uuid4)
    video_id: UUID = Field(..., description="Parent video")
    scene_index: int = Field(..., ge=0, description="Sequential scene number (0-based)")
    start_frame_number: int = Field(..., ge=0)
    end_frame_number: int = Field(..., ge=0)
    start_timestamp_seconds: float = Field(..., ge=0.0)
    end_timestamp_seconds: float = Field(..., ge=0.0)

    @property
    def duration_seconds(self) -> float:
        """Duration of the scene in seconds."""
        return self.end_timestamp_seconds - self.start_timestamp_seconds

    @property
    def frame_count(self) -> int:
        """Number of frames in the scene."""
        return self.end_frame_number - self.start_frame_number + 1


# ── Frame ──────────────────────────────────────────────────────────────────


class Frame(BaseModel):
    """A single decoded video frame.

    ``frame_data_path`` optionally points to a stored JPEG/PNG if frames
    are persisted to disk. It is ``None`` when frames are processed
    in-memory without caching.
    """

    frame_id: UUID = Field(default_factory=uuid4)
    video_id: UUID = Field(..., description="Parent video")
    scene_id: UUID | None = Field(
        default=None, description="Scene this frame belongs to (if detected)"
    )
    frame_number: int = Field(..., ge=0, description="Zero-based frame index")
    timestamp_seconds: float = Field(..., ge=0.0, description="Frame position in the video")
    frame_timestamp: FrameTimestamp | None = Field(
        default=None,
        description="Authoritative temporal provenance from DecodedFrame",
    )
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    frame_data_path: str | None = Field(
        default=None,
        description="Filesystem path to the stored frame image (optional)",
    )


# ── Detection ──────────────────────────────────────────────────────────────


class Detection(BaseModel):
    """A single object detection result from one frame.

    Produced by a ``DetectionProvider``. Multiple detections may exist
    per frame (one per detected object instance).
    """

    detection_id: UUID = Field(default_factory=uuid4)
    frame_id: UUID = Field(..., description="Frame this detection was made in")
    video_id: UUID = Field(..., description="Parent video")
    frame_number: int = Field(default=0, ge=0, description="Sequential frame index")
    timestamp_seconds: float = Field(..., ge=0.0)
    frame_timestamp: FrameTimestamp = Field(
        default=None,  # type: ignore[assignment]
        description="Authoritative temporal provenance from DecodedFrame",
    )
    class_name: str = Field(..., description="Detected object class label")
    class_id: int = Field(..., ge=0, description="Integer class index from the model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box of the detected object")
    mask: list[list[float]] | None = Field(
        default=None,
        description="Optional polygon segmentation mask coordinates [[x1, y1], [x2, y2], ...]",
    )
    provider: str = Field(
        ...,
        description="Provider key that produced this detection (e.g., 'yolo26', 'mock_detector')",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific attributes (e.g. timestamp_source, is_repaired)",
    )

    @model_validator(mode="before")
    @classmethod
    def _ensure_frame_timestamp(cls, data: object) -> object:
        if isinstance(data, dict):
            if data.get("frame_timestamp") is None:
                frame_num = int(data.get("frame_number", 0))
                ts_sec = float(data.get("timestamp_seconds", 0.0))
                data["frame_timestamp"] = FrameTimestamp(
                    frame_index=frame_num,
                    pts_seconds=ts_sec,
                    timestamp_source=TimestampSource.DERIVED,
                )
            elif (
                "timestamp_seconds" not in data
                and isinstance(data["frame_timestamp"], FrameTimestamp)
            ):
                data["timestamp_seconds"] = data["frame_timestamp"].pts_seconds
        return data


# ── Track ──────────────────────────────────────────────────────────────────


class TrackStatus(StrEnum):
    """Lifecycle state of an object track."""

    ACTIVE = "active"
    """Object is actively tracked and detected in current window."""

    LOST = "lost"
    """Object is temporarily unobserved/occluded."""

    TERMINATED = "terminated"
    """Track is closed/completed (object permanently exited scene)."""


class Track(BaseModel):
    """A multi-frame object track — one persistent identity across frames.

    Produced by a ``TrackingProvider`` by associating detections over time.
    ``detection_ids`` references the individual Detection records.
    """

    track_id: UUID = Field(default_factory=uuid4)
    video_id: UUID = Field(..., description="Parent video")
    class_name: str = Field(..., description="Object class (from associated detections)")
    class_id: int = Field(default=0, ge=0, description="Integer class label index")
    first_seen_frame_number: int = Field(..., ge=0)
    last_seen_frame_number: int = Field(..., ge=0)
    first_seen_timestamp_seconds: float = Field(..., ge=0.0)
    last_seen_timestamp_seconds: float = Field(..., ge=0.0)
    detection_ids: list[UUID] = Field(
        default_factory=list,
        description="Ordered list of Detection IDs comprising this track",
    )
    provider: str = Field(..., description="Tracker backend that produced this track")
    status: TrackStatus = Field(
        default=TrackStatus.TERMINATED,
        description="Current lifecycle status of the track",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall track confidence or mean detection confidence",
    )
    start_bbox: BoundingBox | None = Field(
        default=None,
        description="Initial bounding box at appearance (first_seen)",
    )
    end_bbox: BoundingBox | None = Field(
        default=None,
        description="Terminal bounding box at disappearance (last_seen)",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific attributes or lifecycle metadata",
    )

    @property
    def duration_seconds(self) -> float:
        """Seconds the object was tracked."""
        return self.last_seen_timestamp_seconds - self.first_seen_timestamp_seconds

    @property
    def frame_count(self) -> int:
        """Number of frames this track spans."""
        return self.last_seen_frame_number - self.first_seen_frame_number + 1


# ── Trajectory ─────────────────────────────────────────────────────────────


class TrajectoryPoint(BaseModel):
    """One spatial observation along a track's path.

    A sequence of ``TrajectoryPoint`` records for a given ``track_id`` forms
    the full spatial trajectory of a tracked object.
    """

    model_config = ConfigDict(frozen=True)

    track_id: UUID = Field(..., description="Parent track")
    frame_id: UUID = Field(..., description="Frame this point was observed in")
    frame_number: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0)
    frame_timestamp: FrameTimestamp = Field(
        default=None,  # type: ignore[assignment]
        description="Authoritative temporal provenance from DecodedFrame",
    )
    bbox: BoundingBox = Field(..., description="Object position at this point")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _ensure_frame_timestamp(cls, data: object) -> object:
        if isinstance(data, dict):
            if data.get("frame_timestamp") is None:
                frame_num = int(data.get("frame_number", 0))
                ts_sec = float(data.get("timestamp_seconds", 0.0))
                data["frame_timestamp"] = FrameTimestamp(
                    frame_index=frame_num,
                    pts_seconds=ts_sec,
                    timestamp_source=TimestampSource.DERIVED,
                )
            elif (
                "timestamp_seconds" not in data
                and isinstance(data["frame_timestamp"], FrameTimestamp)
            ):
                data["timestamp_seconds"] = data["frame_timestamp"].pts_seconds
        return data


# ── OCR ────────────────────────────────────────────────────────────────────


class OCRObservation(BaseModel):
    """Text detected and recognised in a single video frame.

    Produced by an ``OCRProvider``.
    """

    ocr_id: UUID = Field(default_factory=uuid4)
    frame_id: UUID = Field(..., description="Frame the text was found in")
    video_id: UUID = Field(..., description="Parent video")
    timestamp_seconds: float = Field(..., ge=0.0)
    text: str = Field(..., description="Recognised text string")
    language: str | None = Field(
        default=None,
        description="Detected language ISO code (e.g., 'hi', 'en')",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox | None = Field(
        default=None,
        description="Location of the text in the frame (if available)",
    )
    provider: str = Field(..., description="OCR backend that produced this result")


# ── Audio / ASR ────────────────────────────────────────────────────────────


class AudioSegment(BaseModel):
    """A time-aligned speech transcript segment.

    Produced by an ``ASRProvider`` from the audio track of a video.
    Multiple segments cover the full audio timeline.
    """

    segment_id: UUID = Field(default_factory=uuid4)
    video_id: UUID = Field(..., description="Parent video")
    start_timestamp_seconds: float = Field(..., ge=0.0)
    end_timestamp_seconds: float = Field(..., ge=0.0)
    transcript: str = Field(..., description="Recognised speech text")
    language: str | None = Field(default=None, description="Detected language ISO code")
    confidence: float = Field(..., ge=0.0, le=1.0)
    speaker_id: str | None = Field(
        default=None,
        description="Speaker diarisation label (if available)",
    )
    provider: str = Field(..., description="ASR backend that produced this segment")

    @property
    def duration_seconds(self) -> float:
        """Duration of this audio segment."""
        return self.end_timestamp_seconds - self.start_timestamp_seconds


# ── Event ──────────────────────────────────────────────────────────────────


class EventType(StrEnum):
    """Taxonomy of detectable video events."""

    OBJECT_APPEARED = "object_appeared"
    """A new object entered the scene (or came into frame)."""

    OBJECT_DISAPPEARED = "object_disappeared"
    """A tracked object left the scene (track ended)."""

    OBJECT_ENTERED_ZONE = "object_entered_zone"
    """A tracked object entered a defined region of interest."""

    OBJECT_LEFT_ZONE = "object_left_zone"
    """A tracked object exited a defined region of interest."""

    LOITERING = "loitering"
    """An object dwelled in a zone longer than a configured threshold."""

    CROWD_FORMATION = "crowd_formation"
    """Multiple objects converged in proximity."""

    ANOMALY = "anomaly"
    """A statistically unusual pattern was detected."""

    CUSTOM = "custom"
    """User-defined event type — description in ``Event.description``."""


class Event(BaseModel):
    """A detected semantic event in the video.

    Events are produced by an ``EventDetector`` from the outputs of upstream
    pipeline stages (tracks, trajectories, detections, etc.).
    """

    event_id: UUID = Field(default_factory=uuid4)
    video_id: UUID = Field(..., description="Parent video")
    event_type: EventType = Field(..., description="Category of the event")
    start_timestamp_seconds: float = Field(..., ge=0.0, description="When the event began")
    end_timestamp_seconds: float | None = Field(
        default=None,
        description="When the event ended (None if instantaneous)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    track_ids: list[UUID] = Field(
        default_factory=list,
        description="Tracks involved in this event",
    )
    description: str = Field(..., description="Human-readable description of the event")
    zone_name: str | None = Field(
        default=None,
        description="Named zone involved (if applicable)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-type-specific extra data",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this event record was created",
    )


# ── Evidence ───────────────────────────────────────────────────────────────


class EvidenceType(StrEnum):
    """The category of observation represented by an Evidence record."""

    DETECTION = "detection"
    """Derived from an object detection result."""

    TRACK = "track"
    """Derived from a multi-frame object track."""

    TRAJECTORY = "trajectory"
    """Derived from spatial movement analysis."""

    OCR = "ocr"
    """Derived from text recognised in a frame."""

    AUDIO_TRANSCRIPT = "audio_transcript"
    """Derived from an ASR transcription."""

    VLM_CAPTION = "vlm_caption"
    """Derived from a VLM frame/scene caption."""

    EVENT = "event"
    """Derived from a detected semantic event."""

    SCENE_CHANGE = "scene_change"
    """Records a detected scene boundary."""


class Evidence(BaseModel):
    """The canonical pipeline output record.

    Every stage of the VIDEX pipeline writes structured ``Evidence`` records.
    These are the primary artefacts stored in the evidence database and
    presented in the Intelligence UI.

    Design goals:
    - Flat enough to map 1:1 to a database row without joins.
    - Rich enough to reconstruct the provenance chain.
    - Extensible via ``raw_payload`` without schema migrations.
    """

    evidence_id: UUID = Field(
        default_factory=uuid4,
        description="Unique evidence record identifier (PK in DB)",
    )
    evidence_type: EvidenceType = Field(..., description="Category of this evidence record")
    source_module: str = Field(
        ...,
        description=(
            "Identifier of the pipeline module that generated this record "
            "(e.g., 'yolov8_detector', 'bytetrack', 'easyocr_hi')"
        ),
    )

    # ── Spatial / temporal context ─────────────────────────────
    video_id: UUID = Field(..., description="Parent video")
    frame_id: UUID | None = Field(
        None,
        description="Frame this evidence is anchored to (None for track/audio evidence)",
    )
    timestamp_seconds: float | None = Field(
        None,
        ge=0.0,
        description="Video timestamp in seconds (None if not frame-anchored)",
    )
    bbox: BoundingBox | None = Field(
        None,
        description="Spatial extent in the frame (None for non-spatial evidence)",
    )

    # ── Tracking context ──────────────────────────────────────
    track_id: UUID | None = Field(
        None,
        description="Associated track ID (if this evidence is linked to a track)",
    )

    # ── Quality ───────────────────────────────────────────────
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0, 1] from the source module",
    )

    # ── Content ───────────────────────────────────────────────
    description: str = Field(
        ...,
        description="Human-readable summary of this evidence record",
    )
    raw_payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific structured data. "
            "Allows schema extension without migration. "
            "Example: {'class_name': 'person', 'class_id': 0, 'provider_model': 'yolov8l'}"
        ),
    )

    # ── Provenance ────────────────────────────────────────────
    supporting_observation_ids: list[UUID] = Field(
        default_factory=list,
        description=(
            "IDs of upstream records (Detection, OCRObservation, AudioSegment, etc.) "
            "that this Evidence was derived from"
        ),
    )

    # ── Record metadata ───────────────────────────────────────
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this Evidence record was created",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for filtering/search (e.g., ['person', 'zone-A'])",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata (deployment-specific)",
    )


# ── VideoManifest ──────────────────────────────────────────────────────────


class VideoManifest(BaseModel):
    """Structured manifest produced upon validating and ingesting a video file.

    Encapsulates all verified technical metadata, frame timing information,
    audio availability, and detected scene boundaries needed by downstream
    perception and temporal intelligence engines.
    """

    video_id: UUID = Field(default_factory=uuid4, description="Unique video identifier")
    source_path: str = Field(..., description="Canonical path or URI to the video source file")
    filename: str = Field(..., description="Base filename of the video")
    filesize_bytes: int = Field(..., ge=0, description="Size of the video file in bytes")
    mime_type: str | None = Field(
        default=None, description="MIME type if identifiable (e.g., 'video/mp4')"
    )

    # Technical metadata
    duration_seconds: float = Field(..., ge=0.0, description="Accurate video duration in seconds")
    fps: float = Field(..., gt=0.0, description="Nominal frame rate (frames per second)")
    total_frames: int = Field(..., ge=0, description="Total number of video frames")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    video_codec: str | None = Field(
        default=None, description="Video stream codec identifier (e.g., 'h264')"
    )
    is_vfr: bool = Field(
        default=False,
        description="Whether the video has variable frame rate (VFR)",
    )
    timing_mode: str = Field(
        default="exact",
        description="Timing policy mode used during ingestion ('exact' or 'fast')",
    )
    is_vfr_hint: bool | None = Field(
        default=None,
        description="Preliminary container header hint regarding variable frame rate",
    )

    # Audio availability
    has_audio: bool = Field(default=False, description="Whether the video contains an audio stream")
    audio_codec: str | None = Field(
        default=None, description="Audio stream codec identifier (e.g., 'aac')"
    )

    # Scene boundaries
    scenes: list[Scene] = Field(
        default_factory=list,
        description="Contiguous scene segments detected in the video",
    )

    # Ingestion record
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this manifest was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional technical, container, or stream metadata",
    )

    def to_video(self) -> Video:
        """Convert this manifest into a database-compatible Video entity."""
        return Video(
            video_id=self.video_id,
            source_path=self.source_path,
            duration_seconds=self.duration_seconds,
            fps=self.fps,
            width=self.width,
            height=self.height,
            video_codec=self.video_codec,
            total_frames=self.total_frames,
            ingested_at=self.ingested_at,
            metadata={
                **self.metadata,
                "filename": self.filename,
                "filesize_bytes": self.filesize_bytes,
                "mime_type": self.mime_type,
                "is_vfr": self.is_vfr,
                "timing_mode": self.timing_mode,
                "is_vfr_hint": self.is_vfr_hint,
                "has_audio": self.has_audio,
                "audio_codec": self.audio_codec,
                "scene_count": len(self.scenes),
            },
        )

