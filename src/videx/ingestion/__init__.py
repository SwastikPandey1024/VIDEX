"""VIDEX video ingestion package.

Provides video validation, metadata extraction, deterministic frame access,
scene detection, and structured VideoManifest generation.
"""

from videx.ingestion.base import (
    CorruptVideoError,
    DecodedFrame,
    FrameAccessError,
    FrameOutOfBoundsError,
    IngestionError,
    SceneDetector,
    TimestampIndex,
    TimestampOutOfBoundsError,
    UnsupportedContainerError,
    VideoNotFoundError,
    VideoReader,
    VideoValidationError,
)
from videx.ingestion.manifest import build_manifest, format_manifest_summary
from videx.ingestion.metadata import (
    RawVideoMetadata,
    extract_metadata_ffprobe,
    extract_metadata_opencv,
    extract_video_metadata,
)
from videx.ingestion.reader import OpenCVVideoReader
from videx.ingestion.scenes import PySceneDetectDetector, SingleSceneDetector
from videx.ingestion.service import VideoIngestionService
from videx.ingestion.timing import (
    CFRTimestampIndex,
    ExplicitTimestampIndex,
    PyAVTimestampIndex,
    create_timestamp_index,
    detect_vfr_from_timestamps,
    sanitize_timestamps,
)
from videx.ingestion.validation import DEFAULT_SUPPORTED_EXTENSIONS, validate_video_file

__all__ = [
    "CFRTimestampIndex",
    "CorruptVideoError",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "DecodedFrame",
    "ExplicitTimestampIndex",
    "FrameAccessError",
    "FrameOutOfBoundsError",
    "IngestionError",
    "OpenCVVideoReader",
    "PyAVTimestampIndex",
    "PySceneDetectDetector",
    "RawVideoMetadata",
    "SceneDetector",
    "SingleSceneDetector",
    "TimestampIndex",
    "TimestampOutOfBoundsError",
    "UnsupportedContainerError",
    "VideoIngestionService",
    "VideoNotFoundError",
    "VideoReader",
    "VideoValidationError",
    "build_manifest",
    "create_timestamp_index",
    "detect_vfr_from_timestamps",
    "extract_metadata_ffprobe",
    "extract_metadata_opencv",
    "extract_video_metadata",
    "format_manifest_summary",
    "sanitize_timestamps",
    "validate_video_file",
]
