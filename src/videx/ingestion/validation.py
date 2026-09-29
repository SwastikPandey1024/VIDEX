"""Video file validation functions.

Performs existence, container extension, file size, and header readability checks.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from videx.ingestion.base import (
    CorruptVideoError,
    UnsupportedContainerError,
    VideoNotFoundError,
)

DEFAULT_SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".flv",
    ".ts",
    ".m4v",
)


def validate_video_file(
    path: Path | str,
    supported_extensions: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Validate that a path points to an existing, supported, readable video file.

    Args:
        path: Path to the video file to validate.
        supported_extensions: Allowed file extensions (defaults to standard containers).

    Returns:
        Resolved absolute Path to the validated file.

    Raises:
        VideoNotFoundError: If the file does not exist on disk or is a directory.
        UnsupportedContainerError: If the extension is not in supported_extensions.
        CorruptVideoError: If file size is 0 bytes or container cannot be opened.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise VideoNotFoundError(file_path)

    if not file_path.is_file():
        raise VideoNotFoundError(f"Path is not a regular file: {file_path}")

    # Check file size
    try:
        size = file_path.stat().st_size
        if size == 0:
            raise CorruptVideoError(file_path, "Video file has 0 bytes (empty file)")
    except OSError as err:
        raise CorruptVideoError(file_path, f"Failed to access file attributes: {err}") from err

    # Check extension
    default_exts = DEFAULT_SUPPORTED_EXTENSIONS
    active_exts = supported_extensions if supported_extensions is not None else default_exts
    extensions = tuple(ext.lower() for ext in active_exts)
    suffix = file_path.suffix.lower()
    if suffix not in extensions:
        raise UnsupportedContainerError(file_path, suffix, list(extensions))

    # Fast header / container readability probe using OpenCV
    cap = cv2.VideoCapture(str(file_path))
    try:
        if not cap.isOpened():
            raise CorruptVideoError(
                file_path,
                "Unable to open video stream; container or codec unreadable",
            )
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width <= 0 or height <= 0:
            raise CorruptVideoError(file_path, f"Invalid video dimensions: {width}x{height}")
    finally:
        cap.release()

    return file_path.resolve()
