"""Video metadata extraction functions.

Extracts technical stream metadata, duration, frame rates, codecs, and audio presence
using ffprobe when available, with robust OpenCV fallback.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, Field

from videx.ingestion.base import CorruptVideoError


class RawVideoMetadata(BaseModel):
    """Raw technical metadata extracted from video streams and containers."""

    filename: str
    filesize_bytes: int
    duration_seconds: float
    fps: float
    total_frames: int
    width: int
    height: int
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False
    is_vfr: bool = False
    r_frame_rate: float | None = None
    avg_frame_rate: float | None = None
    mime_type: str | None = None
    container_format: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _parse_rational_fps(rate_str: str) -> float | None:
    """Parse rational frame rates such as '30000/1001' or '25/1' into a float."""
    if not rate_str or rate_str == "0/0":
        return None
    try:
        if "/" in rate_str:
            num_str, denom_str = rate_str.split("/", 1)
            num = float(num_str)
            denom = float(denom_str)
            if denom > 0:
                return num / denom
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return None


def extract_metadata_ffprobe(video_path: Path) -> RawVideoMetadata | None:
    """Attempt to extract rich metadata using ffprobe via subprocess.

    Returns None if ffprobe is not installed on PATH or if execution fails.
    """
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        return None

    cmd = [
        ffprobe_cmd,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None

        probe_data: dict[str, Any] = json.loads(proc.stdout)
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return None

    streams = probe_data.get("streams", [])
    fmt = probe_data.get("format", {})

    video_stream: dict[str, Any] | None = None
    audio_stream: dict[str, Any] | None = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        return None

    # Width and height
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    # Frame rate extraction & VFR detection heuristic
    # In ffprobe stream headers, r_frame_rate represents the lowest framerate with which all
    # timestamps can be represented, while avg_frame_rate is the average framerate.
    # IMPORTANT: Comparing r_frame_rate and avg_frame_rate is an ingestion-time heuristic only
    # and is NOT authoritative. Multiplexers often populate both with identical nominal values
    # even on VFR streams, or report fractional divergence due to container overhead.
    # Authoritative VFR classification requires packet/frame presentation timestamp (PTS)
    # variance analysis via PyAVTimestampIndex / detect_vfr_from_timestamps.
    r_fps: float | None = _parse_rational_fps(str(video_stream.get("r_frame_rate", "")))
    avg_fps: float | None = _parse_rational_fps(str(video_stream.get("avg_frame_rate", "")))

    fps: float | None = avg_fps if (avg_fps and avg_fps > 0) else r_fps
    if fps is None or fps <= 0:
        fps = 25.0  # safe default if undetermined

    is_vfr = False
    if r_fps is not None and avg_fps is not None and r_fps > 0 and avg_fps > 0:
        discrepancy = abs(r_fps - avg_fps) / avg_fps
        if discrepancy > 0.005:  # > 0.5% divergence indicates likely VFR
            is_vfr = True

    # Duration extraction
    duration: float = 0.0
    if "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except (ValueError, TypeError):
            duration = 0.0
    if duration <= 0.0 and "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (ValueError, TypeError):
            duration = 0.0

    # Total frames
    total_frames: int = 0
    if "nb_frames" in video_stream:
        try:
            total_frames = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            total_frames = 0
    if total_frames <= 0 and duration > 0 and fps > 0:
        total_frames = int(round(duration * fps))

    # In case duration is 0 but we have frames
    if duration <= 0.0 and total_frames > 0 and fps > 0:
        duration = total_frames / fps

    video_codec = video_stream.get("codec_name")
    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None

    filesize = video_path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(video_path))

    return RawVideoMetadata(
        filename=video_path.name,
        filesize_bytes=filesize,
        duration_seconds=duration,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        has_audio=has_audio,
        is_vfr=is_vfr,
        r_frame_rate=r_fps,
        avg_frame_rate=avg_fps,
        mime_type=mime_type or "video/mp4",
        container_format=fmt.get("format_name"),
        metadata={
            "ffprobe_streams": len(streams),
            "bitrate": fmt.get("bit_rate"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "r_frame_rate": str(video_stream.get("r_frame_rate", "")),
            "avg_frame_rate": str(video_stream.get("avg_frame_rate", "")),
        },
    )


def extract_metadata_opencv(video_path: Path) -> RawVideoMetadata:
    """Extract metadata using OpenCV VideoCapture as a fallback or verification."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CorruptVideoError(video_path, "OpenCV failed to open video file")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise CorruptVideoError(video_path, f"Invalid dimensions: {width}x{height}")

        if fps <= 0 or fps > 1000.0:
            fps = 25.0

        if total_frames < 0:
            total_frames = 0

        duration = total_frames / fps if (total_frames > 0 and fps > 0) else 0.0

        # Attempt to decode fourcc
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_chars = [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
        codec_name: str | None = "".join(fourcc_chars).strip()
        if not codec_name or not codec_name.isprintable():
            codec_name = None

        filesize = video_path.stat().st_size
        mime_type, _ = mimetypes.guess_type(str(video_path))

        return RawVideoMetadata(
            filename=video_path.name,
            filesize_bytes=filesize,
            duration_seconds=duration,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            video_codec=codec_name,
            audio_codec=None,
            has_audio=False,  # OpenCV cannot inspect audio
            is_vfr=False,  # OpenCV cannot detect VFR streams
            r_frame_rate=fps,
            avg_frame_rate=fps,
            mime_type=mime_type or "video/mp4",
            container_format=video_path.suffix.lstrip(".").lower(),
            metadata={"source": "opencv"},
        )
    finally:
        cap.release()


def extract_video_metadata(video_path: Path) -> RawVideoMetadata:
    """Extract comprehensive technical video metadata.

    Prefers ffprobe when available for audio streams, container formats,
    and accurate codecs. Falls back seamlessly to OpenCV VideoCapture.
    """
    ffprobe_meta = extract_metadata_ffprobe(video_path)
    if ffprobe_meta is not None:
        return ffprobe_meta

    return extract_metadata_opencv(video_path)
