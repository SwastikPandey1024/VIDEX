"""Video manifest generation and formatting utilities."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from videx.domain.schemas import Scene, VideoManifest
from videx.ingestion.metadata import RawVideoMetadata


def build_manifest(
    video_path: Path | str,
    metadata: RawVideoMetadata,
    scenes: list[Scene],
    video_id: UUID | None = None,
    is_vfr: bool | None = None,
    timing_mode: str = "exact",
    is_vfr_hint: bool | None = None,
) -> VideoManifest:
    """Construct a validated VideoManifest from extracted metadata and scenes.

    Args:
        video_path: Resolved path to the video file.
        metadata: Raw technical metadata extracted from the file.
        scenes: Detected scene boundaries.
        video_id: Optional UUID (generates a new UUID4 if None).
        is_vfr: Authoritative VFR classification. If None, defaults to metadata.is_vfr.
        timing_mode: Timing policy mode ("exact" or "fast").
        is_vfr_hint: Preliminary container metadata VFR hint.

    Returns:
        Structured, immutable VideoManifest instance.
    """
    path = Path(video_path)
    vid = video_id or uuid4()
    effective_vfr = metadata.is_vfr if is_vfr is None else is_vfr
    effective_hint = metadata.is_vfr if is_vfr_hint is None else is_vfr_hint

    return VideoManifest(
        video_id=vid,
        source_path=str(path.resolve()),
        filename=metadata.filename,
        filesize_bytes=metadata.filesize_bytes,
        mime_type=metadata.mime_type,
        duration_seconds=metadata.duration_seconds,
        fps=metadata.fps,
        total_frames=metadata.total_frames,
        width=metadata.width,
        height=metadata.height,
        video_codec=metadata.video_codec,
        audio_codec=metadata.audio_codec,
        has_audio=metadata.has_audio,
        is_vfr=effective_vfr,
        timing_mode=timing_mode,
        is_vfr_hint=effective_hint,
        scenes=scenes,
        metadata={
            "container_format": metadata.container_format,
            "r_frame_rate": metadata.r_frame_rate,
            "avg_frame_rate": metadata.avg_frame_rate,
            **metadata.metadata,
        },
    )


def format_manifest_summary(manifest: VideoManifest) -> str:
    """Format a VideoManifest into a human-readable text summary for CLI / logs."""
    filesize_mb = manifest.filesize_bytes / (1024 * 1024)
    mins = int(manifest.duration_seconds // 60)
    secs = manifest.duration_seconds % 60
    duration_str = f"{mins}m {secs:04.1f}s" if mins > 0 else f"{secs:.2f}s"

    audio_info = f"Yes ({manifest.audio_codec or 'unknown'})" if manifest.has_audio else "None"
    video_codec = manifest.video_codec or "unknown"
    rate_mode = "VFR" if manifest.is_vfr else "CFR"
    timing_policy = manifest.timing_mode.upper() if hasattr(manifest, "timing_mode") else "EXACT"

    lines = [
        "=" * 60,
        " VIDEX Video Manifest Summary",
        "=" * 60,
        f" Video ID:        {manifest.video_id}",
        f" Filename:        {manifest.filename}",
        f" Source Path:     {manifest.source_path}",
        f" File Size:       {filesize_mb:.2f} MB ({manifest.filesize_bytes:,} bytes)",
        f" MIME Type:       {manifest.mime_type or 'unknown'}",
        "-" * 60,
        f" Resolution:      {manifest.width} x {manifest.height}",
        f" Frame Rate:      {manifest.fps:.2f} fps ({rate_mode})",
        f" Timing Mode:     {timing_policy}",
        f" Total Frames:    {manifest.total_frames:,}",
        f" Duration:        {duration_str} ({manifest.duration_seconds:.2f}s)",
        f" Video Codec:     {video_codec}",
        f" Audio Stream:    {audio_info}",
        "-" * 60,
        f" Detected Scenes: {len(manifest.scenes)}",
    ]

    for sc in manifest.scenes[:10]:
        dur = sc.end_timestamp_seconds - sc.start_timestamp_seconds
        lines.append(
            f"   [{sc.scene_index:02d}] {sc.start_timestamp_seconds:06.2f}s -> "
            f"{sc.end_timestamp_seconds:06.2f}s  ({dur:.2f}s, "
            f"frames {sc.start_frame_number}..{sc.end_frame_number})"
        )
    if len(manifest.scenes) > 10:
        lines.append(f"   ... and {len(manifest.scenes) - 10} more scene(s)")

    lines.append("=" * 60)
    return "\n".join(lines)
