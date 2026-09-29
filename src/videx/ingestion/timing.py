"""Timestamp indexing, temporal truth, and frame-time mapping implementations.

Provides timing abstractions for Constant Frame Rate (CFR) and Variable Frame Rate (VFR)
video streams, enabling deterministic mapping between sequential frame numbers and
presentation timestamps (PTS) with complete provenance tracking.

Timing Policies:
- TimingMode.EXACT (default): Scans container stream frames via PyAV to establish an
  authoritative, strictly monotonic presentation timestamp index. Prevents silent
  misclassification of VFR streams caused by deceptive container metadata hints.
- TimingMode.FAST: Unlocks formula-based CFRTimestampIndex (t = i / fps) only when
  preliminary container metadata indicates CFR and the fast path is explicitly requested.

Provenance & Quality:
- FrameTimestamp tracks whether each frame's timestamp was extracted directly from the
  container (TimestampSource.CONTAINER) or derived/repaired (TimestampSource.DERIVED).
- sanitize_timestamps guarantees strict presentation monotonicity (t_0 < t_1 < ... < t_{N-1})
  while preserving valid container timestamps and recording explicit repair reasons.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from pathlib import Path

import av

from videx.ingestion.base import (
    CorruptVideoError,
    FrameOutOfBoundsError,
    FrameTimestamp,
    TimestampIndex,
    TimestampOutOfBoundsError,
    TimestampSource,
    TimingMode,
    VideoNotFoundError,
)


def sanitize_timestamps(
    raw_timestamps: Sequence[float | None],
    nominal_fps: float = 25.0,
) -> list[FrameTimestamp]:
    """Defensively sanitize a sequence of presentation timestamps with provenance tracking.

    Guarantees:
    1. Strictly monotonic progression: t_0 < t_1 < t_2 < ... < t_{N-1}.
    2. Non-negative starting timestamp: t_0 >= 0.0.
    3. Handles missing (None/NaN) timestamps via linear interpolation or nominal advance,
       tagging the resulting FrameTimestamp as DERIVED (repair_reason="missing_pts").
    4. Handles duplicate or backward (non-monotonic) PTS defensively, tagging the resulting
       FrameTimestamp as DERIVED (repair_reason="duplicate_pts" or "non_monotonic_pts").
    5. Exact container timestamps that are valid and monotonic are tagged as CONTAINER
       (is_repaired=False, repair_reason=None).

    Args:
        raw_timestamps: Sequence of raw presentation timestamps in seconds.
        nominal_fps: Nominal frame rate for fallback step calculation.

    Returns:
        List of FrameTimestamp instances supporting rich numeric comparison.

    Raises:
        ValueError: If raw_timestamps sequence is empty.
    """
    if not raw_timestamps:
        raise ValueError("Cannot sanitize empty timestamp sequence")

    fps = max(nominal_fps, 0.001)
    nom_delta = 1.0 / fps
    n = len(raw_timestamps)
    provenance: list[FrameTimestamp] = []

    # Step 1: Ensure initial timestamp is valid and non-negative
    first = raw_timestamps[0]
    if first is None or math.isnan(first) or first < 0.0:
        provenance.append(
            FrameTimestamp(
                frame_index=0,
                pts_seconds=0.0,
                timestamp_source=TimestampSource.DERIVED,
                is_repaired=True,
                repair_reason="missing_start_pts",
                original_pts_seconds=first if first is not None and not math.isnan(first) else None,
            )
        )
    else:
        provenance.append(
            FrameTimestamp(
                frame_index=0,
                pts_seconds=float(first),
                timestamp_source=TimestampSource.CONTAINER,
                is_repaired=False,
                repair_reason=None,
                original_pts_seconds=float(first),
            )
        )

    # Step 2: Sequential forward pass enforcing strict monotonicity
    for i in range(1, n):
        t_prev = provenance[i - 1].pts_seconds
        curr = raw_timestamps[i]

        # Case A: Missing or NaN timestamp
        if curr is None or math.isnan(curr):
            next_idx: int | None = None
            for k in range(i + 1, n):
                val = raw_timestamps[k]
                if val is not None and not math.isnan(val) and float(val) > t_prev:
                    next_idx = k
                    break
            if next_idx is not None:
                anchor = float(raw_timestamps[next_idx])  # type: ignore[arg-type]
                step = (anchor - t_prev) / (next_idx - (i - 1))
                pts = t_prev + step
            else:
                pts = t_prev + nom_delta

            provenance.append(
                FrameTimestamp(
                    frame_index=i,
                    pts_seconds=pts,
                    timestamp_source=TimestampSource.DERIVED,
                    is_repaired=True,
                    repair_reason="missing_pts",
                    original_pts_seconds=None,
                )
            )
            continue

        curr_float = float(curr)

        # Case B: Duplicate timestamp
        if curr_float == t_prev:
            next_idx = None
            for k in range(i + 1, n):
                val = raw_timestamps[k]
                if val is not None and not math.isnan(val) and float(val) > t_prev:
                    next_idx = k
                    break
            if next_idx is not None:
                anchor = float(raw_timestamps[next_idx])  # type: ignore[arg-type]
                step = (anchor - t_prev) / (next_idx - (i - 1))
                pts = t_prev + min(step, nom_delta)
            else:
                pts = t_prev + max(nom_delta * 0.01, 1e-4)

            provenance.append(
                FrameTimestamp(
                    frame_index=i,
                    pts_seconds=pts,
                    timestamp_source=TimestampSource.DERIVED,
                    is_repaired=True,
                    repair_reason="duplicate_pts",
                    original_pts_seconds=curr_float,
                )
            )
        # Case C: Backward / non-monotonic timestamp
        elif curr_float < t_prev:
            next_idx = None
            for k in range(i + 1, n):
                val = raw_timestamps[k]
                if val is not None and not math.isnan(val) and float(val) > t_prev:
                    next_idx = k
                    break
            if next_idx is not None:
                anchor = float(raw_timestamps[next_idx])  # type: ignore[arg-type]
                step = (anchor - t_prev) / (next_idx - (i - 1))
                pts = t_prev + min(step, nom_delta)
            else:
                pts = t_prev + max(nom_delta * 0.01, 1e-4)

            provenance.append(
                FrameTimestamp(
                    frame_index=i,
                    pts_seconds=pts,
                    timestamp_source=TimestampSource.DERIVED,
                    is_repaired=True,
                    repair_reason="non_monotonic_pts",
                    original_pts_seconds=curr_float,
                )
            )
        else:
            # Case D: Valid strictly monotonic container timestamp
            provenance.append(
                FrameTimestamp(
                    frame_index=i,
                    pts_seconds=curr_float,
                    timestamp_source=TimestampSource.CONTAINER,
                    is_repaired=False,
                    repair_reason=None,
                    original_pts_seconds=curr_float,
                )
            )

    return provenance


def detect_vfr_from_timestamps(
    timestamps: Sequence[float | FrameTimestamp],
    relative_tolerance: float = 0.02,
) -> bool:
    """Authoritatively determine whether a stream is VFR from presentation timestamps.

    Examines inter-frame presentation intervals (delta PTS). If the maximum
    relative deviation from the median interval exceeds the tolerance threshold,
    the media is classified as Variable Frame Rate (VFR).

    Args:
        timestamps: Sequence of monotonically increasing timestamps or FrameTimestamps in seconds.
        relative_tolerance: Fractional deviation threshold (default 0.02 = 2%).

    Returns:
        True if stream exhibits significant frame interval variance; False for CFR.
    """
    if len(timestamps) < 3:
        return False

    deltas = [float(timestamps[i]) - float(timestamps[i - 1]) for i in range(1, len(timestamps))]
    valid_deltas = [d for d in deltas if d > 1e-6]
    if len(valid_deltas) < 2:
        return False

    sorted_deltas = sorted(valid_deltas)
    median_delta = sorted_deltas[len(sorted_deltas) // 2]
    if median_delta <= 0:
        return False

    max_dev = max(abs(d - median_delta) for d in valid_deltas)
    return (max_dev / median_delta) > relative_tolerance


class CFRTimestampIndex(TimestampIndex):
    """Constant Frame Rate (CFR) timestamp index.

    Calculates presentation timestamps uniformly using nominal frame rate:
        timestamp = frame_index / fps
        frame_index = round(timestamp * fps)
    """

    def __init__(
        self,
        fps: float,
        total_frames: int,
        duration_seconds: float | None = None,
    ) -> None:
        self._fps = max(fps, 0.001)
        self._total_frames = max(0, total_frames)
        if duration_seconds is not None and duration_seconds > 0.0:
            self._duration_seconds = duration_seconds
        else:
            self._duration_seconds = (
                self._total_frames / self._fps if self._total_frames > 0 else 0.0
            )

    @property
    def is_vfr(self) -> bool:
        """Always False for CFR index."""
        return False

    @property
    def fps(self) -> float:
        """Constant frames per second."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total number of frames indexed."""
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        """Total duration spanned in seconds."""
        return self._duration_seconds

    @property
    def timing_mode(self) -> TimingMode:
        """CFR index always operates in FAST mode."""
        return TimingMode.FAST

    @property
    def has_repaired_timestamps(self) -> bool:
        """CFR formula timestamps are mathematical derivations without repairs."""
        return False

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """Return the presentation timestamp in seconds for a 0-based frame index."""
        if frame_index < 0 or (self._total_frames > 0 and frame_index >= self._total_frames):
            raise FrameOutOfBoundsError(frame_index, self._total_frames)
        return frame_index / self._fps

    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int:
        """Return the closest 0-based frame index for a presentation timestamp in seconds."""
        epsilon = 0.05
        if timestamp_seconds < 0.0 or (
            self._duration_seconds > 0.0 and timestamp_seconds > self._duration_seconds + epsilon
        ):
            raise TimestampOutOfBoundsError(timestamp_seconds, self._duration_seconds)
        target = int(round(timestamp_seconds * self._fps))
        max_idx = max(0, self._total_frames - 1) if self._total_frames > 0 else target
        return min(max(0, target), max_idx)

    def get_frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        """Return the detailed provenance record for a 0-based frame index."""
        pts = self.get_timestamp_for_frame(frame_index)
        return FrameTimestamp(
            frame_index=frame_index,
            pts_seconds=pts,
            timestamp_source=TimestampSource.DERIVED,
            is_repaired=False,
            repair_reason=None,
            original_pts_seconds=None,
        )


class ExplicitTimestampIndex(TimestampIndex):
    """Explicit per-frame presentation timestamp (PTS) table index.

    Maintains an explicit list of per-frame timestamps, preventing cumulative
    temporal drift in VFR footage. Frame-to-timestamp lookup is O(1) and
    timestamp-to-frame lookup is O(log N) via binary search.
    """

    def __init__(
        self,
        timestamps: Sequence[float | FrameTimestamp | None],
        nominal_fps: float | None = None,
        is_vfr: bool = True,
        source_path: Path | str | None = None,
        timing_mode: TimingMode = TimingMode.EXACT,
    ) -> None:
        if not timestamps:
            raise ValueError("ExplicitTimestampIndex requires at least one timestamp")

        if all(isinstance(t, FrameTimestamp) for t in timestamps):
            self._provenance: list[FrameTimestamp] = list(timestamps)  # type: ignore[arg-type]
        else:
            raw_floats = [float(t) if t is not None else None for t in timestamps]
            self._provenance = sanitize_timestamps(raw_floats, nominal_fps=nominal_fps or 25.0)

        self._timestamps: list[float] = [p.pts_seconds for p in self._provenance]
        self._total_frames = len(self._timestamps)
        self._duration_seconds = self._timestamps[-1]
        self._is_vfr = is_vfr
        self._source_path = Path(source_path) if source_path else None
        self._timing_mode = timing_mode
        if nominal_fps and nominal_fps > 0:
            self._fps = nominal_fps
        else:
            self._fps = (
                self._total_frames / self._duration_seconds
                if self._duration_seconds > 0
                else 25.0
            )

    @property
    def is_vfr(self) -> bool:
        """Whether the stream operates with variable frame rates."""
        return self._is_vfr

    @property
    def fps(self) -> float:
        """Nominal or average frames per second."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total number of frames indexed."""
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        """Duration spanning from 0.0 to the final presentation timestamp."""
        return self._duration_seconds

    @property
    def timing_mode(self) -> TimingMode:
        """Active timing mode for this index."""
        return self._timing_mode

    @property
    def has_repaired_timestamps(self) -> bool:
        """Whether any timestamps in the index were derived/repaired."""
        return any(p.is_repaired for p in self._provenance)

    @property
    def timestamps(self) -> list[float]:
        """Copy of the internal presentation timestamps in seconds."""
        return list(self._timestamps)

    @property
    def provenance(self) -> list[FrameTimestamp]:
        """Copy of the internal FrameTimestamp provenance records."""
        return list(self._provenance)

    @property
    def source_path(self) -> Path | None:
        """Path to source video file if indexed from disk."""
        return self._source_path

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """Return the explicit presentation timestamp for a 0-based frame index."""
        if frame_index < 0 or frame_index >= self._total_frames:
            raise FrameOutOfBoundsError(frame_index, self._total_frames)
        return self._timestamps[frame_index]

    def get_frame_for_timestamp(self, timestamp_seconds: float) -> int:
        """Return the closest frame index for a presentation timestamp in seconds."""
        epsilon = 0.05
        if timestamp_seconds < 0.0 or timestamp_seconds > self._duration_seconds + epsilon:
            raise TimestampOutOfBoundsError(timestamp_seconds, self._duration_seconds)

        idx = bisect_left(self._timestamps, timestamp_seconds)
        if idx == 0:
            return 0
        if idx >= len(self._timestamps):
            return len(self._timestamps) - 1

        before = self._timestamps[idx - 1]
        after = self._timestamps[idx]
        if abs(timestamp_seconds - before) <= abs(timestamp_seconds - after):
            return idx - 1
        return idx

    def get_frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        """Return the detailed provenance record for a 0-based frame index."""
        if frame_index < 0 or frame_index >= self._total_frames:
            raise FrameOutOfBoundsError(frame_index, self._total_frames)
        return self._provenance[frame_index]


class PyAVTimestampIndex(ExplicitTimestampIndex):
    """Packet- and container-accurate presentation timestamp index backed by PyAV / FFmpeg.

    Scans the video container, decodes per-frame presentation timestamps in display order,
    and constructs a strictly monotonic, packet-accurate timing index with full provenance.
    """

    @classmethod
    def from_video(
        cls,
        video_path: Path | str,
        fallback_fps: float = 25.0,
    ) -> PyAVTimestampIndex:
        """Scan a video file and build a packet-accurate presentation timestamp index.

        Executes a lightweight decoder pass in C via PyAV container.decode(stream) to
        obtain presentation timestamps in presentation order without copying uncompressed
        pixel buffers to Python NumPy arrays.

        Args:
            video_path: Path to target video file.
            fallback_fps: Fallback nominal frame rate if container metadata lacks it.

        Returns:
            PyAVTimestampIndex populated with per-frame presentation timestamps and provenance.

        Raises:
            VideoNotFoundError: If the file does not exist.
            CorruptVideoError: If the container is unreadable or lacks video streams.
        """
        path = Path(video_path).resolve()
        if not path.is_file():
            raise VideoNotFoundError(path)

        try:
            container = av.open(str(path))
        except (FileNotFoundError, av.FileNotFoundError) as e:
            raise VideoNotFoundError(path) from e
        except (av.FFmpegError, ValueError) as e:
            raise CorruptVideoError(path, f"Failed to open container with PyAV: {e}") from e

        try:
            if not container.streams.video:
                raise CorruptVideoError(path, "Container contains no video streams")
            stream = container.streams.video[0]

            # Extract nominal frame rate
            fps = fallback_fps
            if stream.average_rate is not None and stream.average_rate > 0:
                fps = float(stream.average_rate)
            elif stream.base_rate is not None and stream.base_rate > 0:
                fps = float(stream.base_rate)
            elif stream.guessed_rate is not None and stream.guessed_rate > 0:
                fps = float(stream.guessed_rate)

            # Extract raw presentation timestamps in presentation order
            raw_pts: list[float | None] = []
            for frame in container.decode(stream):
                if frame.time is not None:
                    raw_pts.append(float(frame.time))
                elif frame.pts is not None and stream.time_base is not None:
                    raw_pts.append(float(frame.pts * stream.time_base))
                else:
                    raw_pts.append(None)
        except (av.FFmpegError, ValueError) as e:
            raise CorruptVideoError(
                path, f"PyAV decoding error during timestamp indexing: {e}"
            ) from e
        finally:
            container.close()

        if not raw_pts:
            raise CorruptVideoError(path, "Zero video frames decoded from stream")

        sanitized_provenance = sanitize_timestamps(raw_pts, nominal_fps=fps)
        is_vfr = detect_vfr_from_timestamps(sanitized_provenance)

        return cls(
            timestamps=sanitized_provenance,
            nominal_fps=fps,
            is_vfr=is_vfr,
            source_path=path,
            timing_mode=TimingMode.EXACT,
        )


def create_timestamp_index(
    fps: float,
    total_frames: int,
    duration_seconds: float,
    video_path: Path | str | None = None,
    mode: TimingMode | str = TimingMode.EXACT,
    is_vfr_hint: bool | None = False,
    explicit_timestamps: Sequence[float | FrameTimestamp] | None = None,
    is_vfr: bool | None = None,
) -> TimestampIndex:
    """Factory creating an appropriate TimestampIndex implementation under configured policy.

    Timing Policy:
    - TimingMode.EXACT (default): When video_path is provided, ALWAYS constructs
      PyAVTimestampIndex by scanning presentation timestamps from the container.
      Eliminates false-negative metadata heuristic risks.
    - TimingMode.FAST: When explicitly selected, allows formula-based CFRTimestampIndex
      (t = i / fps) if is_vfr_hint is False.

    Args:
        fps: Nominal or average frames per second.
        total_frames: Total frame count.
        duration_seconds: Total video duration in seconds.
        video_path: Optional path to video file for container-level indexing.
        mode: Timing policy mode (TimingMode.EXACT or TimingMode.FAST).
        is_vfr_hint: Preliminary metadata hint regarding frame rate variance.
        explicit_timestamps: Optional explicit PTS list for VFR streams.
        is_vfr: Optional backward-compatibility alias for is_vfr_hint.

    Returns:
        Configured TimestampIndex instance.
    """
    if is_vfr is not None:
        is_vfr_hint = is_vfr

    hint_bool = bool(is_vfr_hint)
    mode_enum = TimingMode(mode)

    if explicit_timestamps:
        return ExplicitTimestampIndex(
            explicit_timestamps,
            nominal_fps=fps,
            is_vfr=hint_bool,
            source_path=video_path,
            timing_mode=mode_enum,
        )

    if mode_enum == TimingMode.EXACT:
        if video_path is not None and Path(video_path).is_file():
            return PyAVTimestampIndex.from_video(
                video_path=video_path,
                fallback_fps=fps,
            )
        # Fallback if no video file on disk (e.g. synthetic test)
        if is_vfr_hint:
            return ExplicitTimestampIndex(
                timestamps=[i / max(fps, 0.001) for i in range(total_frames)],
                nominal_fps=fps,
                is_vfr=True,
                source_path=video_path,
                timing_mode=TimingMode.EXACT,
            )
        return CFRTimestampIndex(
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
        )

    # TimingMode.FAST path
    if not is_vfr_hint:
        return CFRTimestampIndex(
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
        )

    if video_path is not None and Path(video_path).is_file():
        return PyAVTimestampIndex.from_video(
            video_path=video_path,
            fallback_fps=fps,
        )

    return CFRTimestampIndex(
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration_seconds,
    )
