"""Audio extraction and metadata inspection utilities for VIDEX.

Extracts audio streams from video containers using PyAV without modifying or
corrupting the original video file.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import av
import av.audio.resampler
import numpy as np

from videx.domain.schemas import AudioMetadata

__all__ = [
    "AudioExtractionError",
    "CorruptAudioError",
    "EmptyAudioError",
    "NoAudioStreamError",
    "extract_audio_bytes_wav",
    "extract_audio_metadata",
    "extract_audio_stream",
    "extract_audio_to_wav",
]


class AudioExtractionError(Exception):
    """Base exception for audio extraction failures."""


class NoAudioStreamError(AudioExtractionError):
    """Raised when the video file contains no audio stream."""


class CorruptAudioError(AudioExtractionError):
    """Raised when the audio stream is unreadable or corrupted."""


class EmptyAudioError(AudioExtractionError):
    """Raised when the extracted audio stream contains zero frames or samples."""


def extract_audio_metadata(video_path: str | Path) -> AudioMetadata:
    """Extract audio stream metadata from a video container.

    Args:
        video_path: Path to the video file.

    Returns:
        AudioMetadata with stream parameters.

    Raises:
        FileNotFoundError: If the video file does not exist.
        NoAudioStreamError: If the video contains no audio stream.
        CorruptAudioError: If the container cannot be opened or parsed.
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        container = av.open(str(path_obj))
    except Exception as exc:
        raise CorruptAudioError(f"Failed to open video container {video_path}: {exc}") from exc

    with container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise NoAudioStreamError(f"No audio stream found in {video_path}")

        stream = audio_streams[0]
        sample_rate = stream.sample_rate or stream.rate or 44100
        channels_raw = stream.channels or (stream.layout.channels if stream.layout else 2)
        channel_count: int = (
            channels_raw
            if isinstance(channels_raw, int)
            else (len(channels_raw) if isinstance(channels_raw, (tuple, list)) else 2)
        )
        codec_name = (
            stream.codec_context.name if stream.codec_context else (stream.name or "unknown")
        )
        bit_rate = (
            stream.bit_rate or (stream.codec_context.bit_rate if stream.codec_context else None)
        )
        layout_name = stream.layout.name if stream.layout else None

        # Determine duration
        duration_sec = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_sec = float(container.duration / av.time_base)

        return AudioMetadata(
            sample_rate=int(sample_rate),
            channels=channel_count,
            duration_seconds=max(0.0, float(duration_sec)),
            codec_name=str(codec_name) if codec_name else None,
            bit_rate=int(bit_rate) if bit_rate else None,
            channel_layout=str(layout_name) if layout_name else None,
            attributes={
                "stream_index": stream.index,
                "time_base": str(stream.time_base),
            },
        )


def extract_audio_stream(
    video_path: str | Path,
    target_sample_rate: int = 16000,
) -> tuple[np.ndarray, AudioMetadata]:
    """Decode and resample audio from video into a 1D float32 numpy array.

    The returned array is 16kHz mono float32 normalized in [-1.0, 1.0],
    which is directly compatible with faster-whisper and other ASR engines.

    Args:
        video_path: Path to the video file.
        target_sample_rate: Desired audio sample rate in Hz (default 16000).

    Returns:
        Tuple of (audio_array, audio_metadata).

    Raises:
        FileNotFoundError: If the video file does not exist.
        NoAudioStreamError: If the video contains no audio stream.
        CorruptAudioError: If decoding fails.
        EmptyAudioError: If the audio stream contains zero samples.
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    metadata = extract_audio_metadata(path_obj)

    try:
        container = av.open(str(path_obj))
    except Exception as exc:
        raise CorruptAudioError(f"Failed to open video container {video_path}: {exc}") from exc

    with container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise NoAudioStreamError(f"No audio stream found in {video_path}")

        stream = audio_streams[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=target_sample_rate,
        )

        audio_chunks: list[np.ndarray] = []
        try:
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for rf in resampled_frames:
                        # rf.to_ndarray() has shape (1, num_samples) for mono float32
                        arr = rf.to_ndarray()
                        if arr.ndim > 1:
                            arr = arr.flatten()
                        audio_chunks.append(arr.astype(np.float32))

            # Flush resampler buffer if needed
            remaining_frames = resampler.resample(None)
            if remaining_frames:
                for rf in remaining_frames:
                    arr = rf.to_ndarray()
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    audio_chunks.append(arr.astype(np.float32))

        except Exception as exc:
            raise CorruptAudioError(
                f"Failed decoding audio frames from {video_path}: {exc}"
            ) from exc

    if not audio_chunks:
        raise EmptyAudioError(f"Audio stream in {video_path} yielded 0 samples.")

    full_audio = np.concatenate(audio_chunks, axis=0)
    if len(full_audio) == 0:
        raise EmptyAudioError(f"Extracted audio array from {video_path} is empty.")

    # Update metadata duration if container header was missing duration
    actual_duration = len(full_audio) / float(target_sample_rate)
    if metadata.duration_seconds == 0.0:
        metadata = AudioMetadata(
            sample_rate=target_sample_rate,
            channels=1,
            duration_seconds=actual_duration,
            codec_name=metadata.codec_name,
            bit_rate=metadata.bit_rate,
            channel_layout="mono",
            attributes=metadata.attributes,
        )

    return full_audio, metadata


def extract_audio_bytes_wav(
    video_path: str | Path,
    target_sample_rate: int = 16000,
) -> tuple[bytes, AudioMetadata]:
    """Decode audio and package it as standard 16-bit PCM WAV bytes in-memory.

    Args:
        video_path: Path to the video file.
        target_sample_rate: Target sample rate in Hz (default 16000).

    Returns:
        Tuple of (wav_bytes, metadata).
    """
    audio_arr, meta = extract_audio_stream(video_path, target_sample_rate=target_sample_rate)

    # Scale float32 [-1.0, 1.0] to int16 [-32768, 32767]
    clipped = np.clip(audio_arr, -1.0, 1.0)
    int16_samples = (clipped * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(target_sample_rate)
        wav_file.writeframes(int16_samples.tobytes())

    return buffer.getvalue(), meta


def extract_audio_to_wav(
    video_path: str | Path,
    output_wav_path: str | Path,
    target_sample_rate: int = 16000,
) -> Path:
    """Extract audio track from video and save to a WAV file on disk.

    Args:
        video_path: Path to the source video file.
        output_wav_path: Target path for the output WAV file.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        Path to the written WAV file.
    """
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav_bytes, _ = extract_audio_bytes_wav(video_path, target_sample_rate=target_sample_rate)
    out_path.write_bytes(wav_bytes)
    return out_path
