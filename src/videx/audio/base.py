"""Base configurations, data structures, and protocol re-exports for Audio/ASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from videx.domain.schemas import AudioMetadata
from videx.providers.base import ASRProvider, AudioProvider, SoundEventProvider

__all__ = [
    "ASRProvider",
    "AudioProvider",
    "FasterWhisperConfig",
    "MockASRConfig",
    "ResolvedAudioInput",
    "SoundEventProvider",
    "resolve_audio_input",
]


@dataclass(frozen=True)
class FasterWhisperConfig:
    """Configuration for faster-whisper ASR runtime engine.

    Allows full injection of model, device, compute type, language hint,
    beam size, VAD filtering, and word-level timestamp options.
    """

    model_size_or_path: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    best_of: int = 5
    vad_filter: bool = True
    vad_parameters: dict[str, Any] | None = None
    word_timestamps: bool = True
    temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    download_root: str | None = None
    provider_name: str = "faster_whisper"
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockASRConfig:
    """Configuration for deterministic testing mock ASR provider."""

    provider_name: str = "mock_asr"
    supported_languages: tuple[str, ...] = ("en", "hi")
    canned_segments: list[dict[str, Any]] | None = None
    default_text: str = "Sample speech transcript"
    default_language: str = "en"
    default_confidence: float = 0.95
    latency_ms: float = 0.0
    segment_duration_seconds: float = 2.0


@dataclass(frozen=True)
class ResolvedAudioInput:
    """Standardized audio payload passed into ASR engines.

    Guarantees video association and duration tracking.
    """

    video_id: UUID
    audio_array: np.ndarray | None = None  # 1D float32 16kHz array
    audio_path: str | None = None
    raw_bytes: bytes | None = None
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    metadata: AudioMetadata | None = None


def resolve_audio_input(
    audio_input: str | Path | bytes | np.ndarray | ResolvedAudioInput,
    video_id: UUID | None = None,
) -> ResolvedAudioInput:
    """Resolve various audio representations into a standard ResolvedAudioInput.

    Args:
        audio_input: Path, audio bytes, 1D float32 numpy array, or ResolvedAudioInput.
        video_id: Optional parent video ID. If None, a new UUID is generated.

    Returns:
        ResolvedAudioInput.
    """
    v_id = video_id or uuid4()

    if isinstance(audio_input, ResolvedAudioInput):
        if video_id is not None and audio_input.video_id != video_id:
            return ResolvedAudioInput(
                video_id=video_id,
                audio_array=audio_input.audio_array,
                audio_path=audio_input.audio_path,
                raw_bytes=audio_input.raw_bytes,
                sample_rate=audio_input.sample_rate,
                duration_seconds=audio_input.duration_seconds,
                metadata=audio_input.metadata,
            )
        return audio_input

    if isinstance(audio_input, (str, Path)):
        p_str = str(audio_input)
        return ResolvedAudioInput(
            video_id=v_id,
            audio_path=p_str,
            sample_rate=16000,
        )

    if isinstance(audio_input, np.ndarray):
        arr = audio_input.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()
        duration = len(arr) / 16000.0
        return ResolvedAudioInput(
            video_id=v_id,
            audio_array=arr,
            sample_rate=16000,
            duration_seconds=duration,
        )

    if isinstance(audio_input, bytes):
        return ResolvedAudioInput(
            video_id=v_id,
            raw_bytes=audio_input,
            sample_rate=16000,
        )

    raise TypeError(f"Unsupported audio input type: {type(audio_input).__name__}")
