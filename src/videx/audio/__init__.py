"""VIDEX Audio Intelligence and Timestamped ASR package.

Provides audio extraction, metadata analysis, speech recognition with faster-whisper,
conservative normalization, temporal transcript fusion, and structured evidence propagation.
"""

from videx.audio.base import (
    ASRProvider,
    AudioProvider,
    FasterWhisperConfig,
    MockASRConfig,
    ResolvedAudioInput,
    SoundEventProvider,
    resolve_audio_input,
)
from videx.audio.extraction import (
    AudioExtractionError,
    CorruptAudioError,
    EmptyAudioError,
    NoAudioStreamError,
    extract_audio_bytes_wav,
    extract_audio_metadata,
    extract_audio_stream,
    extract_audio_to_wav,
)
from videx.audio.fusion import TemporalTranscriptFusion, TemporalTranscriptFusionConfig
from videx.audio.mock import MockASRProvider
from videx.audio.normalization import normalize_transcript
from videx.audio.pipeline import AudioPipeline, AudioPipelineResult
from videx.audio.whisper import FasterWhisperASRProvider
from videx.domain.schemas import (
    AudioMetadata,
    AudioSegment,
    SoundObservation,
    TranscriptSegment,
    WordTimestamp,
)

__all__ = [
    "ASRProvider",
    "AudioExtractionError",
    "AudioMetadata",
    "AudioPipeline",
    "AudioPipelineResult",
    "AudioProvider",
    "AudioSegment",
    "CorruptAudioError",
    "EmptyAudioError",
    "FasterWhisperASRProvider",
    "FasterWhisperConfig",
    "MockASRConfig",
    "MockASRProvider",
    "NoAudioStreamError",
    "ResolvedAudioInput",
    "SoundEventProvider",
    "SoundObservation",
    "TemporalTranscriptFusion",
    "TemporalTranscriptFusionConfig",
    "TranscriptSegment",
    "WordTimestamp",
    "extract_audio_bytes_wav",
    "extract_audio_metadata",
    "extract_audio_stream",
    "extract_audio_to_wav",
    "normalize_transcript",
    "resolve_audio_input",
]
