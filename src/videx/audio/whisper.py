"""Faster-Whisper ASR Provider implementation for VIDEX.

Wraps CTranslate2-accelerated faster-whisper models to deliver high-throughput,
timestamped speech transcription with word-level alignment and language routing.
"""

from __future__ import annotations

import logging
import math
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from videx.audio.base import FasterWhisperConfig, resolve_audio_input
from videx.audio.normalization import normalize_transcript
from videx.domain.schemas import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)

__all__ = ["FasterWhisperASRProvider"]


class FasterWhisperASRProvider:
    """ASR Provider backend powered by faster-whisper (CTranslate2).

    Supports CPU and CUDA inference, int8/float16 quantization, voice activity
    detection (VAD), word-level timestamps, and language-specific routing (en/hi).
    """

    def __init__(self, config: FasterWhisperConfig | None = None) -> None:
        self.config = config or FasterWhisperConfig()
        self._model: Any = None
        self._is_warmed_up = False

    @property
    def provider_name(self) -> str:
        """Unique provider identifier string."""
        return self.config.provider_name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Verified supported languages for this provider."""
        return ("en", "hi")

    def _load_model(self) -> Any:  # noqa: ANN401
        """Lazily load and cache the faster-whisper WhisperModel."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install it via `uv add faster-whisper`."
                ) from exc

            logger.info(
                "Loading faster-whisper model '%s' on %s (%s)",
                self.config.model_size_or_path,
                self.config.device,
                self.config.compute_type,
            )
            self._model = WhisperModel(
                self.config.model_size_or_path,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=self.config.download_root,
                **self.config.extra_options,
            )
        return self._model

    def warmup(self) -> None:
        """Warm up model weights and allocate execution context."""
        if self._is_warmed_up:
            return

        model = self._load_model()
        try:
            # 0.5s of silence at 16kHz
            dummy_audio = np.zeros(8000, dtype=np.float32)
            # Transcribe dummy audio to warm up execution kernels
            _ = list(
                model.transcribe(
                    dummy_audio,
                    language=self.config.language or "en",
                    beam_size=1,
                    vad_filter=False,
                )[0]
            )
            self._is_warmed_up = True
            logger.info("faster-whisper provider '%s' warmed up successfully.", self.provider_name)
        except Exception as exc:
            logger.warning("Warmup failed for faster-whisper provider: %s", exc)

    def transcribe(
        self,
        audio_input: Any,  # noqa: ANN401
        video_id: UUID | None = None,
        language: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[TranscriptSegment]:
        """Transcribe audio into time-aligned TranscriptSegment records.

        Args:
            audio_input: Audio file path, raw WAV bytes, or 1D float32 numpy array.
            video_id: Optional parent video UUID.
            language: Optional language ISO code hint (e.g. 'en', 'hi').
            **kwargs: Extra parameters passed to model.transcribe().

        Returns:
            List of TranscriptSegment records ordered by start_timestamp_seconds.
        """
        model = self._load_model()
        resolved = resolve_audio_input(audio_input, video_id=video_id)
        v_id = resolved.video_id

        # Determine audio payload for faster-whisper
        audio_source: Any
        if resolved.audio_array is not None:
            audio_source = resolved.audio_array
        elif resolved.audio_path is not None:
            audio_source = resolved.audio_path
        elif resolved.raw_bytes is not None:
            import io

            audio_source = io.BytesIO(resolved.raw_bytes)
        else:
            raise ValueError("No valid audio array, path, or bytes found in ResolvedAudioInput.")

        # Determine language hint
        target_lang = language or self.config.language

        # Merge transcribe arguments
        transcribe_kwargs: dict[str, Any] = {
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "vad_filter": self.config.vad_filter,
            "word_timestamps": self.config.word_timestamps,
            "temperature": self.config.temperature,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "log_prob_threshold": self.config.log_prob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "condition_on_previous_text": self.config.condition_on_previous_text,
        }
        if self.config.vad_parameters is not None:
            transcribe_kwargs["vad_parameters"] = self.config.vad_parameters
        if self.config.initial_prompt is not None:
            transcribe_kwargs["initial_prompt"] = self.config.initial_prompt
        if target_lang is not None:
            transcribe_kwargs["language"] = target_lang

        # Apply any caller overrides
        transcribe_kwargs.update(kwargs)

        try:
            segments_gen, info = model.transcribe(audio_source, **transcribe_kwargs)
            detected_lang = info.language if hasattr(info, "language") else (target_lang or "en")

            results: list[TranscriptSegment] = []
            for seg in segments_gen:
                raw_text = seg.text.strip()
                if not raw_text:
                    continue

                norm_text = normalize_transcript(raw_text)

                # Confidence calculation from avg_logprob
                conf: float = 0.9
                if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                    try:
                        conf = float(math.exp(seg.avg_logprob))
                        conf = max(0.0, min(1.0, conf))
                    except (OverflowError, ValueError):
                        conf = 0.5

                # Extract word-level timestamps if available
                words_list: list[WordTimestamp] = []
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        w_conf = None
                        if hasattr(w, "probability") and w.probability is not None:
                            w_conf = max(0.0, min(1.0, float(w.probability)))
                        words_list.append(
                            WordTimestamp(
                                word=w.word.strip(),
                                start_timestamp_seconds=float(w.start),
                                end_timestamp_seconds=float(w.end),
                                confidence=w_conf,
                            )
                        )

                no_speech_prob = (
                    float(seg.no_speech_prob)
                    if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None
                    else None
                )

                results.append(
                    TranscriptSegment(
                        segment_id=uuid4(),
                        video_id=v_id,
                        start_timestamp_seconds=float(seg.start),
                        end_timestamp_seconds=float(seg.end),
                        raw_text=raw_text,
                        normalized_text=norm_text,
                        language=str(detected_lang),
                        confidence=conf,
                        provider=self.provider_name,
                        words=words_list,
                        no_speech_prob=no_speech_prob,
                        attributes={
                            "avg_logprob": getattr(seg, "avg_logprob", None),
                            "compression_ratio": getattr(seg, "compression_ratio", None),
                        },
                    )
                )

            return results

        except Exception as exc:
            logger.error("faster-whisper transcription failed for video %s: %s", v_id, exc)
            raise RuntimeError(f"faster-whisper transcription error: {exc}") from exc
