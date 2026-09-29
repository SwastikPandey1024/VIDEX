"""Deterministic Mock ASR Provider for hermetic testing and pipeline evaluation."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID, uuid4

from videx.audio.base import MockASRConfig, resolve_audio_input
from videx.audio.normalization import normalize_transcript
from videx.domain.schemas import TranscriptSegment, WordTimestamp

__all__ = ["MockASRProvider"]


class MockASRProvider:
    """Mock ASR provider that generates deterministic, repeatable transcript segments.

    Enables completely hermetic unit tests and CI runs without downloading models
    or executing heavy neural inference.
    """

    def __init__(self, config: MockASRConfig | None = None) -> None:
        self.config = config or MockASRConfig()
        self._is_warmed_up = False

    @property
    def provider_name(self) -> str:
        """Provider identifier string."""
        return self.config.provider_name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Supported language codes."""
        return self.config.supported_languages

    def warmup(self) -> None:
        """Simulate model initialization."""
        self._is_warmed_up = True

    def transcribe(
        self,
        audio_input: Any,  # noqa: ANN401
        video_id: UUID | None = None,
        language: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[TranscriptSegment]:
        """Transcribe mock audio into deterministic TranscriptSegment records.

        Args:
            audio_input: Audio input (file path, array, or bytes).
            video_id: Optional video UUID.
            language: Optional language code hint ('en', 'hi').
            **kwargs: Extra parameters.

        Returns:
            List of TranscriptSegment objects.
        """
        if self.config.latency_ms > 0:
            time.sleep(self.config.latency_ms / 1000.0)

        resolved = resolve_audio_input(audio_input, video_id=video_id)
        v_id = resolved.video_id
        target_lang = language or self.config.default_language

        # If canned segments are explicitly configured, return them mapped to TranscriptSegment
        if self.config.canned_segments is not None:
            segments: list[TranscriptSegment] = []
            for item in self.config.canned_segments:
                raw_text = str(
                    item.get("raw_text") or item.get("text") or item.get("transcript", "")
                )
                norm_text = str(item.get("normalized_text") or normalize_transcript(raw_text))
                start_sec = float(item.get("start_timestamp_seconds", 0.0))
                end_sec = float(
                    item.get(
                        "end_timestamp_seconds",
                        start_sec + self.config.segment_duration_seconds,
                    )
                )
                conf = float(item.get("confidence", self.config.default_confidence))
                seg_lang = str(item.get("language", target_lang))

                # Build mock words if words not provided
                raw_words = item.get("words")
                word_objs: list[WordTimestamp] = []
                if raw_words:
                    for w in raw_words:
                        word_objs.append(
                            WordTimestamp(
                                word=str(w.get("word", "")),
                                start_timestamp_seconds=float(
                                    w.get("start_timestamp_seconds", start_sec)
                                ),
                                end_timestamp_seconds=float(
                                    w.get("end_timestamp_seconds", end_sec)
                                ),
                                confidence=(
                                    float(w.get("confidence", conf))
                                    if w.get("confidence") is not None
                                    else None
                                ),
                            )
                        )
                else:
                    tokens = raw_text.split()
                    if tokens:
                        w_dur = (end_sec - start_sec) / max(1, len(tokens))
                        for i, tok in enumerate(tokens):
                            w_st = start_sec + i * w_dur
                            w_et = w_st + w_dur
                            word_objs.append(
                                WordTimestamp(
                                    word=tok,
                                    start_timestamp_seconds=round(w_st, 3),
                                    end_timestamp_seconds=round(w_et, 3),
                                    confidence=conf,
                                )
                            )

                segments.append(
                    TranscriptSegment(
                        segment_id=(
                            UUID(str(item["segment_id"])) if "segment_id" in item else uuid4()
                        ),
                        video_id=v_id,
                        start_timestamp_seconds=start_sec,
                        end_timestamp_seconds=end_sec,
                        raw_text=raw_text,
                        normalized_text=norm_text,
                        language=seg_lang,
                        confidence=conf,
                        provider=self.provider_name,
                        words=word_objs,
                        speaker_id=item.get("speaker_id"),
                        no_speech_prob=float(item.get("no_speech_prob", 0.01)),
                        attributes=item.get("attributes", {}),
                    )
                )
            return segments

        # Default synthetic generation
        duration = resolved.duration_seconds if resolved.duration_seconds > 0 else 4.0
        seg_dur = self.config.segment_duration_seconds
        num_segments = max(1, int(round(duration / seg_dur)))

        if target_lang == "hi":
            default_texts = [
                "नमस्ते यह एक परीक्षण ऑडियो संदेश है",
                "भारत एक महान देश है",
                "प्रौद्योगिकी निरंतर विकसित हो रही है",
            ]
        else:
            default_texts = [
                self.config.default_text,
                "Welcome to the VIDEX video intelligence pipeline",
                "Speech recognition transcript segment",
            ]

        results: list[TranscriptSegment] = []
        for i in range(num_segments):
            st = i * seg_dur
            et = min(duration, (i + 1) * seg_dur)
            raw = default_texts[i % len(default_texts)]
            norm = normalize_transcript(raw)

            tokens = raw.split()
            word_objs = []
            if tokens:
                w_dur = (et - st) / len(tokens)
                for j, tok in enumerate(tokens):
                    word_objs.append(
                        WordTimestamp(
                            word=tok,
                            start_timestamp_seconds=round(st + j * w_dur, 3),
                            end_timestamp_seconds=round(st + (j + 1) * w_dur, 3),
                            confidence=self.config.default_confidence,
                        )
                    )

            results.append(
                TranscriptSegment(
                    segment_id=uuid4(),
                    video_id=v_id,
                    start_timestamp_seconds=st,
                    end_timestamp_seconds=et,
                    raw_text=raw,
                    normalized_text=norm,
                    language=target_lang,
                    confidence=self.config.default_confidence,
                    provider=self.provider_name,
                    words=word_objs,
                    no_speech_prob=0.01,
                )
            )

        return results
