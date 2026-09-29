"""Temporal transcript fusion for ASR observations.

Merges repeated, adjacent, and overlapping transcript segments without destroying
underlying raw ASR evidence or creating clock drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from videx.domain.schemas import TranscriptSegment, WordTimestamp

__all__ = ["TemporalTranscriptFusion", "TemporalTranscriptFusionConfig"]


@dataclass(frozen=True)
class TemporalTranscriptFusionConfig:
    """Configuration for temporal transcript fusion."""

    max_gap_seconds: float = 0.5
    min_overlap_seconds: float = 0.0
    similarity_threshold: float = 0.85
    merge_identical_adjacent: bool = True
    deduplicate_words: bool = True


class TemporalTranscriptFusion:
    """Fuses raw ASR transcript segments across time.

    Handles:
    - Repeated utterances in overlapping time windows (common with sliding audio windows).
    - Consecutive utterances with identical or contiguous text.
    - Preserves full provenance back to raw segment IDs.
    """

    def __init__(self, config: TemporalTranscriptFusionConfig | None = None) -> None:
        self.config = config or TemporalTranscriptFusionConfig()

    def fuse(
        self,
        raw_segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Fuse a list of raw transcript segments into clean, deduplicated segments.

        Args:
            raw_segments: Raw segments produced by an ASRProvider.

        Returns:
            List of fused TranscriptSegment records with supporting IDs in attributes.
        """
        if not raw_segments:
            return []

        # Sort chronologically by start timestamp then end timestamp
        sorted_segs = sorted(
            raw_segments,
            key=lambda s: (s.start_timestamp_seconds, s.end_timestamp_seconds),
        )

        fused: list[TranscriptSegment] = []
        current_cluster: list[TranscriptSegment] = [sorted_segs[0]]

        for seg in sorted_segs[1:]:
            last_seg = current_cluster[-1]

            # Check if this segment should be merged with current_cluster
            if self._should_merge(last_seg, seg):
                current_cluster.append(seg)
            else:
                fused.append(self._merge_cluster(current_cluster))
                current_cluster = [seg]

        if current_cluster:
            fused.append(self._merge_cluster(current_cluster))

        return fused

    def _should_merge(self, seg_a: TranscriptSegment, seg_b: TranscriptSegment) -> bool:
        """Determine whether two chronologically adjacent segments should merge."""
        # Must share same video_id and language
        if seg_a.video_id != seg_b.video_id or seg_a.language != seg_b.language:
            return False

        # Check temporal overlap or adjacency
        gap = seg_b.start_timestamp_seconds - seg_a.end_timestamp_seconds
        is_temporally_close = gap <= self.config.max_gap_seconds

        if not is_temporally_close:
            return False

        # Case 1: Identical normalized text
        norm_a = seg_a.normalized_text
        norm_b = seg_b.normalized_text
        if norm_a and norm_b and norm_a == norm_b and self.config.merge_identical_adjacent:
            return True

        # Case 2: One is a substring or prefix of another in overlapping window
        if gap <= 0:  # overlapping
            if norm_a in norm_b or norm_b in norm_a:
                return True

        return False

    def _merge_cluster(self, cluster: list[TranscriptSegment]) -> TranscriptSegment:
        """Merge a cluster of related segments into a single fused TranscriptSegment."""
        if len(cluster) == 1:
            seg = cluster[0]
            # Return copy with supporting_segment_ids preserved in attributes
            attrs = dict(seg.attributes)
            attrs["supporting_segment_ids"] = [str(seg.segment_id)]
            return TranscriptSegment(
                segment_id=seg.segment_id,
                video_id=seg.video_id,
                start_timestamp_seconds=seg.start_timestamp_seconds,
                end_timestamp_seconds=seg.end_timestamp_seconds,
                raw_text=seg.raw_text,
                normalized_text=seg.normalized_text,
                language=seg.language,
                confidence=seg.confidence,
                provider=seg.provider,
                source=seg.source,
                words=seg.words,
                speaker_id=seg.speaker_id,
                no_speech_prob=seg.no_speech_prob,
                attributes=attrs,
            )

        # Multiple segments to merge
        first = cluster[0]
        v_id = first.video_id
        lang = first.language
        provider = first.provider

        start_sec = min(s.start_timestamp_seconds for s in cluster)
        end_sec = max(s.end_timestamp_seconds for s in cluster)

        # Pick representative text from the segment with highest confidence or longest text
        best_seg = max(cluster, key=lambda s: (s.confidence or 0.0, len(s.raw_text)))
        raw_text = best_seg.raw_text
        norm_text = best_seg.normalized_text

        # Combine confidences
        conf_values = [s.confidence for s in cluster if s.confidence is not None]
        mean_conf = sum(conf_values) / len(conf_values) if conf_values else None

        # Combine word timestamps
        all_words: list[WordTimestamp] = []
        seen_words: set[tuple[str, float, float]] = set()
        for s in cluster:
            for w in s.words:
                key = (
                    w.word,
                    round(w.start_timestamp_seconds, 2),
                    round(w.end_timestamp_seconds, 2),
                )
                if self.config.deduplicate_words:
                    if key not in seen_words:
                        seen_words.add(key)
                        all_words.append(w)
                else:
                    all_words.append(w)

        # Sort words chronologically
        all_words.sort(key=lambda w: (w.start_timestamp_seconds, w.end_timestamp_seconds))

        supporting_ids = [str(s.segment_id) for s in cluster]
        merged_attrs: dict[str, Any] = {
            "supporting_segment_ids": supporting_ids,
            "fusion_count": len(cluster),
        }

        return TranscriptSegment(
            segment_id=uuid4(),
            video_id=v_id,
            start_timestamp_seconds=start_sec,
            end_timestamp_seconds=end_sec,
            raw_text=raw_text,
            normalized_text=norm_text,
            language=lang,
            confidence=mean_conf,
            provider=provider,
            source=first.source,
            words=all_words,
            speaker_id=best_seg.speaker_id,
            attributes=merged_attrs,
        )
