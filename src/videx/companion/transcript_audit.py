"""Transcript quality audit module for VIDEX Learning Companion.

Performs rigorous audit of transcript fidelity:
- Acquisition method and language verification
- Temporal continuity (gap and overlap analysis)
- Speaking rate and text density analysis (words per minute vs expected 120-160 wpm)
- Identification of empty or duplicate segments
- Honest assessment of whether transcript is an abridged synopsis vs verbatim ASR
- Supportability verdict for full-lecture technical notes
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from videx.companion.schemas import TranscriptDocument

logger = logging.getLogger(__name__)

# Expected conversational/technical lecture speaking rate
EXPECTED_MIN_WPM = 100.0
EXPECTED_MAX_WPM = 180.0
TYPICAL_LECTURE_WPM = 135.0


class TranscriptAuditReport(BaseModel):
    """Detailed audit report on transcript completeness, continuity, and density."""

    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="Target video identifier")
    video_title: str = Field(default="", description="Video title")
    acquisition_method: str = Field(..., description="Method used to acquire transcript")
    language: str = Field(default="en", description="Transcript language code")
    is_auto_generated: bool = Field(default=False, description="Whether captions are ASR-generated")

    # Segment and temporal metrics
    segment_count: int = Field(default=0, ge=0, description="Total timestamped segments")
    start_seconds: float = Field(default=0.0, ge=0.0, description="First segment start timestamp")
    end_seconds: float = Field(default=0.0, ge=0.0, description="Last segment end timestamp")
    transcript_span_seconds: float = Field(
        default=0.0, ge=0.0, description="Span from first start to last end"
    )
    video_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Total video duration in seconds"
    )
    temporal_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Transcript span / video duration percentage"
    )

    # Text density and speaking rate
    total_words: int = Field(default=0, ge=0, description="Total word count in transcript")
    total_characters: int = Field(default=0, ge=0, description="Total character count")
    words_per_minute: float = Field(
        default=0.0, ge=0.0, description="Effective speaking rate across duration"
    )
    expected_word_count_min: int = Field(
        default=0, ge=0, description="Expected minimum words for this duration"
    )
    expected_word_count_max: int = Field(
        default=0, ge=0, description="Expected maximum words for this duration"
    )
    text_density_ratio: float = Field(
        default=0.0, ge=0.0, description="Actual words / typical expected words"
    )

    # Anomaly counts
    gap_count: int = Field(default=0, ge=0, description="Number of timestamp gaps > 2.0s")
    max_gap_seconds: float = Field(
        default=0.0, ge=0.0, description="Largest gap duration in seconds"
    )
    overlap_count: int = Field(default=0, ge=0, description="Number of timestamp overlaps > 0.5s")
    max_overlap_seconds: float = Field(
        default=0.0, ge=0.0, description="Largest overlap duration in seconds"
    )
    empty_segment_count: int = Field(
        default=0, ge=0, description="Number of segments with whitespace-only text"
    )
    duplicate_segment_count: int = Field(
        default=0, ge=0, description="Number of consecutive or identical text segments"
    )

    # Qualitative verdict
    is_abridged: bool = Field(
        default=False,
        description="True if speaking rate is significantly below conversational speed (< 50 wpm)",
    )
    can_support_full_lecture_notes: bool = Field(
        default=False,
        description="Whether transcript can ground step-by-step mathematical derivations",
    )
    can_support_topic_synopsis: bool = Field(
        default=True,
        description="Whether transcript contains sufficient key concepts for topical overview",
    )
    findings: tuple[str, ...] = Field(
        default_factory=tuple, description="Key empirical observations from audit"
    )
    recommendations: tuple[str, ...] = Field(
        default_factory=tuple, description="Actionable next steps to improve transcript grounding"
    )


def audit_transcript(
    transcript_doc: TranscriptDocument,
    video_duration: float | None = None,
    video_title: str = "",
) -> TranscriptAuditReport:
    """Audit transcript quality, density, temporal continuity, and completeness."""
    segments = transcript_doc.segments
    if not segments:
        return TranscriptAuditReport(
            video_id=transcript_doc.video_id,
            video_title=video_title,
            acquisition_method=transcript_doc.acquisition_method,
            language=transcript_doc.language,
            is_auto_generated=transcript_doc.is_auto_generated,
            can_support_full_lecture_notes=False,
            can_support_topic_synopsis=False,
            findings=("Transcript document contains 0 segments.",),
            recommendations=(
                "Provide user transcript file (.json, .vtt, .srt) or process media with VIDEX ASR.",
            ),
        )

    start_sec = segments[0].start_seconds
    end_sec = segments[-1].end_seconds
    span_sec = max(0.0, end_sec - start_sec)
    duration_sec = video_duration or transcript_doc.total_duration_seconds or span_sec

    # Coverage
    coverage_pct = round(
        min(100.0, (span_sec / duration_sec * 100.0)) if duration_sec > 0 else 100.0, 1
    )

    # Word and character counts
    all_texts = [s.text.strip() for s in segments]
    total_words = sum(len(t.split()) for t in all_texts if t)
    total_chars = sum(len(t) for t in all_texts if t)

    # Speaking rate
    duration_minutes = duration_sec / 60.0 if duration_sec > 0 else 1.0
    wpm = round(total_words / duration_minutes, 2)
    expected_min_words = int(duration_minutes * EXPECTED_MIN_WPM)
    expected_max_words = int(duration_minutes * EXPECTED_MAX_WPM)
    typical_words = int(duration_minutes * TYPICAL_LECTURE_WPM)
    density_ratio = round(total_words / typical_words, 3) if typical_words > 0 else 1.0

    # Gap and overlap detection
    gap_count = 0
    max_gap = 0.0
    overlap_count = 0
    max_overlap = 0.0
    empty_count = 0
    duplicate_count = 0

    for i, seg in enumerate(segments):
        if not seg.text.strip():
            empty_count += 1

        if i > 0:
            prev = segments[i - 1]
            diff = seg.start_seconds - prev.end_seconds
            if diff > 2.0:
                gap_count += 1
                max_gap = max(max_gap, diff)
            elif diff < -0.5:
                overlap_count += 1
                max_overlap = max(max_overlap, abs(diff))

            if seg.text.strip() and seg.text.strip().lower() == prev.text.strip().lower():
                duplicate_count += 1

    findings: list[str] = []
    recommendations: list[str] = []

    # Abridged check
    is_abridged = wpm < 50.0
    if is_abridged:
        findings.append(
            f"Abridged transcript detected: speaking rate is {wpm:.1f} wpm "
            f"(normal lecture rate is {EXPECTED_MIN_WPM:.0f}–{EXPECTED_MAX_WPM:.0f} wpm). "
            f"Transcript contains {total_words} words across {duration_minutes:.1f} minutes "
            f"(expected ~{typical_words} words)."
        )
        recommendations.append(
            "Acquire full verbatim ASR transcript via faster-whisper or official captions "
            "before claiming exhaustive derivation-level coverage."
        )
    else:
        findings.append(
            f"Dense conversational transcript: speaking rate is {wpm:.1f} wpm "
            f"with {total_words} words."
        )

    # Continuity findings
    if gap_count > 0:
        findings.append(
            f"Detected {gap_count} timestamp gaps > 2.0s (maximum gap: {max_gap:.1f}s)."
        )
    else:
        findings.append("Zero timestamp gaps detected: temporal continuity is strictly contiguous.")

    if overlap_count > 0:
        findings.append(
            f"Detected {overlap_count} timestamp overlaps > 0.5s "
            f"(maximum overlap: {max_overlap:.1f}s)."
        )
    else:
        findings.append("Zero timestamp overlaps detected.")

    if empty_count > 0:
        findings.append(f"Found {empty_count} empty/whitespace text segments.")

    if duplicate_count > 0:
        findings.append(f"Found {duplicate_count} duplicated consecutive segments.")

    can_support_full = (not is_abridged) and (coverage_pct >= 85.0) and (gap_count <= 5)
    can_support_synopsis = (total_words >= 150) and (coverage_pct >= 75.0)

    if not can_support_full and can_support_synopsis:
        findings.append(
            "Verdict: Transcript can support conceptual overview, definitions, and high-level "
            "exam traps, but CANNOT support verbatim proofs or step-by-step intermediate algebra "
            "derivations."
        )

    return TranscriptAuditReport(
        video_id=transcript_doc.video_id,
        video_title=video_title,
        acquisition_method=transcript_doc.acquisition_method,
        language=transcript_doc.language,
        is_auto_generated=transcript_doc.is_auto_generated,
        segment_count=len(segments),
        start_seconds=round(start_sec, 2),
        end_seconds=round(end_sec, 2),
        transcript_span_seconds=round(span_sec, 2),
        video_duration_seconds=round(duration_sec, 2),
        temporal_coverage_pct=coverage_pct,
        total_words=total_words,
        total_characters=total_chars,
        words_per_minute=wpm,
        expected_word_count_min=expected_min_words,
        expected_word_count_max=expected_max_words,
        text_density_ratio=density_ratio,
        gap_count=gap_count,
        max_gap_seconds=round(max_gap, 2),
        overlap_count=overlap_count,
        max_overlap_seconds=round(max_overlap, 2),
        empty_segment_count=empty_count,
        duplicate_segment_count=duplicate_count,
        is_abridged=is_abridged,
        can_support_full_lecture_notes=can_support_full,
        can_support_topic_synopsis=can_support_synopsis,
        findings=tuple(findings),
        recommendations=tuple(recommendations),
    )
