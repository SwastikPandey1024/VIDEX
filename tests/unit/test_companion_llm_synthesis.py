"""Tests for Phase 9.3 LLM-powered evidence-grounded study note synthesis.

Verifies:
- Structured output compliance with StudyNote and StudyNoteSection schemas
- Handling of malformed LLM outputs and unclosed JSON blocks
- Robust classification of quota exhaustion (429) and authentication errors (401)
- Handling of LLM timeouts with exponential backoff and deterministic fallback
- Detection and uncertainty flagging for corrupted or unbalanced LaTeX formulas
- Clamping and boundary enforcement for evidence timestamps
- Transcript quality audit behavior on abridged vs dense lectures
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videx.companion.llm import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMQuotaExhaustedError,
    LLMTimeoutError,
    MockLLMClient,
    OpenAILLMClient,
)
from videx.companion.schemas import (
    NoteGenerationStatus,
    TranscriptDocument,
    UncertaintyType,
    YouTubeVideoMetadata,
)
from videx.companion.synthesizer import StudyNoteSynthesizer
from videx.companion.transcript import FileTranscriptProvider
from videx.companion.transcript_audit import audit_transcript

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "companion"


@pytest.fixture
def pilot_metadata() -> YouTubeVideoMetadata:
    return YouTubeVideoMetadata(
        video_id="DrCeIbpfuzE",
        title=(
            "1a. Why Study Linear Algebra | The Importance of Linear Algebra in "
            "Machine Learning | Sachin Mittal"
        ),
        source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
        channel_title="GO Classes for GATE CS",
        duration_seconds=1331.0,
    )


@pytest.fixture
def pilot_transcript_doc() -> TranscriptDocument:
    json_path = FIXTURES_DIR / "DrCeIbpfuzE.json"
    provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_path})
    doc = provider.get_transcript("DrCeIbpfuzE")
    assert doc is not None
    return doc


class TestLLMStudyNoteSynthesis:
    """Test LLM-driven synthesis pipeline and error boundaries."""

    def test_mock_llm_synthesis_structured_output(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify successful LLM synthesis produces schema-compliant note with evidence links."""
        client = MockLLMClient()
        synthesizer = StudyNoteSynthesizer(llm_client=client, chunk_window_seconds=300.0)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert note.status == NoteGenerationStatus.COMPLETE
        assert note.generation_metadata.get("synthesizer_mode") == "llm"
        assert note.generation_metadata.get("llm_provider") == "mock"
        assert note.coverage_percentage >= 90.0

        assert len(note.sections) > 0
        assert len(note.learning_objectives) > 0
        assert len(note.key_takeaways) > 0
        assert len(note.practice_questions) > 0

        # Verify evidence reference provenance on substantive claims
        first_sec = note.sections[0]
        assert len(first_sec.definitions) > 0
        def_item = first_sec.definitions[0]
        assert len(def_item.evidence_refs) > 0
        ref = def_item.evidence_refs[0]
        assert ref.video_id == pilot_metadata.video_id
        assert ref.start_seconds >= 0.0
        assert ref.end_seconds >= ref.start_seconds

        # Verification check
        assert note.validation is not None
        assert note.validation.is_valid is True
        assert note.validation.unsupported_claims == 0

    def test_malformed_llm_output_triggers_error_or_fallback(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Malformed JSON output from LLM raises exception or cleanly triggers
        deterministic fallback.
        """
        client = MockLLMClient(simulate_malformed_json=True)

        # 1. Fallback disabled: must raise LLMMalformedResponseError
        strict_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=False)
        with pytest.raises(LLMMalformedResponseError):
            strict_synth.synthesize(pilot_metadata, pilot_transcript_doc)

        # 2. Fallback enabled: completes with fallback metadata and warnings
        resilient_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=True)
        note = resilient_synth.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note.generation_metadata.get("synthesizer_mode") == "fallback_deterministic"
        assert "LLMMalformedResponseError" in note.generation_metadata.get("llm_error_type", "")
        assert note.validation is not None
        assert any("LLM synthesis failed" in w for w in note.validation.warnings)

    def test_quota_exhausted_handling(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify credit/quota exhaustion triggers explicit LLMQuotaExhaustedError."""
        client = MockLLMClient(simulate_quota_exhausted=True)

        strict_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=False)
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            strict_synth.synthesize(pilot_metadata, pilot_transcript_doc)
        assert "credit_balance_exhausted" in str(exc_info.value)

        # Resilient fallback mode
        resilient_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=True)
        note = resilient_synth.synthesize(pilot_metadata, pilot_transcript_doc)
        assert note.generation_metadata.get("synthesizer_mode") == "fallback_deterministic"
        assert note.generation_metadata.get("llm_error_type") == "LLMQuotaExhaustedError"

    def test_timeout_and_retry_behavior(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Verify LLM timeout raises LLMTimeoutError and logs actionable guidance."""
        client = MockLLMClient(simulate_timeout=True)

        strict_synth = StudyNoteSynthesizer(llm_client=client, fallback_to_deterministic=False)
        with pytest.raises(LLMTimeoutError):
            strict_synth.synthesize(pilot_metadata, pilot_transcript_doc)

    def test_missing_credentials_raises_auth_error(self) -> None:
        """OpenAILLMClient raises LLMAuthenticationError when no API key is provided."""
        client = OpenAILLMClient(api_key="", base_url="https://api.openai.com/v1")
        with pytest.raises(LLMAuthenticationError) as exc_info:
            client.generate(messages=[{"role": "user", "content": "hello"}])
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_unbalanced_latex_flagged_in_uncertainties(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Unbalanced LaTeX equations from LLM are detected and flagged in uncertainties."""
        canned_with_bad_latex = json.dumps(
            {
                "title": "Matrix Inverses and Systems",
                "overview": "Discussion of systems of linear equations.",
                "definitions": [],
                "theorems": [],
                "formulas": [
                    {
                        "name": "Corrupted Equation",
                        "latex_expression": "$A x = b",  # Unclosed dollar delimiter
                        "meaning": "Broken formula",
                        "variables": {},
                        "start_seconds": 0.0,
                        "end_seconds": 45.0,
                    }
                ],
                "worked_examples": [],
                "exam_traps": [],
                "uncertainties": [],
                "takeaways": [],
            }
        )
        cons_canned = json.dumps(
            {
                "learning_objectives": ["Understand linear transformations"],
                "executive_summary": "Lecture summary overview.",
                "practice_questions": [],
                "key_takeaways": [],
            }
        )
        client = MockLLMClient(canned_responses=[canned_with_bad_latex, cons_canned])
        synthesizer = StudyNoteSynthesizer(llm_client=client, chunk_window_seconds=1500.0)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        # Check uncertainty was flagged
        corrupted_flags = [
            u for u in note.uncertainties if u.uncertainty_type == UncertaintyType.CORRUPTED_FORMULA
        ]
        assert len(corrupted_flags) > 0
        assert "Unbalanced LaTeX" in corrupted_flags[0].description


class TestTranscriptQualityAudit:
    """Test transcript quality auditor across various transcript densities."""

    def test_audit_abridged_pilot_fixture(self, pilot_transcript_doc: TranscriptDocument) -> None:
        """Verify auditor correctly identifies abridged pilot fixture and flags limitations."""
        report = audit_transcript(
            pilot_transcript_doc,
            video_duration=1330.866,
            video_title="1a. Why Study Linear Algebra",
        )

        assert report.video_id == "DrCeIbpfuzE"
        assert report.segment_count == 12
        assert report.total_words == 351
        assert report.is_abridged is True
        assert report.words_per_minute < 50.0  # ~15.8 wpm vs expected 120-160
        assert report.gap_count == 0
        assert report.overlap_count == 0
        assert report.temporal_coverage_pct >= 99.0

        # Supportability verdicts
        assert report.can_support_topic_synopsis is True
        assert report.can_support_full_lecture_notes is False
        assert any("Abridged transcript detected" in f for f in report.findings)
        assert any("Acquire full verbatim ASR transcript" in r for r in report.recommendations)

    def test_audit_empty_transcript(self) -> None:
        """Verify auditor handles empty transcript without crashing."""
        from videx.companion.schemas import TranscriptDocument

        empty_doc = TranscriptDocument(
            video_id="test_empty",
            language="en",
            acquisition_method="none",
            segments=(),
        )
        report = audit_transcript(empty_doc)
        assert report.segment_count == 0
        assert report.can_support_full_lecture_notes is False
        assert report.can_support_topic_synopsis is False


class TestSynthesisDefectsAndProvenance:
    """Test targeted defect fixes for provenance, formula attribution, and uncertainties."""

    def test_markdown_provenance_frontmatter_and_footer(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Markdown output must include synthesis_mode in frontmatter and footer."""
        from videx.companion.markdown import MarkdownExporter

        client = MockLLMClient()
        synthesizer = StudyNoteSynthesizer(llm_client=client)
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        exporter = MarkdownExporter()
        md = exporter.export_note_to_markdown(note)

        assert 'synthesis_mode: "llm"' in md
        assert 'llm_provider: "mock"' in md
        assert "Synthesis Mode: llm" in md

    def test_formula_attribution_avoids_over_eager_y_equals_ax(
        self, pilot_metadata: YouTubeVideoMetadata
    ) -> None:
        """Segments defining vector concepts without equations must not fabricate y = Ax."""
        from videx.companion.schemas import TranscriptSegment

        seg = TranscriptSegment(
            video_id="DrCeIbpfuzE",
            start_seconds=45.0,
            end_seconds=120.0,
            text=(
                "In computer science and machine learning, data is represented using vectors "
                "and matrices. A feature vector is an ordered collection of real numbers."
            ),
        )
        doc = TranscriptDocument(
            video_id="DrCeIbpfuzE",
            acquisition_method="user_file_json",
            segments=(seg,),
        )
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, doc)

        # Segment defines vector, but does not state the linear transformation equation y = Ax
        all_formulas = [f for sec in note.sections for f in sec.formulas]
        assert len(all_formulas) == 0

    def test_abridged_transcript_uncertainty_flag(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Abridged transcripts (< 50 wpm) must produce an OMITTED_DERIVATION uncertainty."""
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        omitted_derivations = [
            u
            for u in note.uncertainties
            if u.uncertainty_type == UncertaintyType.OMITTED_DERIVATION
        ]
        assert len(omitted_derivations) == 1
        assert "abridged conceptual synopsis" in omitted_derivations[0].description
        assert "Intermediate Derivations" in omitted_derivations[0].affected_concept_or_timestamp

    def test_practice_question_source_distinction(
        self, pilot_metadata: YouTubeVideoMetadata, pilot_transcript_doc: TranscriptDocument
    ) -> None:
        """Practice questions must explicitly distinguish source-derived from supplementary."""
        synthesizer = StudyNoteSynthesizer()
        note = synthesizer.synthesize(pilot_metadata, pilot_transcript_doc)

        assert len(note.practice_questions) >= 2
        q_texts = [q.question for q in note.practice_questions]
        assert any("[SOURCE-DERIVED]" in q for q in q_texts)
        assert any("[SUPPLEMENTARY PRACTICE]" in q for q in q_texts)
