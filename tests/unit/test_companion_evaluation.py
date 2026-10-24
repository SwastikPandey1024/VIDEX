"""Unit tests for companion evaluation metrics and quality gates."""

from pathlib import Path

from videx.companion.evaluation import evaluate_study_note
from videx.companion.schemas import (
    YouTubeVideoMetadata,
)
from videx.companion.synthesizer import StudyNoteSynthesizer
from videx.companion.transcript import FileTranscriptProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestCompanionEvaluation:
    """Test evaluation metrics and quality gates."""

    def test_evaluate_grounded_study_note(self):
        meta = YouTubeVideoMetadata(
            video_id="DrCeIbpfuzE",
            title=(
                "1a. Why Study Linear Algebra | The Importance of Linear Algebra in "
                "Machine Learning | Sachin Mittal"
            ),
            source_url="https://www.youtube.com/watch?v=DrCeIbpfuzE",
            duration_seconds=1331.0,
        )
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        doc = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file}).get_transcript(
            "DrCeIbpfuzE"
        )
        note = StudyNoteSynthesizer().synthesize(meta, doc)

        ref_concepts = [
            "vector",
            "matrix",
            "linear transformation",
            "machine learning",
        ]

        result = evaluate_study_note(note, reference_concepts=ref_concepts)

        assert result.video_id == "DrCeIbpfuzE"
        assert result.total_claims > 0
        assert result.grounded_claims == result.total_claims
        assert result.unsupported_claim_rate == 0.0
        assert result.latex_equation_fidelity == 1.0
        assert result.timestamp_accuracy_rate == 1.0
        assert result.concept_coverage >= 0.75
        assert result.is_acceptable

    def test_evaluate_empty_transcript_note(self):
        meta = YouTubeVideoMetadata(
            video_id="empty_vid",
            title="Empty Video",
            source_url="https://youtube.com/watch?v=empty_vid",
            duration_seconds=100.0,
        )
        note = StudyNoteSynthesizer().synthesize(meta, transcript_doc=None)
        result = evaluate_study_note(note)

        assert result.total_claims == 0
        assert result.unsupported_claim_rate == 0.0
        assert result.is_acceptable
