"""Unit tests for transcript provider interfaces and file parsers."""

from pathlib import Path

import pytest

from videx.companion.schemas import TranscriptAcquisitionMethod
from videx.companion.transcript import (
    CompositeTranscriptProvider,
    FileTranscriptProvider,
    YouTubeCaptionProvider,
    parse_timestamp_seconds,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "companion"


class TestTranscriptParsers:
    """Test parsing across multiple caption formats."""

    def test_parse_timestamp_seconds(self):
        assert parse_timestamp_seconds("00:00:15.500") == pytest.approx(15.5)
        assert parse_timestamp_seconds("01:23.456") == pytest.approx(83.456)
        assert parse_timestamp_seconds("01:02:03,400") == pytest.approx(3723.4)
        assert parse_timestamp_seconds("15.5") == pytest.approx(15.5)
        assert parse_timestamp_seconds("invalid") == 0.0

    def test_parse_json_transcript(self):
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})
        doc = provider.get_transcript("DrCeIbpfuzE")

        assert doc is not None
        assert doc.video_id == "DrCeIbpfuzE"
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_JSON
        assert len(doc.segments) == 12
        assert doc.segments[0].start_seconds == 0.0
        assert doc.segments[0].end_seconds == 45.0
        assert "Welcome to Linear Algebra" in doc.segments[0].text
        assert doc.total_duration_seconds == 1331.0
        assert doc.word_count > 100

    def test_parse_vtt_transcript(self):
        vtt_file = FIXTURES_DIR / "sample_lecture.vtt"
        provider = FileTranscriptProvider(file_map={"vtt_vid": vtt_file})
        doc = provider.get_transcript("vtt_vid")

        assert doc is not None
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_VTT
        assert len(doc.segments) == 3
        assert doc.segments[0].start_seconds == 0.0
        assert doc.segments[0].end_seconds == pytest.approx(15.5)
        assert doc.segments[1].start_seconds == pytest.approx(15.5)
        assert doc.segments[2].end_seconds == pytest.approx(90.0)

    def test_parse_srt_transcript(self):
        srt_file = FIXTURES_DIR / "sample_lecture.srt"
        provider = FileTranscriptProvider(file_map={"srt_vid": srt_file})
        doc = provider.get_transcript("srt_vid")

        assert doc is not None
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_SRT
        assert len(doc.segments) == 3
        assert doc.segments[0].start_seconds == 0.0
        assert doc.segments[0].end_seconds == pytest.approx(15.5)
        assert "foundational pillar" in doc.segments[1].text

    def test_missing_transcript_returns_none(self, tmp_path):
        provider = FileTranscriptProvider(transcript_dir=tmp_path)
        doc = provider.get_transcript("non_existent_id")
        assert doc is None

    def test_corrupted_json_handled_safely(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{corrupted: json [", encoding="utf-8")
        provider = FileTranscriptProvider(file_map={"bad_vid": bad_json})
        doc = provider.get_transcript("bad_vid")
        assert doc is None


class TestCompositeTranscriptProvider:
    """Test provider chaining and priority fallback."""

    def test_composite_fallback_to_file(self):
        json_file = FIXTURES_DIR / "DrCeIbpfuzE.json"
        file_provider = FileTranscriptProvider(file_map={"DrCeIbpfuzE": json_file})
        yt_provider = YouTubeCaptionProvider()

        composite = CompositeTranscriptProvider(providers=[yt_provider, file_provider])
        doc = composite.get_transcript("DrCeIbpfuzE")

        assert doc is not None
        assert doc.video_id == "DrCeIbpfuzE"
        assert doc.acquisition_method == TranscriptAcquisitionMethod.USER_FILE_JSON

    def test_composite_returns_none_when_all_fail(self):
        composite = CompositeTranscriptProvider(providers=[YouTubeCaptionProvider()])
        doc = composite.get_transcript("any_id")
        assert doc is None
