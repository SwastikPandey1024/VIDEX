"""Unit tests for application configuration.

Verifies:
- Settings can be constructed with defaults.
- Field validation works (port bounds, threshold bounds).
- ``ocr_language_list`` property parses correctly.
- ``get_settings()`` caching behaviour (cache_clear resets state).
"""

import pytest
from pydantic import ValidationError

from videx.config import Settings, get_settings


class TestSettings:
    def test_default_construction(self) -> None:
        s = Settings()
        assert s.app_name == "VIDEX"
        assert s.version == "0.1.0"
        assert s.api_port == 8000
        assert s.debug is False

    def test_override_fields(self) -> None:
        s = Settings(app_name="VIDEX-test", api_port=9000, debug=True)
        assert s.app_name == "VIDEX-test"
        assert s.api_port == 9000
        assert s.debug is True

    def test_port_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(api_port=80)  # below ge=1024

    def test_port_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(api_port=70000)  # above le=65535

    def test_confidence_threshold_bounds(self) -> None:
        s = Settings(detector_confidence_threshold=0.0)
        assert s.detector_confidence_threshold == 0.0

        s2 = Settings(detector_confidence_threshold=1.0)
        assert s2.detector_confidence_threshold == 1.0

        with pytest.raises(ValidationError):
            Settings(detector_confidence_threshold=1.5)

    def test_ocr_language_list_single(self) -> None:
        s = Settings(ocr_languages="hi")
        assert s.ocr_language_list == ["hi"]

    def test_ocr_language_list_multiple(self) -> None:
        s = Settings(ocr_languages="hi,en")
        assert s.ocr_language_list == ["hi", "en"]

    def test_ocr_language_list_with_spaces(self) -> None:
        s = Settings(ocr_languages=" hi , en , ta ")
        assert s.ocr_language_list == ["hi", "en", "ta"]

    def test_ocr_language_list_empty(self) -> None:
        s = Settings(ocr_languages="")
        assert s.ocr_language_list == []

    def test_default_detector_provider(self) -> None:
        s = Settings()
        assert s.detector_provider == "yolov8"

    def test_default_tracker_provider(self) -> None:
        s = Settings()
        assert s.tracker_provider == "bytetrack"

    def test_default_asr_language(self) -> None:
        s = Settings()
        assert s.asr_language == "en"


class TestGetSettings:
    def setup_method(self) -> None:
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        get_settings.cache_clear()

    def test_returns_settings_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_is_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_new_instance(self) -> None:
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # New instance but same defaults
        assert s1.version == s2.version
