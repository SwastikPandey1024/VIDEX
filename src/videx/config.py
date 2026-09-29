"""VIDEX application configuration.

All configuration is read from environment variables (or a .env file).
Application code must never read os.environ directly — always use the
Settings object to ensure type safety and a single source of truth.

Usage::

    from videx.config import get_settings

    settings = get_settings()
    print(settings.api_port)
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VIDEX runtime configuration.

    Values are read (in priority order) from:
    1. Environment variables
    2. .env file (if present)
    3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────
    app_name: str = Field(default="VIDEX", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # ── API Server ────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="Bind host")  # noqa: S104
    api_port: int = Field(default=8000, ge=1024, le=65535, description="Bind port")

    # ── Storage paths ─────────────────────────────────────────
    data_dir: str = Field(default="data", description="Root data directory")
    frames_dir: str = Field(default="data/frames", description="Decoded frames directory")
    evidence_dir: str = Field(default="data/evidence", description="Evidence output directory")

    # ── Database (Phase 1+) ───────────────────────────────────
    database_url: str = Field(
        default="",
        description="PostgreSQL async connection URL (required in production)",
    )

    # ── Perception / Detection (Phase 1+) ─────────────────────
    detector_provider: str = Field(
        default="yolov8",
        description="Detector backend key (matches provider registry)",
    )
    detector_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence",
    )
    detector_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS",
    )
    detector_device: str = Field(
        default="cpu",
        description="Inference device: 'cpu', 'cuda', 'cuda:0', 'mps'",
    )
    detector_model_path: str = Field(
        default="models/weights/detector.pt",
        description="Path to detector model weights",
    )

    # ── Tracking (Phase 1+) ───────────────────────────────────
    tracker_provider: str = Field(default="bytetrack", description="Tracker backend key")
    tracker_max_lost_frames: int = Field(
        default=30,
        ge=1,
        description="Frames before a lost track is deleted",
    )

    # ── OCR (Phase 1+) ────────────────────────────────────────
    ocr_provider: str = Field(default="easyocr", description="OCR backend key")
    ocr_languages: str = Field(
        default="hi,en",
        description="Comma-separated ISO language codes for OCR",
    )

    # ── ASR / Audio (Phase 1+ / Phase 4.0) ────────────────────
    asr_provider: str = Field(default="faster_whisper", description="ASR backend key")
    asr_model_size: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large-v3",
    )
    asr_language: str = Field(default="en", description="Primary ASR language code ('en', 'hi')")
    asr_device: str = Field(default="cpu", description="Inference device ('cpu', 'cuda')")
    asr_compute_type: str = Field(
        default="int8", description="Quantization/compute type ('int8', 'float16', 'float32')"
    )

    # ── VLM / Semantic Reasoning (Phase 2+) ──────────────────
    vlm_provider: str = Field(default="openai", description="VLM backend key")
    vlm_model: str = Field(default="gpt-4o", description="VLM model identifier")
    openai_api_key: str = Field(default="", description="OpenAI API key (not logged)")

    @property
    def ocr_language_list(self) -> list[str]:
        """Return OCR languages as a list."""
        return [lang.strip() for lang in self.ocr_languages.split(",") if lang.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Using lru_cache ensures the .env file is parsed exactly once per process.
    In tests, call ``get_settings.cache_clear()`` before overriding settings.
    """
    return Settings()
