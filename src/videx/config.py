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

from pydantic import AliasChoices, Field
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

    # ── Temporal Event Intelligence (Phase 5) ────────────────
    event_lifecycle_confirmation_threshold: int = Field(
        default=2,
        ge=1,
        description="Minimum confirmed frames before confirming appearance (filters noise)",
    )
    event_disappearance_threshold_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Temporal absence (s) before confirming disappearance vs occlusion",
    )
    event_movement_velocity_threshold_px_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Pixel velocity threshold distinguishing stationary dwelling from motion",
    )
    event_direction_change_degrees_threshold: float = Field(
        default=45.0,
        ge=0.0,
        le=180.0,
        description="Angular deflection threshold in degrees required for direction change",
    )
    event_spatial_boundary_tolerance_px: float = Field(
        default=5.0,
        ge=0.0,
        description="Pixel-space buffer around zone boundaries to absorb jitter",
    )
    event_spatial_debounce_interval_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum temporal gap (s) before re-emitting enter/exit events",
    )
    event_temporal_near_interval_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum temporal interval (s) between events to be NEAR_IN_TIME",
    )

    # ── Semantic Intelligence & Router (Phase 6) ─────────────
    semantic_router_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("semantic_router_enabled", "videx_semantic_router_enabled"),
        description="Enable Layer 4 Semantic Router gating and candidate selection",
    )
    semantic_provider: str = Field(
        default="mock",
        validation_alias=AliasChoices("semantic_provider", "videx_semantic_provider"),
        description="VLM provider key: 'mock', 'qwen', 'qwen3_vl', 'openai'",
    )
    semantic_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        validation_alias=AliasChoices("semantic_model", "videx_semantic_model"),
        description="Underlying VLM model checkpoint or deployment identifier",
    )
    semantic_execution_mode: str = Field(
        default="mock",
        validation_alias=AliasChoices("semantic_execution_mode", "videx_semantic_execution_mode"),
        description="Qwen3-VL execution mode: 'disabled', 'mock', 'api', 'local_cpu', 'local_gpu'",
    )
    semantic_api_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("semantic_api_base_url", "videx_semantic_api_base_url"),
        description="Base URL for remote OpenAI-compatible VLM inference server (e.g. vLLM/Ollama)",
    )
    semantic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("semantic_api_key", "videx_semantic_api_key"),
        description="API key for remote inference endpoint if required (never logged)",
    )
    semantic_saliency_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum candidate saliency score to justify VLM invocation",
    )
    semantic_query_relevance_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum query relevance score required when a query is provided",
    )
    semantic_max_tokens_per_minute: int = Field(
        default=100_000,
        ge=1000,
        description="Sliding-window token budget cap to prevent VLM cost overruns",
    )
    semantic_max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Sliding-window maximum VLM requests permitted per minute",
    )
    semantic_crop_margin: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Context margin fraction added to bounding boxes for visual crops",
    )
    semantic_cache_ttl_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Time-to-live in seconds for cached semantic inference results",
    )

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
