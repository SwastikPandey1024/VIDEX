"""Configurable OCR Router for language- and script-aware OCR provider dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from videx.domain.schemas import Frame, OCRObservation
from videx.ingestion.base import DecodedFrame
from videx.providers.base import OCRProvider


@dataclass
class OCRRouterConfig:
    """Configuration mapping languages and scripts to specific OCR providers.

    All route mappings are configurable rather than hardcoded in business logic.
    """

    default_provider_name: str = "general"
    default_provider_key: str | None = None
    providers: dict[str, OCRProvider] | None = None
    language_map: dict[str, str] | None = None
    # Mapping from ISO language code to route key (e.g., 'en' -> 'general', 'hi' -> 'indic')
    language_routes: dict[str, str] = field(
        default_factory=lambda: {
            "en": "general",
            "latin": "general",
            "zh": "general",
            "es": "general",
            "fr": "general",
            "de": "general",
            "hi": "indic",
            "mr": "indic",
            "sa": "indic",
            "devanagari": "indic",
        }
    )
    # Mapping from script name to route key
    script_routes: dict[str, str] = field(
        default_factory=lambda: {
            "latin": "general",
            "devanagari": "indic",
        }
    )

    def __post_init__(self) -> None:
        if self.default_provider_key is not None:
            self.default_provider_name = self.default_provider_key
        if self.language_map is not None:
            self.language_routes.update(self.language_map)


class OCRRouter:
    """Selects and dispatches to appropriate OCRProvider based on language/script context.

    Rule: Never represent PP-OCRv6 as the Hindi engine.
    General Latin/English is routed to general providers; Hindi/Devanagari is explicitly
    routed to specialized Indic providers.
    """

    def __init__(
        self,
        providers: dict[str, OCRProvider] | OCRRouterConfig | None = None,
        config: OCRRouterConfig | None = None,
    ) -> None:
        if isinstance(providers, OCRRouterConfig):
            self.config = providers
            providers = self.config.providers
        else:
            self.config = config or OCRRouterConfig()

        if providers is None:
            if self.config.providers:
                providers = self.config.providers
            else:
                from videx.ocr.paddle import PaddleGeneralOCRProvider, PaddleIndicOCRProvider
                providers = {
                    "general": PaddleGeneralOCRProvider(),
                    "indic": PaddleIndicOCRProvider(),
                }
        if not providers:
            raise ValueError("OCRRouter requires at least one registered provider.")
        self._providers = dict(providers)

        # Fallback provider if requested route is missing
        if self.config.default_provider_name in self._providers:
            self._default_provider = self._providers[self.config.default_provider_name]
        else:
            self._default_provider = next(iter(self._providers.values()))

    def register_provider(self, route_key: str, provider: OCRProvider) -> None:
        """Register a provider instance under a route key."""
        self._providers[route_key] = provider

    def map_language_route(self, language: str, route_key: str) -> None:
        """Map an ISO language code to a provider route key."""
        self.config.language_routes[language.lower()] = route_key

    def map_script_route(self, script: str, route_key: str) -> None:
        """Map a script name to a provider route key."""
        self.config.script_routes[script.lower()] = route_key

    def get_provider(
        self,
        language: str | None = None,
        script: str | None = None,
    ) -> OCRProvider:
        """Resolve the appropriate OCRProvider for the requested language/script.

        Priority order:
        1. Explicit language code lookup in language_routes.
        2. Script name lookup in script_routes.
        3. Provider direct capability match.
        4. Default fallback provider.
        """
        # 1. Language lookup
        if language:
            route_key = self.config.language_routes.get(language.lower())
            if route_key and route_key in self._providers:
                return self._providers[route_key]

        # 2. Script lookup
        if script:
            route_key = self.config.script_routes.get(script.lower())
            if route_key and route_key in self._providers:
                return self._providers[route_key]

        # 3. Direct provider capability inspection
        if language:
            lang_clean = language.lower()
            for p in self._providers.values():
                if lang_clean in (lang_code.lower() for lang_code in p.supported_languages):
                    return p

        # 4. Fallback
        return self._default_provider

    def detect_text(
        self,
        frame: DecodedFrame | Frame,
        language: str | None = None,
        script: str | None = None,
    ) -> list[OCRObservation]:
        """Route frame to the appropriate OCRProvider and return detections."""
        provider = self.get_provider(language=language, script=script)
        return provider.detect_text(frame)

    def detect_text_batch(
        self,
        frames: Sequence[DecodedFrame | Frame],
        language: str | None = None,
        script: str | None = None,
    ) -> list[list[OCRObservation]]:
        """Batch route frames."""
        provider = self.get_provider(language=language, script=script)
        return provider.detect_text_batch(frames)
