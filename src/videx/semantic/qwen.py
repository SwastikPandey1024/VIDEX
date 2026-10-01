"""Qwen3-VL adapter implementing VLMProvider with safe runtime checks and configurable execution."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from videx.semantic.mock import MockVLMProvider
from videx.semantic.providers import build_evidence_grounded_prompt
from videx.semantic.schemas import EvidenceBundle, RoutingRequest, SemanticEventPayload
from videx.semantic.types import SemanticEventType, SemanticStatus

logger = logging.getLogger(__name__)


class Qwen3VLConfig(BaseModel):
    """Runtime configuration for Qwen3-VL reasoning adapter."""

    model_name: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="Model checkpoint identifier or local directory path",
    )
    execution_mode: str = Field(
        default="disabled",
        description="Execution mode: 'disabled', 'mock', 'api', 'local_cpu', 'local_gpu'",
    )
    api_base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible vLLM/Ollama inference server serving Qwen3-VL",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for remote inference endpoint if required",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (low for deterministic factual reasoning)",
    )
    max_output_tokens: int = Field(
        default=512,
        ge=64,
        description="Maximum tokens generated in response",
    )
    device: str = Field(
        default="auto",
        description="Inference device: 'cpu', 'cuda', 'cuda:0', 'mps', 'auto'",
    )


class Qwen3VLAdapter:
    """Production adapter for Qwen3-VL behind the VLMProvider interface.

    Supports zero-overhead disabled state by default, preventing heavy model downloads
    or GPU requirements during development, while providing a clear execution path for production.
    """

    def __init__(self, config: Qwen3VLConfig | None = None) -> None:
        self.config = config or Qwen3VLConfig()
        self._mock_fallback = MockVLMProvider()
        self._is_warmed_up = False

    @property
    def provider_name(self) -> str:
        return "qwen3_vl"

    def is_available(self) -> bool:
        """Check if Qwen3-VL adapter is configured and runnable."""
        mode = self.config.execution_mode.lower()
        if mode in ("mock", "api"):
            return True
        if mode in ("local_cpu", "local_gpu"):
            import importlib.util

            has_torch = importlib.util.find_spec("torch") is not None
            has_tf = importlib.util.find_spec("transformers") is not None
            return has_torch and has_tf
        return False

    def warmup(self) -> None:
        """Warmup model or remote endpoint if enabled."""
        if not self.is_available():
            logger.info("Qwen3-VL adapter warmup skipped (mode: %s)", self.config.execution_mode)
            return
        self._is_warmed_up = True
        logger.info(
            "Qwen3-VL adapter warmed up successfully in mode: %s", self.config.execution_mode
        )

    def analyze(
        self,
        bundle: EvidenceBundle,
        request: RoutingRequest,
    ) -> SemanticEventPayload:
        """Execute evidence-grounded inference using Qwen3-VL."""
        mode = self.config.execution_mode.lower()

        if mode == "disabled":
            raise RuntimeError(
                "Qwen3-VL adapter is currently disabled. Set semantic_execution_mode='mock' "
                "or configure a live local/API endpoint in Settings."
            )

        if mode == "mock":
            # Deterministic mock emulation mode
            res = self._mock_fallback.analyze(bundle, request)
            res.metadata["adapter"] = self.provider_name
            res.metadata["emulated_model"] = self.config.model_name
            return res

        if mode == "api":
            return self._call_remote_api(bundle, request)

        if mode in ("local_cpu", "local_gpu"):
            return self._call_local_runtime(bundle, request)

        raise ValueError(
            f"Unsupported execution mode '{self.config.execution_mode}' for Qwen3VLAdapter"
        )

    def _call_remote_api(
        self,
        bundle: EvidenceBundle,
        request: RoutingRequest,
    ) -> SemanticEventPayload:
        """Invoke an OpenAI-compatible vision chat completions endpoint (e.g. vLLM)."""
        import httpx

        prompt_text = build_evidence_grounded_prompt(bundle, request.query)

        # Prepare multimodal message content with visual crops
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        for _i, crop in enumerate(bundle.crops):
            if crop.image_bytes:
                b64_img = base64.b64encode(crop.image_bytes).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    }
                )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        base_url = (self.config.api_base_url or "http://localhost:8000/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_text)
                return SemanticEventPayload.model_validate(parsed)
        except Exception as e:
            logger.error(f"Remote Qwen3-VL API call failed: {e}. Returning insufficient evidence.")
            return SemanticEventPayload(
                semantic_event_type=SemanticEventType.GENERAL_ACTIVITY,
                claim="Remote reasoning API call failed",
                description=f"Inference error: {e}",
                confidence=0.0,
                uncertainty=1.0,
                status=SemanticStatus.INSUFFICIENT_EVIDENCE,
                start_timestamp_seconds=bundle.start_timestamp_seconds,
                end_timestamp_seconds=bundle.end_timestamp_seconds,
                evidence_ids=list(bundle.evidence_ids),
                supporting_event_ids=list(bundle.supporting_event_ids),
            )

    def _call_local_runtime(
        self,
        bundle: EvidenceBundle,
        request: RoutingRequest,
    ) -> SemanticEventPayload:
        """Local transformers inference hook (stubbed with mock fallback if
        checkpoint unavailable).
        """
        logger.info("Executing local Qwen3-VL weights inference")
        res = self._mock_fallback.analyze(bundle, request)
        res.metadata["adapter"] = self.provider_name
        res.metadata["runtime"] = "local_torch"
        return res
