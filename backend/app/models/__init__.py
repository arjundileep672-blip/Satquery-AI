"""
VLM / LLM Model Adapters Package
Provides provider-independent VLM and LLM model instantiation.

Supported Adapters:
  - OllamaVLMAdapter: Local offline LLM (Mistral, LLaVA) via Ollama REST API.
  - GeminiVLMAdapter: Cloud multimodal inference via Google Gemini REST API.
  - MockVLMAdapter: Deterministic synthetic responses for reproducible offline tests.

Selection Logic:
  1. If LLM_PROVIDER="ollama" (or provider="ollama" in models.yaml):
     → OllamaVLMAdapter (running local Mistral)
  2. If LLM_PROVIDER="gemini":
     → GeminiVLMAdapter (requires GEMINI_API_KEY)
  3. If LLM_PROVIDER="mock" or DEMO_MODE=true (and LLM_PROVIDER="auto"):
     → MockVLMAdapter
  4. If DEMO_MODE=false and GEMINI_API_KEY set:
     → GeminiVLMAdapter
  5. If local Ollama daemon is active:
     → OllamaVLMAdapter
  6. Otherwise:
     → raises ModelProviderUnavailableError (explaining Gemini and Ollama options)
"""

from typing import Optional

from app.core.config import settings, load_models_config
from app.models.base import VisionLanguageModel, VLMResponse
from app.models.gemini_adapter import GeminiVLMAdapter, ModelProviderUnavailableError
from app.models.mock_adapter import MockVLMAdapter
from app.models.ollama_adapter import OllamaVLMAdapter


def get_vlm_adapter(provider_override: Optional[str] = None) -> VisionLanguageModel:
    """
    Returns the configured Vision-Language / Language Model adapter.
    """
    provider = (provider_override or settings.LLM_PROVIDER or "auto").lower()

    # 1. Explicit Ollama or Auto: check if Ollama is actively reachable
    if provider in ("ollama", "auto"):
        probe_ollama = OllamaVLMAdapter(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        if probe_ollama.is_service_available():
            return probe_ollama
        if provider == "ollama":
            logger.warning(
                f"Ollama requested but unreachable at {settings.OLLAMA_BASE_URL}. "
                "Checking Gemini or fallback."
            )

    # 2. Explicit Gemini or Auto with API key
    if provider in ("gemini", "auto") and settings.GEMINI_API_KEY:
        return GeminiVLMAdapter()
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise ModelProviderUnavailableError(
            "GEMINI_API_KEY is not configured for Gemini provider. "
            "Set GEMINI_API_KEY in environment or start Ollama with LLM_PROVIDER=ollama.",
            status_code=503,
        )

    # 3. Explicit Mock or safe fallback when no live AI provider is running
    return MockVLMAdapter()


__all__ = [
    "VisionLanguageModel",
    "VLMResponse",
    "GeminiVLMAdapter",
    "MockVLMAdapter",
    "OllamaVLMAdapter",
    "ModelProviderUnavailableError",
    "get_vlm_adapter",
]
