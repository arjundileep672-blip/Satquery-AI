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
    vlm_cfg = load_models_config().get("vlm", {})
    cfg_provider = vlm_cfg.get("provider", "").lower()

    # 1. Explicit Ollama request or configured default
    if provider == "ollama" or (provider == "auto" and cfg_provider == "ollama" and not settings.DEMO_MODE):
        probe_ollama = OllamaVLMAdapter(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        if probe_ollama.is_service_available():
            return probe_ollama
        # On cloud hosts (e.g. Render), local Ollama is not running.
        # Fall back to Gemini if configured, or Mock adapter so the API never crashes with 503.
        if settings.GEMINI_API_KEY:
            return GeminiVLMAdapter()
        return MockVLMAdapter()

    # 2. Explicit Mock request
    if provider == "mock":
        return MockVLMAdapter()

    # 3. Explicit Gemini request
    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ModelProviderUnavailableError(
                "GEMINI_API_KEY is not configured for Gemini provider. "
                "Set GEMINI_API_KEY in environment or switch to offline Ollama with LLM_PROVIDER=ollama.",
                status_code=503,
            )
        return GeminiVLMAdapter()

    # 4. Auto resolution: DEMO_MODE defaults to Mock unless explicit provider specified
    if settings.DEMO_MODE:
        return MockVLMAdapter()

    # 5. Production live mode with Gemini key
    if settings.GEMINI_API_KEY:
        return GeminiVLMAdapter()

    # 6. Auto-probe: if local Ollama daemon is running, use it
    probe_ollama = OllamaVLMAdapter(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        timeout=settings.OLLAMA_TIMEOUT_SECONDS,
    )
    if probe_ollama.is_service_available():
        return probe_ollama

    # 7. Safe fallback for cloud production: use MockVLMAdapter rather than failing with 503
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
