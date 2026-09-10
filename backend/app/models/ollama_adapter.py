"""
Ollama Vision-Language / Language Model Adapter
Communicates with a local Ollama instance (e.g., Mistral, LLaVA) via REST API
for 100% offline, local multimodal and language reasoning.
"""

import base64
import os
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings, load_models_config
from app.core.logging import get_logger
from app.models.base import VisionLanguageModel, VLMResponse
from app.models.gemini_adapter import ModelProviderUnavailableError

logger = get_logger("satquery.models.ollama")


class OllamaVLMAdapter(VisionLanguageModel):
    """
    Adapter for local Ollama instances running offline models like Mistral.
    Communicates via Ollama's native HTTP REST API (/api/generate).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        vlm_cfg = load_models_config().get("vlm", {})
        self._base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
            or vlm_cfg.get("ollama_base_url", "http://localhost:11434")
        ).rstrip("/")

        self._model = (
            model
            or os.getenv("OLLAMA_MODEL")
            or getattr(settings, "OLLAMA_MODEL", "mistral")
            or vlm_cfg.get("model", "mistral")
        )

        self._timeout = timeout or getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 60.0)

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model_name(self) -> str:
        return self._model

    def is_service_available(self) -> bool:
        """Quick health-check probe to see if Ollama daemon is reachable."""
        try:
            with httpx.Client(timeout=0.5) as client:
                res = client.get(f"{self._base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_local_models(self) -> List[str]:
        """Query Ollama for available local models."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self._base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            logger.warning(f"Could not retrieve Ollama model list: {exc}")
        return []

    def generate_response(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        """
        Process user query and remote sensing imagery/context through local Ollama instance.
        """
        # Contextualize prompt with verified remote sensing technical metadata
        meta = metadata or {}
        context_lines = [
            "You are SatQuery AI, an expert offline remote sensing analysis assistant."
        ]
        if meta:
            context_lines.append("Image Technical Metadata:")
            if meta.get("is_geotiff"):
                context_lines.append(f"- GeoTIFF CRS: {meta.get('crs')}")
                context_lines.append(f"- Bounds: {meta.get('bounds')}")
            if meta.get("width") and meta.get("height"):
                context_lines.append(f"- Dimensions: {meta.get('width')}x{meta.get('height')} pixels")
            if meta.get("channels") or meta.get("band_count"):
                context_lines.append(
                    f"- Band count: {meta.get('channels', meta.get('band_count', 3))}"
                )

        if system_instruction:
            context_lines.append(system_instruction)

        full_system_prompt = "\n".join(context_lines)

        # Check if the model supports multimodal vision (e.g. llava, bakllava, moondream)
        is_vision_model = any(k in self._model.lower() for k in ("llava", "vision", "moondream", "minicpm"))

        # Moondream does not support system prompt blocks (causes empty generation)
        supports_system = not any(k in self._model.lower() for k in ("moondream",))

        # Prepare request payload for Ollama /api/generate
        url = f"{self._base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        }
        if supports_system and full_system_prompt:
            payload["system"] = full_system_prompt

        if is_vision_model and image_bytes:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            payload["images"] = [b64_img]

        try:
            client_timeout = httpx.Timeout(self._timeout, connect=2.0)
            with httpx.Client(timeout=client_timeout) as client:
                response = client.post(url, json=payload)

            if response.status_code == 404:
                raise ModelProviderUnavailableError(
                    f"Ollama model '{self._model}' not found on local instance at {self._base_url}. "
                    f"Run: `ollama pull {self._model}` in your terminal."
                )
            elif response.status_code != 200:
                logger.error(f"Ollama API returned HTTP {response.status_code}: {response.text}")
                raise ModelProviderUnavailableError(
                    f"Ollama returned error code {response.status_code}: {response.text}"
                )

            data = response.json()
            response_text = data.get("response", "").strip()

            if not response_text:
                logger.warning(
                    f"Ollama model '{self._model}' returned empty text. "
                    "Falling back to structured results summary."
                )
                raise ModelProviderUnavailableError(
                    f"Ollama returned an empty response from model '{self._model}'."
                )

            return VLMResponse(
                text=response_text,
                model_name=self.name,
                confidence=None,  # Local LLM outputs are qualitative
                raw_metadata={
                    "total_duration": data.get("total_duration"),
                    "eval_count": data.get("eval_count"),
                    "eval_duration": data.get("eval_duration"),
                    "provider": "ollama",
                    "model": self._model,
                },
            )

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError) as exc:
            logger.warning(f"Ollama request issue ({exc}); returning structured fallback response.")
            return VLMResponse(
                text="",
                model_name=self.name,
                confidence=0.85,
                raw_metadata={"error": str(exc), "fallback": True},
            )
