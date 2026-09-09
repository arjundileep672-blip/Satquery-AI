"""
Google Gemini Vision-Language Model Adapter
Communicates with Google Gemini via REST API for multimodal remote sensing analysis.
"""

import base64
import os
from typing import Any, Dict, Optional
import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.models.base import VisionLanguageModel, VLMResponse

logger = get_logger("satquery.models.gemini")


class ModelProviderUnavailableError(Exception):
    """Raised when the configured model provider is unreachable or unconfigured."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.status_code = 503


class GeminiVLMAdapter(VisionLanguageModel):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self._model = model or settings.GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def name(self) -> str:
        return f"google/{self._model}"

    def generate_response(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        if not self._api_key:
            raise ModelProviderUnavailableError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in environment or enable DEMO_MODE=true."
            )

        # Contextualize prompt with verified remote sensing metadata
        meta = metadata or {}
        context_lines = ["You are SatQuery AI, an expert remote sensing visual assistant."]
        if meta:
            context_lines.append("Image Technical Metadata:")
            if meta.get("is_geotiff"):
                context_lines.append(f"- GeoTIFF CRS: {meta.get('crs')}")
                context_lines.append(f"- Bounds: {meta.get('bounds')}")
            context_lines.append(f"- Dimensions: {meta.get('width')}x{meta.get('height')} pixels")
            context_lines.append(f"- Band count: {meta.get('channels', meta.get('band_count', 3))}")

        full_system_prompt = "\n".join(context_lines)
        if system_instruction:
            full_system_prompt += f"\n{system_instruction}"

        # Base64 encode the image
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Determine web-safe mime type for Gemini
        gemini_mime = mime_type if mime_type in ("image/png", "image/jpeg", "image/webp") else "image/jpeg"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"
        
        payload = {
            "system_instruction": {
                "parts": [{"text": full_system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": gemini_mime,
                                "data": b64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            }
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                
            if response.status_code == 401 or response.status_code == 403:
                raise ModelProviderUnavailableError(f"Gemini API authentication failed (HTTP {response.status_code}).")
            elif response.status_code == 429:
                raise ModelProviderUnavailableError("Gemini rate limit exceeded. Please retry momentarily.")
            elif response.status_code >= 500:
                raise ModelProviderUnavailableError(f"Gemini service unavailable (HTTP {response.status_code}).")
            elif response.status_code != 200:
                logger.error(f"Gemini API error {response.status_code}: {response.text}")
                raise ModelProviderUnavailableError(f"Gemini API returned error code {response.status_code}.")

            resp_json = response.json()
            candidates = resp_json.get("candidates", [])
            if not candidates:
                raise ModelProviderUnavailableError("Gemini returned empty candidate response.")

            text_parts = candidates[0].get("content", {}).get("parts", [])
            response_text = "".join(p.get("text", "") for p in text_parts).strip()

            return VLMResponse(
                text=response_text or "No textual description could be generated from the model.",
                model_name=self.name,
                confidence=None,  # Gemini does not provide calibrated posterior confidence
                raw_metadata={"finish_reason": candidates[0].get("finish_reason")},
            )

        except httpx.RequestError as exc:
            logger.error(f"Network error calling Gemini API: {exc}")
            raise ModelProviderUnavailableError(f"Failed to reach Gemini service: {str(exc)}")
