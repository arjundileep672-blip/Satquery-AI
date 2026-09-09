"""
Abstract Vision-Language Model (VLM) Interface
Enforces provider independence so Gemini, GeoChat, or other foundation models
can be swapped seamlessly without changing the Planner or Tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class VLMResponse(BaseModel):
    text: str
    model_name: str
    confidence: Optional[float] = None  # None if uncalibrated/qualitative
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class VisionLanguageModel(ABC):
    """Abstract interface for all Vision-Language Model adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the underlying model / adapter."""
        pass

    @abstractmethod
    def generate_response(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        """
        Process the image and prompt through the vision-language model.
        Returns a standardized VLMResponse.
        """
        pass
