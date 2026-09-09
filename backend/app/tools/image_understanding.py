"""
Image Understanding Tool for SatQuery AI
Interprets scene-level context, dominant terrain, and high-level remote-sensing structures.
"""

from typing import Any, Dict, List, Optional
from app.models.base import VisionLanguageModel
from app.models import get_vlm_adapter
from app.tools.base import BaseTool, ToolExecutionOutput


class ImageUnderstandingTool(BaseTool):
    def __init__(self, vlm: Optional[VisionLanguageModel] = None):
        self._vlm = vlm

    @property
    def vlm(self) -> VisionLanguageModel:
        if self._vlm is None:
            return get_vlm_adapter()
        return self._vlm

    @property
    def name(self) -> str:
        return "image_understanding"

    @property
    def description(self) -> str:
        return "Provides scene-level remote sensing classification, land cover composition, and structural context."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_bytes": {"type": "string", "description": "Raw image bytes"},
                "mime_type": {"type": "string", "description": "MIME type of the image"},
                "query": {"type": "string", "description": "User intent / prompt"},
                "metadata": {"type": "object", "description": "Extracted raster metadata"},
            },
            "required": ["image_bytes", "mime_type"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "observations": {"type": "array", "items": {"type": "string"}},
                "measurements": {"type": "object"},
                "confidence": {"type": ["number", "null"]},
            },
        }

    @property
    def capabilities(self) -> List[str]:
        return [
            "scene_classification",
            "land_cover_summary",
            "terrain_context",
            "multispectral_context_integration",
        ]

    @property
    def supported_data_types(self) -> List[str]:
        return ["PNG", "JPEG", "TIFF", "GeoTIFF"]

    @property
    def limitations(self) -> List[str]:
        return [
            "Cannot provide sub-pixel segmentation masks in Phase 1.",
            "Confidence is qualitative/uncalibrated from underlying VLM.",
            "Visual features are subject to resolution limits and atmospheric conditions.",
        ]

    def execute(self, **kwargs) -> ToolExecutionOutput:
        image_bytes: bytes = kwargs.get("image_bytes", b"")
        mime_type: str = kwargs.get("mime_type", "image/jpeg")
        query: str = kwargs.get("query", "Describe the primary features and terrain visible in this satellite scene.")
        metadata: Optional[Dict[str, Any]] = kwargs.get("metadata")

        system_prompt = (
            "You are analyzing a remote-sensing image.\n"
            "Answer the user's question using only information visually supported by the provided image.\n"
            "Do not invent objects, coordinates, measurements, sensor properties, or geographic locations.\n"
            "If the image does not contain enough visual evidence to answer the question reliably, "
            "explicitly state that limitation.\n"
            "Distinguish visual interpretation from quantitative remote-sensing measurements.\n"
            "User question:\n"
            f"{query}"
        )

        vlm_resp = self.vlm.generate_response(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=query,
            metadata=metadata,
            system_instruction=system_prompt,
        )

        observations = [
            "Scene-level land cover and functional zones evaluated.",
            "Dominant optical reflectance profiles and spatial patterns identified.",
        ]

        return ToolExecutionOutput(
            tool_name=self.name,
            status="success",
            answer=vlm_resp.text,
            confidence=vlm_resp.confidence,
            observations=observations,
            measurements={},
            model_name=vlm_resp.model_name,
        )
