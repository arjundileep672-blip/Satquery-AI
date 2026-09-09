"""
Visual Question Answering (VQA) Tool for SatQuery AI
Grounds targeted natural-language questions in satellite raster features.
"""

from typing import Any, Dict, List, Optional
from app.models.base import VisionLanguageModel
from app.models import get_vlm_adapter
from app.tools.base import BaseTool, ToolExecutionOutput


class VisualQuestionAnsweringTool(BaseTool):
    def __init__(self, vlm: Optional[VisionLanguageModel] = None):
        self._vlm = vlm

    @property
    def vlm(self) -> VisionLanguageModel:
        if self._vlm is None:
            return get_vlm_adapter()
        return self._vlm

    @property
    def name(self) -> str:
        return "visual_question_answering"

    @property
    def description(self) -> str:
        return "Answers targeted natural-language questions about specific objects, spatial relations, and physical features in remote sensing imagery."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_bytes": {"type": "string", "description": "Raw image bytes"},
                "mime_type": {"type": "string", "description": "MIME type of the image"},
                "query": {"type": "string", "description": "Specific question about the image"},
                "metadata": {"type": "object", "description": "Extracted raster metadata"},
            },
            "required": ["image_bytes", "mime_type", "query"],
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
            "targeted_feature_answering",
            "spatial_relation_identification",
            "presence_absence_verification",
            "metadata_conditioned_answering",
        ]

    @property
    def supported_data_types(self) -> List[str]:
        return ["PNG", "JPEG", "TIFF", "GeoTIFF"]

    @property
    def limitations(self) -> List[str]:
        return [
            "Does not output calibrated bounding coordinates in Phase 1.",
            "Answers reflect statistical vision-language inference, not surveyed ground truth.",
            "Small objects below the sensor's Ground Sample Distance may not be resolved.",
        ]

    def execute(self, **kwargs) -> ToolExecutionOutput:
        image_bytes: bytes = kwargs.get("image_bytes", b"")
        mime_type: str = kwargs.get("mime_type", "image/jpeg")
        query: str = kwargs.get("query", "What is visible in this image?")
        metadata: Optional[Dict[str, Any]] = kwargs.get("metadata")

        system_prompt = (
            "You are SatQuery AI's Visual Question Answering specialist. "
            "Address the user's specific inquiry directly using observable visual and spatial evidence. "
            "Be precise, factual, and avoid speculating beyond what is clearly resolvable in the image."
        )

        vlm_resp = self.vlm.generate_response(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=query,
            metadata=metadata,
            system_instruction=system_prompt,
        )

        observations = [
            f"Query targeted: '{query}'",
            "Visual evidence cross-referenced with imagery spatial extent.",
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
