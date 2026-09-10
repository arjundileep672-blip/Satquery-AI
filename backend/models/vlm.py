"""
SatQuery AI — VLM Adapter
Thin bridge between the orchestrator and the existing Phase-1 VLM adapters.
Adds structured interpret/synthesize methods used by the orchestrator.
"""

from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models import get_vlm_adapter
from app.models.base import VisionLanguageModel, VLMResponse

logger = get_logger("satquery.models.vlm")

# System prompt for the final-answer synthesis step
_SYNTHESIS_PROMPT = """\
You are SatQuery AI — an expert remote sensing analysis assistant.
You will receive structured vision model results and must generate a concise,
accurate natural-language answer to the user's query.

Rules:
- Base your answer ONLY on the provided model outputs. Never invent detections.
- If a model was unavailable or used a fallback, clearly say so.
- Do not claim higher confidence than the models provide.
- Be specific: mention counts, percentages, and regions when available.
- Do not speculate about geographic location unless georeferencing data is provided.
"""

_QUERY_INTERPRET_PROMPT = """\
You are a remote sensing query analyst.
Given a user query about satellite/aerial imagery, extract:
1. The primary intent (detection, counting, change analysis, segmentation, etc.)
2. The specific objects or classes of interest
3. Any temporal context (before/after, T1/T2)

Respond concisely in 2-3 sentences.
"""


class VLMBridge:
    """
    Wraps the configured VLM adapter with domain-specific methods.
    Delegates to MockVLMAdapter (DEMO_MODE) or GeminiVLMAdapter (production).
    """

    def __init__(self):
        self._vlm: Optional[VisionLanguageModel] = None

    @property
    def vlm(self) -> VisionLanguageModel:
        if self._vlm is None:
            self._vlm = get_vlm_adapter()
        return self._vlm

    @property
    def model_name(self) -> str:
        return self.vlm.name

    def interpret_query(
        self,
        query: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/png",
    ) -> str:
        """
        Optional: use VLM to clarify query intent.
        Returns the interpretation text.
        If image_bytes is None, send a minimal placeholder.
        """
        placeholder = image_bytes or b""
        if not placeholder:
            return query  # No VLM call needed when image not provided

        resp = self.vlm.generate_response(
            image_bytes=placeholder,
            mime_type=mime_type,
            prompt=f"User query: {query}\n\nBriefly interpret what analysis is needed.",
            metadata={},
            system_instruction=_QUERY_INTERPRET_PROMPT,
        )
        return resp.text

    def synthesize_answer(
        self,
        query: str,
        task: str,
        results_summary: str,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/png",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a grounded natural-language answer from structured vision results.

        Args:
            query:           Original user query.
            task:            TaskType string (from query router).
            results_summary: Structured text summarising actual model outputs.
            image_bytes:     Optional image for visual context.
            mime_type:       MIME type of image.
            metadata:        Image metadata dict.

        Returns:
            Natural-language answer string.
        """
        effective_bytes = image_bytes or b""
        if not effective_bytes:
            # No image → return a structured text answer without VLM
            return f"Analysis complete. {results_summary}"

        if hasattr(self.vlm, "is_service_available") and not self.vlm.is_service_available():
            return results_summary

        try:
            if "moondream" in self.vlm.name.lower():
                # moondream requires direct, natural questions without meta system prompts
                clean_query = query.strip() if query else "Describe what is visible in this satellite image."
                resp = self.vlm.generate_response(
                    image_bytes=effective_bytes,
                    mime_type=mime_type,
                    prompt=clean_query,
                    metadata=metadata or {},
                    system_instruction=None,
                )
                vlm_text = (resp.text or "").strip()
                if results_summary and vlm_text:
                    return f"{results_summary}\n\n{vlm_text}"
                return vlm_text or results_summary

            prompt = (
                f"User query: \"{query}\"\n"
                f"Task type: {task}\n\n"
                f"Vision model results:\n{results_summary}\n\n"
                "Generate a concise, accurate answer to the user's query based solely on the above results."
            )

            resp = self.vlm.generate_response(
                image_bytes=effective_bytes,
                mime_type=mime_type,
                prompt=prompt,
                metadata=metadata or {},
                system_instruction=_SYNTHESIS_PROMPT,
            )
            return resp.text or results_summary
        except Exception as exc:
            logger.warning(f"VLM synthesis timed out or failed ({exc}); falling back to structured vision summary.")
            return results_summary


# Module-level singleton
_bridge: Optional[VLMBridge] = None


def get_vlm_bridge() -> VLMBridge:
    global _bridge
    if _bridge is None:
        _bridge = VLMBridge()
    return _bridge
