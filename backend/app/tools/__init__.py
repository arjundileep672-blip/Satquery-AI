"""
Controlled Tool Registry Initialization for Phase 1
Registers Phase 1 tools: image_understanding and visual_question_answering.
"""

from app.tools.base import BaseTool, ToolExecutionOutput
from app.tools.registry import ToolRegistry, registry
from app.tools.image_understanding import ImageUnderstandingTool
from app.tools.visual_question_answering import VisualQuestionAnsweringTool
from app.tools.object_detection import ObjectDetectionTool
from app.tools.segmentation import SegmentationTool
from app.tools.change_detection import ChangeDetectionTool

# Register tools
registry.register(ImageUnderstandingTool())
registry.register(VisualQuestionAnsweringTool())
registry.register(ObjectDetectionTool())
registry.register(SegmentationTool())
registry.register(ChangeDetectionTool())

__all__ = [
    "BaseTool",
    "ToolExecutionOutput",
    "ToolRegistry",
    "registry",
    "ImageUnderstandingTool",
    "VisualQuestionAnsweringTool",
    "ObjectDetectionTool",
    "SegmentationTool",
    "ChangeDetectionTool",
]
