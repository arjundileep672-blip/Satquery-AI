"""
Analysis Result Schema for SatQuery AI
Defines the complete structured response delivered to the client.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.evidence import Evidence


class AnalysisResult(BaseModel):
    request_id: str
    answer: str
    operation: str  # "image_understanding" | "visual_question_answering" | "object_detection" | "segmentation" | "change_detection"
    task: Optional[str] = None
    confidence: Optional[float] = None  # None if uncalibrated/qualitative
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    masks: List[Dict[str, Any]] = Field(default_factory=list)
    changes: List[Dict[str, Any]] = Field(default_factory=list)
    multitemporal_report: Optional[Dict[str, Any]] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    models_used: List[str] = Field(default_factory=list)
    analysis_trace: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

