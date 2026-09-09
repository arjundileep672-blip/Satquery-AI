"""
Agent Orchestration, Tool Contracts, and Structured Response Schemas
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from app.schemas.geojson import GeoJSONFeatureCollection


class ToolInvocation(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str


class ToolExecutionRecord(BaseModel):
    step_number: int
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "failed", "skipped"]
    execution_time_ms: float
    summary: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class DetectedObjectItem(BaseModel):
    id: str
    label: str
    confidence: float
    bbox_geo: Optional[List[float]] = None      # [min_lon, min_lat, max_lon, max_lat]
    bbox_pixel: Optional[List[int]] = None     # [x_min, y_min, x_max, y_max]
    oriented_bbox: Optional[Dict[str, Any]] = None  # {cx, cy, w, h, angle_deg}
    area_m2: Optional[float] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    source_tool: str
    modality: str                              # "optical", "sar", "cross_modal", "geospatial"
    metric_name: str
    metric_value: Any
    confidence: float
    description: str


class ChangeRegionItem(BaseModel):
    region_id: str
    change_type: str                           # "urban_expansion", "vegetation_loss", "water_inundation"
    area_m2: float
    confidence: float
    geometry_geojson: Optional[Dict[str, Any]] = None


class StructuredAnalysisResult(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    detected_objects: List[DetectedObjectItem] = Field(default_factory=list)
    bounding_boxes: Optional[GeoJSONFeatureCollection] = None
    segmentation_masks: Optional[GeoJSONFeatureCollection] = None
    change_regions: List[ChangeRegionItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_steps: List[ToolExecutionRecord] = Field(default_factory=list)
    models_tools_used: List[str] = Field(default_factory=list)
