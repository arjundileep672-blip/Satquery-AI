"""
SatQuery AI — Vision Response Schemas
Unified structured types for all vision model outputs.
Returned by every endpoint; frontend consumes this contract.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Patch-Based Multi-Label Scene Classification ──────────────────────────────

class PatchLabelSchema(BaseModel):
    """Single class label from patch tile-voting aggregation."""
    class_name: str
    coverage_pct: float    # % of tiles where this class was the top-1 winner
    tile_count: int        # number of winning tiles
    mean_confidence: float # mean softmax prob across winning tiles
    rank: int              # 1 = dominant class


class PatchSceneClassification(BaseModel):
    """
    Multi-label scene classification from patch-based tiling.
    Returned under statistics['patch_scene_classification'].
    """
    labels: List[PatchLabelSchema] = Field(default_factory=list)
    total_tiles: int = 0
    tile_size: int = 224
    stride: int = 112
    inference_ms: float = 0.0
    model: str = "eurosat_efficientnet_b0"
    available: bool = True


# ── Detection ─────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class OrientedBox(BaseModel):
    """Oriented bounding box returned by OBB models (e.g. DOTA)."""
    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float
    points: List[List[float]] = Field(default_factory=list)  # 4 corner [x,y] pairs


class Detection(BaseModel):
    id: str
    label: str
    confidence: float
    bbox: BoundingBox
    oriented_bbox: Optional[OrientedBox] = None
    area_px: Optional[float] = None
    area_m2: Optional[float] = None
    centroid: Optional[List[float]] = None  # [cx_px, cy_px]
    model: str = ""
    dataset: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    detections: List[Detection] = Field(default_factory=list)
    count: int = 0
    model: str = ""
    dataset: str = ""
    confidence_threshold: float = 0.25
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Segmentation ──────────────────────────────────────────────────────────────

class Mask(BaseModel):
    id: str
    label: str
    polygon: List[List[float]] = Field(default_factory=list)  # [[x,y], ...]
    bbox: Optional[BoundingBox] = None
    area_px: float = 0.0
    area_m2: Optional[float] = None
    confidence: Optional[float] = None
    model: str = ""


class SegmentationResult(BaseModel):
    masks: List[Mask] = Field(default_factory=list)
    count: int = 0
    total_area_px: float = 0.0
    model: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Change Detection ──────────────────────────────────────────────────────────

class ChangeRegion(BaseModel):
    region_id: str
    bbox: Optional[BoundingBox] = None
    changed_pixels: int = 0
    area_px: float = 0.0
    area_m2: Optional[float] = None
    change_type: str = "unclassified"
    model: str = ""


class ChangeDetectionResult(BaseModel):
    change_mask_shape: List[int] = Field(default_factory=list)  # [H, W]
    changed_pixels: int = 0
    total_pixels: int = 0
    change_percentage: float = 0.0
    change_regions: List[ChangeRegion] = Field(default_factory=list)
    registration_metrics: Dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    fallback_used: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Changed Objects ───────────────────────────────────────────────────────────

class ChangedObject(BaseModel):
    object_id: str
    label: str = "building"
    bbox: Optional[BoundingBox] = None
    mask_polygon: List[List[float]] = Field(default_factory=list)
    centroid: Optional[List[float]] = None
    change_overlap: float = 0.0       # IoU of object mask with change mask
    change_percentage: float = 0.0    # % of object pixels that are changed
    confidence: Optional[float] = None  # None unless from a calibrated model


class ChangedObjectsResult(BaseModel):
    changed_objects: List[ChangedObject] = Field(default_factory=list)
    total_objects_detected: int = 0
    total_changed: int = 0
    change_detection: Optional[ChangeDetectionResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Multitemporal Analysis Schemas ─────────────────────────────────────────────

class ParameterComparison(BaseModel):
    parameter: str
    category: str  # "built_up" | "roads" | "vegetation" | "water" | "land_cover"
    date_1: str
    date_2: str
    value_1: Optional[float] = None
    value_2: Optional[float] = None
    absolute_change: Optional[float] = None
    percentage_change: Optional[float] = None
    percentage_change_text: str = ""
    unit: str = ""
    status: str = "UNCHANGED"  # "INCREASED" | "DECREASED" | "UNCHANGED" | "NOT_AVAILABLE"
    confidence: Optional[float] = None
    notes: Optional[str] = None


class ObjectChangeItem(BaseModel):
    object_id: str
    object_class: str = "Building"
    status: str = "UNCHANGED"  # "NEW" | "REMOVED" | "EXPANDED" | "CONTRACTED" | "UNCHANGED" | "UNCERTAIN"
    date_first_detected: str = ""
    date_last_detected: str = ""
    area_date1: Optional[float] = None
    area_date2: Optional[float] = None
    area_change: Optional[float] = None
    percentage_change: Optional[float] = None
    percentage_change_text: Optional[str] = None
    centroid_pixel: Optional[List[float]] = None
    centroid_geo: Optional[Dict[str, float]] = None
    bbox_pixel: Optional[List[float]] = None
    polygon_pixel: List[List[float]] = Field(default_factory=list)
    perimeter_px: Optional[float] = None
    shape_compactness: Optional[float] = None
    region_sector: Optional[str] = None
    confidence: Optional[float] = None


class TransitionMatrixRow(BaseModel):
    previous_class: str
    current_class: str
    area_changed_m2: Optional[float] = None
    area_changed_hectares: Optional[float] = None
    area_changed_pixels: int = 0
    percentage_of_total_change: float = 0.0


class ChangeRankingItem(BaseModel):
    rank: int
    parameter: str
    category: str
    change_summary: str
    direction: str = "stable"  # "increase" | "decrease" | "stable"
    magnitude_type: str = "percentage"  # "percentage" | "area" | "count" | "length"
    raw_magnitude: float = 0.0


class TimeSeriesStep(BaseModel):
    date: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MultitemporalReport(BaseModel):
    executive_summary: str
    dates: List[str] = Field(default_factory=list)
    registration_quality: Dict[str, Any] = Field(default_factory=dict)
    parameters: List[ParameterComparison] = Field(default_factory=list)
    object_changes: List[ObjectChangeItem] = Field(default_factory=list)
    transition_matrix: List[TransitionMatrixRow] = Field(default_factory=list)
    change_ranking: List[ChangeRankingItem] = Field(default_factory=list)
    time_series: List[TimeSeriesStep] = Field(default_factory=list)
    limitations_and_disclaimers: List[str] = Field(default_factory=list)
    spatial_summary: Dict[str, Any] = Field(default_factory=dict)


# ── Unified Analysis Response (primary /analyze contract) ─────────────────────

class VisualizationRef(BaseModel):
    type: str        # "overlay" | "change_mask" | "detection_overlay"
    url: str         # relative URL served by FastAPI


class AnalysisResponse(BaseModel):
    request_id: str
    task: str        # QueryRouter task type
    answer: str      # NL answer grounded in model outputs
    detections: List[Detection] = Field(default_factory=list)
    masks: List[Mask] = Field(default_factory=list)
    changes: List[ChangeRegion] = Field(default_factory=list)
    changed_objects: List[ChangedObject] = Field(default_factory=list)
    multitemporal_report: Optional[MultitemporalReport] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    visualizations: List[VisualizationRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    # Phase-1 compatibility fields
    operation: Optional[str] = None
    confidence: Optional[float] = None
    evidence: List[Any] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    models_used: List[str] = Field(default_factory=list)
    analysis_trace: List[str] = Field(default_factory=list)

