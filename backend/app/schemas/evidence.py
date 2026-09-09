"""
Normalized Evidence Schema for SatQuery AI
Ensures auditability and transparent provenance across all tool outputs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str
    source_asset: str
    observation: str
    spatial_extent: Optional[List[float]] = None  # [min_x, min_y, max_x, max_y] or null
    confidence: Optional[float] = None            # Empirical/model score or null
    model_tool_used: str
    processing_steps: List[str] = Field(default_factory=list)
    measurements: Optional[Dict[str, Any]] = None # Quantitative metrics if derived
    geometry: Optional[Dict[str, Any]] = None     # GeoJSON geometry if localized
