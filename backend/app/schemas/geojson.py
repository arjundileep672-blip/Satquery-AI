"""
RFC 7946 Compliant GeoJSON Schemas for SatQuery AI
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class GeoJSONGeometry(BaseModel):
    type: Literal["Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"]
    coordinates: Any  # Supports nested coordinate arrays per RFC 7946


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[Union[str, int]] = None


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(default_factory=list)
    bbox: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
