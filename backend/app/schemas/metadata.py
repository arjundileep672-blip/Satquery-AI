"""
Image Metadata Schemas for Standard and GeoTIFF Rasters
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    channels: int
    size_bytes: int
    is_geotiff: bool = False
    
    # GeoTIFF specific attributes (null for ordinary images)
    crs: Optional[str] = None
    transform: Optional[List[float]] = None  # Affine matrix [a, b, c, d, e, f]
    bounds: Optional[List[float]] = None     # [min_x, min_y, max_x, max_y]
    nodata: Optional[float] = None
    dtype: Optional[str] = None
    band_count: Optional[int] = None
    tags: Dict[str, Any] = Field(default_factory=dict)
