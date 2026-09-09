"""
Imagery Asset & Geospatial Metadata Schemas
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModalityEnum(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    FUSED = "fused"


class BandInfo(BaseModel):
    name: str              # e.g., 'Red', 'Green', 'Blue', 'NIR', 'VV', 'VH'
    index: int             # 1-indexed band number in raster
    wavelength_nm: Optional[float] = None
    description: Optional[str] = None


class GeoBounds(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class ImageryAsset(BaseModel):
    id: str
    title: str
    sensor: str            # 'Sentinel-2A', 'Sentinel-1B', 'PlanetScope', etc.
    modality: ModalityEnum
    acquisition_date: str  # ISO 8601 string
    spatial_resolution_m: float
    crs: str = "EPSG:4326"
    bounds: GeoBounds
    bands: List[BandInfo] = Field(default_factory=list)
    cloud_cover_percentage: Optional[float] = None  # None for SAR
    raster_path: str
    preview_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
