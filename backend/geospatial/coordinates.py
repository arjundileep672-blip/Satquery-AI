"""
SatQuery AI — Geospatial Coordinate Operations
Converts between pixel coordinates, projected CRS coordinates, and WGS84 lat/long.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import pyproj
    import rasterio
    import rasterio.transform
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


def extract_georeferencing(filepath_or_bytes: Any) -> Dict[str, Any]:
    """Extract CRS, affine transform, dimensions, and bounds from a raster."""
    if not _GEO_AVAILABLE:
        return {"georeferencing_available": False, "area_available": False}

    try:
        if isinstance(filepath_or_bytes, (str, bytes)):
            import io
            source = io.BytesIO(filepath_or_bytes) if isinstance(filepath_or_bytes, bytes) else filepath_or_bytes
            with rasterio.open(source) as src:
                crs = src.crs
                transform = src.transform
                bounds = src.bounds
                res = src.res
                width, height = src.width, src.height

                if crs is None or transform.is_identity:
                    return {
                        "georeferencing_available": False,
                        "area_available": False,
                        "width": width,
                        "height": height,
                    }

                return {
                    "georeferencing_available": True,
                    "area_available": True,
                    "crs": crs.to_string(),
                    "transform": list(transform),
                    "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    "resolution": list(res),
                    "width": width,
                    "height": height,
                }
    except Exception:
        pass

    return {"georeferencing_available": False, "area_available": False}


def pixel_to_latlon(
    x_px: float,
    y_px: float,
    transform: Any,
    crs_str: Optional[str],
) -> Optional[Tuple[float, float]]:
    """Convert (pixel_x, pixel_y) -> (latitude, longitude) in EPSG:4326."""
    if not _GEO_AVAILABLE or not crs_str or not transform:
        return None

    try:
        import rasterio.transform
        from pyproj import Transformer

        if isinstance(transform, list):
            transform = rasterio.Affine(*transform[:6])

        # Pixel to projected CRS coordinates
        proj_x, proj_y = rasterio.transform.xy(transform, y_px, x_px, offset="center")

        # Projected CRS -> WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(proj_x, proj_y)
        return round(float(lat), 6), round(float(lon), 6)
    except Exception:
        return None
