"""
SatQuery AI — Geospatial Area Calculations
Calculates polygon and raster area in m² and km² when CRS is available.
Falls back to pixel area when georeferencing is unavailable.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def calculate_polygon_area(
    points: List[List[float]],
    transform: Optional[Any] = None,
    crs_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate polygon area in pixels and square meters (if georeferenced)."""
    if len(points) < 3:
        return {"area_pixels": 0.0, "georeferencing_available": False, "area_available": False}

    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace formula for pixel area
    area_px = float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    result = {
        "area_pixels": round(area_px, 2),
        "georeferencing_available": False,
        "area_available": False,
    }

    if not transform or not crs_str:
        return result

    try:
        import rasterio
        from shapely.geometry import Polygon
        import pyproj
        from pyproj import Transformer
        from shapely.ops import transform as shapely_transform

        if isinstance(transform, list):
            affine = rasterio.Affine(*transform[:6])
        else:
            affine = transform

        # Transform pixel vertices to projected CRS coordinates
        proj_pts = [rasterio.transform.xy(affine, py, px, offset="center") for px, py in points]
        poly_proj = Polygon(proj_pts)

        # Check if CRS is projected (units in meters)
        crs = pyproj.CRS.from_user_input(crs_str)
        if crs.is_projected:
            area_m2 = float(poly_proj.area)
        else:
            # Geographic CRS (degrees) -> reproject to equal-area projection
            # Use auto UTM or Web Mercator approximation
            bounds = poly_proj.bounds
            center_lon = (bounds[0] + bounds[2]) / 2.0
            utm_zone = int((center_lon + 180) / 6) + 1
            utm_crs = f"EPSG:326{utm_zone:02d}" if bounds[1] >= 0 else f"EPSG:327{utm_zone:02d}"
            transformer = Transformer.from_crs(crs_str, utm_crs, always_xy=True)
            poly_utm = shapely_transform(transformer.transform, poly_proj)
            area_m2 = float(poly_utm.area)

        result["georeferencing_available"] = True
        result["area_available"] = True
        result["area_m2"] = round(area_m2, 2)
        result["area_km2"] = round(area_m2 / 1_000_000.0, 6)
    except Exception:
        pass

    return result
