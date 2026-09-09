"""
SatQuery AI - Spatial Analysis Pipeline
Extracts geospatial metrics from image metadata and detected regions.
"""
from typing import Any, Dict, List, Optional
import numpy as np

from app.core.logging import get_logger

logger = get_logger("satquery.pipelines.spatial_analysis")


def run_spatial_analysis(
    image: np.ndarray,
    meta: Dict[str, Any],
    polygons: Optional[List[List[List[float]]]] = None,
) -> Dict[str, Any]:
    """
    Compute spatial metrics for an image and optional polygon list.

    Args:
        image:    Decoded image array.
        meta:     Metadata dict from extract_metadata().
        polygons: Optional list of [[x,y],...] polygons for area/centroid calc.

    Returns:
        Spatial analysis result dict.
    """
    h, w = image.shape[:2]
    result: Dict[str, Any] = {
        "image_width_px": w,
        "image_height_px": h,
        "total_pixels": h * w,
    }

    is_geo = bool(meta.get("is_geotiff") and meta.get("crs"))
    result["georeferencing_available"] = is_geo

    if not is_geo:
        result["area_available"] = False
        result["note"] = "No valid CRS/geotransform — metrics are pixel-relative."
        return result

    result["crs"] = str(meta.get("crs"))
    result["bounds"] = meta.get("bounds")
    result["area_available"] = True

    if polygons:
        try:
            from app.geospatial.raster_ops import pixel_area_to_m2
            transform = meta.get("transform")
            polygon_areas = []
            for poly in polygons:
                import numpy as _np
                pts = _np.array(poly)
                # Shoelace for pixel area
                x = pts[:, 0]
                y = pts[:, 1]
                area_px = 0.5 * abs(_np.dot(x, _np.roll(y, 1)) - _np.dot(y, _np.roll(x, 1)))
                if transform:
                    area_m2 = pixel_area_to_m2(area_px, transform, meta.get("crs"))
                    polygon_areas.append({"area_px": area_px, "area_m2": round(area_m2, 1)})
                else:
                    polygon_areas.append({"area_px": area_px})
            result["polygon_areas"] = polygon_areas
        except Exception as exc:
            logger.warning(f"Polygon area calculation failed: {exc}")
            result["polygon_area_error"] = str(exc)

    return result
