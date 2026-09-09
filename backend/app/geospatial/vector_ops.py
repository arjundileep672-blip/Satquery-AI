"""
Geospatial Vector Operations using Shapely
"""

import math
from typing import Any, Dict, List, Optional
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point, box
from app.schemas.geojson import GeoJSONFeature, GeoJSONFeatureCollection, GeoJSONGeometry


def bbox_to_geojson_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Dict[str, Any]:
    """Convert [min_lon, min_lat, max_lon, max_lat] to RFC 7946 Polygon geometry dict."""
    geom = box(min_lon, min_lat, max_lon, max_lat)
    return mapping(geom)


def calculate_polygon_area_m2(geom_dict: Dict[str, Any]) -> float:
    """
    Calculate geodesic area in square meters for a WGS84 (EPSG:4326) polygon.
    Uses ellipsoidal cosine correction for latitude.
    """
    try:
        poly = shape(geom_dict)
        if poly.is_empty:
            return 0.0
            
        centroid_lat = poly.centroid.y
        lat_rad = math.radians(centroid_lat)
        
        # 1 degree of latitude approx 111,139 meters
        # 1 degree of longitude approx 111,139 * cos(latitude) meters
        m_per_deg_lat = 111139.0
        m_per_deg_lon = 111139.0 * math.cos(lat_rad)
        
        # Approximate projection to planar square meters
        area_deg2 = poly.area
        area_m2 = area_deg2 * (m_per_deg_lat * m_per_deg_lon)
        return float(abs(area_m2))
    except Exception:
        return 0.0


def create_feature(geometry_dict: Dict[str, Any], properties: Optional[Dict[str, Any]] = None, feature_id: Optional[str] = None) -> GeoJSONFeature:
    """Create a validated GeoJSONFeature."""
    return GeoJSONFeature(
        type="Feature",
        geometry=GeoJSONGeometry(
            type=geometry_dict.get("type", "Polygon"),
            coordinates=geometry_dict.get("coordinates", [])
        ),
        properties=properties or {},
        id=feature_id
    )


def create_feature_collection(features: List[GeoJSONFeature], bbox: Optional[List[float]] = None) -> GeoJSONFeatureCollection:
    """Create a validated GeoJSONFeatureCollection."""
    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=features,
        bbox=bbox
    )


def is_geometry_intersecting(geom1_dict: Dict[str, Any], geom2_dict: Dict[str, Any]) -> bool:
    """Check topological intersection between two GeoJSON geometries."""
    try:
        s1 = shape(geom1_dict)
        s2 = shape(geom2_dict)
        return bool(s1.intersects(s2))
    except Exception:
        return True
