"""
SatQuery AI — Polygon Utilities & Spatial Relationships
"""

from typing import Any, Dict, List, Optional
import numpy as np


def polygon_centroid(points: List[List[float]]) -> List[float]:
    """Centroid [cx, cy] of polygon points."""
    if not points:
        return [0.0, 0.0]
    pts = np.array(points, dtype=np.float64)
    return [round(float(pts[:, 0].mean()), 2), round(float(pts[:, 1].mean()), 2)]


def polygon_to_bbox(points: List[List[float]]) -> List[float]:
    """[x1, y1, x2, y2] from polygon coordinates."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    pts = np.array(points, dtype=np.float64)
    return [
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    ]
