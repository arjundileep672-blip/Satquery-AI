"""
SatQuery AI - Detection Pipeline
Thin wrapper over yolo_detector and remote_detector for use in orchestrator.
"""
from typing import List, Optional
import numpy as np
from app.core.logging import get_logger

logger = get_logger("satquery.pipelines.detection")

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


def run_generic_detection(image: np.ndarray, classes: Optional[List[str]] = None, conf: float = 0.25):
    """YOLO12n generic detection."""
    from models.yolo_detector import detect
    return detect(image, classes=classes, conf=conf)


def run_remote_detection(image: np.ndarray, query: str = "", conf: float = 0.25):
    """YOLO26n-OBB DOTA detection. Filters classes mentioned in the query."""
    from models.remote_detector import detect_remote
    q = query.lower()
    filter_classes = [c for c in DOTA_CLASSES if c in q] or None
    return detect_remote(image, classes=filter_classes, conf=conf)
