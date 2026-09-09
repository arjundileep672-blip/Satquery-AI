"""
SatQuery AI - Segmentation Pipeline
Detects objects with YOLO then segments with SAM2.
"""
import numpy as np
from app.core.logging import get_logger
from app.schemas.vision import SegmentationResult

logger = get_logger("satquery.pipelines.segmentation")


def run_segmentation(image: np.ndarray, conf: float = 0.20) -> SegmentationResult:
    """Generic segmentation: YOLO detection -> SAM2 masks."""
    from models.yolo_detector import detect
    from models.sam2_segmenter import segment_with_boxes

    det = detect(image, conf=conf)
    if not det.detections:
        logger.info("No objects detected for segmentation.")
        return SegmentationResult(masks=[], count=0, model="none")

    boxes = [d.bbox for d in det.detections]
    labels = [d.label for d in det.detections]
    return segment_with_boxes(image, boxes, labels)


def run_building_segmentation(image: np.ndarray) -> dict:
    """Building-specific segmentation via SpaceNet + SAM2."""
    from models.building_detector import detect_buildings
    return detect_buildings(image)
