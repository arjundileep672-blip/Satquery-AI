"""
SatQuery AI — Remote Sensing Oriented Bounding Box Detector
Wraps YOLO26n-OBB / YOLOv8n-OBB trained on DOTA-v1 for aerial/satellite imagery.

PROVENANCE (accurate — do not modify):
  Model:   YOLO26n-OBB (fallback: YOLOv8n-OBB)
  Dataset: DOTA-v1 (Dataset for Object deTection in Aerial images, v1.0)
  Task:    Oriented object detection
  Classes: plane, ship, storage-tank, baseball-diamond, tennis-court,
           basketball-court, ground-track-field, harbor, bridge,
           large-vehicle, small-vehicle, helicopter, roundabout,
           soccer-ball-field, swimming-pool

IMPORTANT:
  DOTA-v1 does NOT contain generic building footprints.
  Use building_detector.py for building detection.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings, load_models_config, resolve_device
from app.core.logging import get_logger
from app.schemas.vision import BoundingBox, Detection, DetectionResult, OrientedBox

logger = get_logger("satquery.models.remote_detector")

# Official DOTA-v1 class names
DOTA_V1_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

_model_instance = None  # Lazy singleton


def _load_model(checkpoint: Optional[str] = None, device: Optional[str] = None):
    """Lazy-load YOLO OBB model."""
    global _model_instance
    if _model_instance is not None and checkpoint is None:
        return _model_instance

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is not installed.") from exc

    cfg = load_models_config().get("remote_detector", {})
    ckpt_rel = checkpoint or cfg.get("checkpoint", "models/yolo26_obb/yolo26n-obb.pt")
    fb_rel = cfg.get("fallback_checkpoint", "checkpoints/yolov8n-obb.pt")

    ckpt_path = settings.BASE_DIR.parent / ckpt_rel
    fb_path = settings.BASE_DIR.parent / fb_rel
    root_fallback = settings.BASE_DIR.parent / "yolov8n-obb.pt"

    if ckpt_path.exists():
        ckpt = str(ckpt_path)
    elif fb_path.exists():
        ckpt = str(fb_path)
    elif root_fallback.exists():
        ckpt = str(root_fallback)
    else:
        ckpt = "yolov8n-obb.pt"

    logger.info(f"Loading Remote OBB detector model from: {ckpt}")
    model = YOLO(ckpt)

    dev = device or cfg.get("device", resolve_device())
    if dev != "auto":
        model.to(dev)

    if checkpoint is None:
        _model_instance = model
    return model


class RemoteDetector:
    """
    Standard interface for YOLO26n-OBB / DOTA remote sensing detector.
    """

    def __init__(self, checkpoint: Optional[str] = None, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    def load(self):
        if self.model is None:
            self.model = _load_model(self.checkpoint, self.device)
        return self

    def predict(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        conf: float = 0.25,
    ) -> DetectionResult:
        self.load()
        return detect_remote(image, classes=classes, conf=conf)


def _obb_to_corners(cx: float, cy: float, w: float, h: float, angle_rad: float) -> List[List[float]]:
    """Compute 4 corner [x, y] points of an oriented box."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    half_w, half_h = w / 2.0, h / 2.0
    offsets = [
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
    ]
    corners = []
    for dx, dy in offsets:
        rx = cos_a * dx - sin_a * dy + cx
        ry = sin_a * dx + cos_a * dy + cy
        corners.append([round(rx, 2), round(ry, 2)])
    return corners


def detect_remote(
    image: np.ndarray,
    classes: Optional[List[str]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1024,
) -> DetectionResult:
    """
    Run YOLO26n-OBB detection for aerial/remote-sensing imagery.
    """
    t0 = time.perf_counter()

    try:
        model = _load_model()
    except (ImportError, Exception) as exc:
        logger.warning(f"Remote detector unavailable: {exc}")
        return DetectionResult(
            detections=[],
            count=0,
            model="yolo26n-obb",
            dataset="DOTA-v1",
            confidence_threshold=conf,
            metadata={"error": str(exc), "status": "unavailable"},
        )

    # Ensure uint8 BGR
    if image.dtype != np.uint8:
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            image = ((image.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Class ID filtering
    class_ids: Optional[List[int]] = None
    if classes:
        name_to_id = {v.lower(): k for k, v in model.names.items()}
        class_ids = [name_to_id[c.lower()] for c in classes if c.lower() in name_to_id]
        class_ids = class_ids or None

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=class_ids,
        verbose=False,
    )

    detections: List[Detection] = []
    result = results[0]

    if hasattr(result, "obb") and result.obb is not None:
        obb_data = result.obb
        xywhr = obb_data.xywhr.cpu().numpy()
        confs = obb_data.conf.cpu().numpy()
        cls_ids = obb_data.cls.cpu().numpy().astype(int)

        for i, (xywhr_row, conf_val, cls_id) in enumerate(zip(xywhr, confs, cls_ids)):
            cx, cy, w, h, angle_rad = xywhr_row
            angle_deg = float(np.degrees(angle_rad))
            label = model.names.get(int(cls_id), f"class_{cls_id}")

            corners = _obb_to_corners(float(cx), float(cy), float(w), float(h), float(angle_rad))
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]

            detections.append(
                Detection(
                    id=f"obb_{i:04d}",
                    label=label,
                    confidence=round(float(conf_val), 4),
                    bbox=BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
                    oriented_bbox=OrientedBox(
                        cx=round(float(cx), 2),
                        cy=round(float(cy), 2),
                        width=round(float(w), 2),
                        height=round(float(h), 2),
                        angle_deg=round(angle_deg, 2),
                        points=corners,
                    ),
                    area_px=round(float(w) * float(h), 1),
                    centroid=[round(float(cx), 1), round(float(cy), 1)],
                    model="yolo26n-obb",
                    dataset="DOTA-v1",
                    properties={"class_id": int(cls_id), "angle_rad": float(angle_rad)},
                )
            )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"Remote detector found {len(detections)} objects in {elapsed_ms}ms")

    return DetectionResult(
        detections=detections,
        count=len(detections),
        model="yolo26n-obb",
        dataset="DOTA-v1",
        confidence_threshold=conf,
        metadata={
            "processing_ms": elapsed_ms,
            "image_shape": list(image.shape),
            "dota_classes": DOTA_V1_CLASSES,
        },
    )
