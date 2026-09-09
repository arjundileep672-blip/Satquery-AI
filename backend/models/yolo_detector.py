"""
SatQuery AI — YOLO Generic Object Detector
Wraps Ultralytics YOLO for fast object detection on single images.

IMPORTANT PROVENANCE DISCLAIMER:
- Default model: YOLO12n (fallback: yolo11n.pt) (COCO-trained generic detector)
- This model is NOT trained on remote-sensing/satellite imagery.
- It cannot reliably detect small objects at satellite spatial resolutions.
- Use remote_detector.py (YOLO26n-OBB / DOTA-v1) for aerial imagery.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings, load_models_config, resolve_device
from app.core.logging import get_logger
from app.schemas.vision import BoundingBox, Detection, DetectionResult

logger = get_logger("satquery.models.yolo")

# COCO classes that are physically plausible from aerial / remote-sensing perspectives.
# Excludes ground-level / indoor items like "traffic light", "fire hydrant", "bench",
# "chair", "dining table", etc. which produce high false-positive rates from overhead views.
AERIAL_COCO_CLASSES = [
    "airplane", "car", "bus", "truck", "boat", "train", "motorcycle", "bicycle"
]

EXCLUDED_COCO_CLASSES = {
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
}

_model_instance = None  # Lazy singleton


def _load_model(checkpoint: Optional[str] = None, device: Optional[str] = None):
    """Lazy-load YOLO model. Raises ImportError if Ultralytics is not installed."""
    global _model_instance
    if _model_instance is not None and checkpoint is None:
        return _model_instance

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    cfg = load_models_config().get("yolo", {})
    ckpt_rel = checkpoint or cfg.get("checkpoint", "models/yolo/yolo12n.pt")
    fallback_rel = cfg.get("fallback_checkpoint", "checkpoints/yolo11n.pt")
    
    ckpt_path = settings.BASE_DIR.parent / ckpt_rel
    fb_path = settings.BASE_DIR.parent / fallback_rel
    root_fallback = settings.BASE_DIR.parent / "yolo11n.pt"

    if ckpt_path.exists():
        ckpt = str(ckpt_path)
    elif fb_path.exists():
        logger.info(f"YOLO primary checkpoint not found at {ckpt_path}; using fallback {fb_path}")
        ckpt = str(fb_path)
    elif root_fallback.exists():
        ckpt = str(root_fallback)
    else:
        # Fall back to hub download using just the model name
        ckpt = Path(ckpt_rel).name

    logger.info(f"Loading YOLO model from: {ckpt}")
    model = YOLO(ckpt)
    
    dev = device or cfg.get("device", resolve_device())
    if dev != "auto":
        model.to(dev)

    if checkpoint is None:
        _model_instance = model
    return model


class YOLODetector:
    """
    Standard Object Detector interface for YOLO12n.
    """

    def __init__(self, checkpoint: Optional[str] = None, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    def load(self):
        """Load model into memory lazily."""
        if self.model is None:
            self.model = _load_model(self.checkpoint, self.device)
        return self

    def predict(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        conf: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Run detection and return structured dictionary format.
        """
        self.load()
        res = detect(image, classes=classes, conf=conf)
        detections = []
        for d in res.detections:
            cls_id = d.properties.get("class_id", 0)
            detections.append({
                "class_id": cls_id,
                "class_name": d.label,
                "confidence": d.confidence,
                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
            })
        return {
            "detections": detections,
            "count": len(detections),
        }

    def count(self, image: np.ndarray, classes: Optional[List[str]] = None) -> int:
        """Count detected objects."""
        res = self.predict(image, classes=classes)
        return res["count"]


def detect(
    image: np.ndarray,
    classes: Optional[List[str]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> DetectionResult:
    """
    Run YOLO detection on a single image.
    """
    t0 = time.perf_counter()

    try:
        model = _load_model()
    except (ImportError, Exception) as exc:
        logger.warning(f"YOLO unavailable: {exc}")
        return DetectionResult(
            detections=[],
            count=0,
            model="yolo_unavailable",
            dataset="N/A",
            confidence_threshold=conf,
            metadata={"error": str(exc), "status": "unavailable"},
        )

    # Ensure uint8 BGR for Ultralytics
    if image.dtype != np.uint8:
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            image = ((image.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Class filtering: resolve string names -> integer IDs
    class_ids: Optional[List[int]] = None
    if classes:
        names_lower = {v.lower(): k for k, v in model.names.items()}
        class_ids = [names_lower[c.lower()] for c in classes if c.lower() in names_lower]
        if not class_ids:
            class_ids = None

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
    boxes = result.boxes

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
        conf_val = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        label = model.names.get(cls_id, f"class_{cls_id}")

        # Unless caller explicitly requested this class, suppress non-aerial ground/indoor classes
        if classes is None and label.lower() in EXCLUDED_COCO_CLASSES:
            continue

        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area_px = (x2 - x1) * (y2 - y1)

        detections.append(
            Detection(
                id=f"det_{len(detections):04d}",
                label=label,
                confidence=round(conf_val, 4),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                oriented_bbox=None,
                area_px=round(area_px, 1),
                centroid=[round(cx, 1), round(cy, 1)],
                model="yolo12n",
                dataset="COCO",
                properties={"class_id": cls_id},
            )
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"YOLO detected {len(detections)} objects in {elapsed_ms}ms")

    return DetectionResult(
        detections=detections,
        count=len(detections),
        model="yolo12n",
        dataset="COCO (generic — not remote-sensing-specific)",
        confidence_threshold=conf,
        metadata={
            "processing_ms": elapsed_ms,
            "image_shape": list(image.shape),
            "imgsz": imgsz,
        },
    )
