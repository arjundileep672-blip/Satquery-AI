"""
SatQuery AI — SAM 2.1 Segmenter
Wraps Meta SAM 2.1 for prompted instance segmentation.

Primary pipeline:
  Detector -> bounding box -> SAM 2.1 -> precise mask

Fallback (if SAM2 not installed or PyTorch unavailable):
  GrabCut-based segmentation (OpenCV — always available).

SAM 2.1 is lazy-loaded on first use.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings, load_models_config, resolve_device
from app.core.logging import get_logger
from app.schemas.vision import BoundingBox, Mask, SegmentationResult

logger = get_logger("satquery.models.sam2")

_predictor = None   # SAM2 predictor singleton
_sam2_available: Optional[bool] = None  # None = not yet probed


def _probe_sam2() -> bool:
    """Check whether SAM2 + PyTorch are importable and checkpoint exists."""
    global _sam2_available
    if _sam2_available is not None:
        return _sam2_available
    try:
        import torch  # noqa: F401
        from sam2.build_sam import build_sam2  # noqa: F401
        _sam2_available = True
    except ImportError:
        _sam2_available = False
    return _sam2_available


def _load_sam2():
    """Load SAM2 predictor lazily. Raises ImportError if unavailable."""
    global _predictor
    if _predictor is not None:
        return _predictor
    if not _probe_sam2():
        raise ImportError("SAM2 or PyTorch not installed.")

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg = load_models_config().get("sam2", {})
    ckpt_rel = cfg.get("checkpoint", "models/sam2/sam2.1_hiera_tiny.pt")
    model_cfg = cfg.get("model_cfg", "sam2.1_hiera_t.yaml")
    
    ckpt_path = settings.BASE_DIR.parent / ckpt_rel
    if not ckpt_path.exists():
        fallback_path = settings.BASE_DIR.parent / "checkpoints" / "sam2.1_hiera_tiny.pt"
        if fallback_path.exists():
            ckpt_path = fallback_path

    device = resolve_device()

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {ckpt_path}. "
            "Run python scripts/download_models.py to download."
        )

    sam2_model = build_sam2(model_cfg, str(ckpt_path), device=device)
    _predictor = SAM2ImagePredictor(sam2_model)
    logger.info(f"SAM2 loaded on device={device}")
    return _predictor


class SAM2Segmenter:
    """
    Standard interface for SAM 2.1 Segmenter.
    Implements lazy loading and prompt-based segmentation.
    """

    def __init__(self, checkpoint: Optional[str] = None, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self.predictor = None

    def load(self):
        """Lazy loader: loads SAM2 only when requested."""
        if self.predictor is None and _probe_sam2():
            try:
                self.predictor = _load_sam2()
            except Exception as exc:
                logger.warning(f"SAM2 lazy load failed: {exc}; using GrabCut fallback.")
        return self

    def segment_from_box(self, image: np.ndarray, box: Any) -> Dict[str, Any]:
        """Segment single object given bounding box [x1, y1, x2, y2] or BoundingBox."""
        self.load()
        if isinstance(box, (list, tuple)):
            bbox = BoundingBox(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]))
        else:
            bbox = box

        res = segment_with_boxes(image, [bbox])
        if res.masks:
            m = res.masks[0]
            return {
                "mask": m.polygon,
                "polygon": m.polygon,
                "area_pixels": m.area_px,
                "model": m.model,
            }
        return {"mask": [], "polygon": [], "area_pixels": 0.0, "model": "none"}

    def segment_from_point(self, image: np.ndarray, point: Tuple[float, float], label: int = 1) -> Dict[str, Any]:
        """Segment given point (x, y) prompt."""
        self.load()
        # Create a small surrogate bounding box around point for fallback compatibility
        px, py = point
        delta = 20.0
        bbox = BoundingBox(x1=max(0, px - delta), y1=max(0, py - delta), x2=px + delta, y2=py + delta)
        return self.segment_from_box(image, bbox)

    def segment_from_mask(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Refine a coarse mask using bounding box contour."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return {"mask": [], "polygon": [], "area_pixels": 0.0, "model": "none"}
        bbox = BoundingBox(x1=float(xs.min()), y1=float(ys.min()), x2=float(xs.max()), y2=float(ys.max()))
        return self.segment_from_box(image, bbox)


def _mask_to_polygon(mask: np.ndarray, simplify_epsilon: float = 1.0) -> List[List[float]]:
    """Convert binary mask to polygon contour with adaptive simplification."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    # Try progressively smaller epsilon for finer polygons (min 5 points)
    for eps_factor in (simplify_epsilon, simplify_epsilon * 0.5, 0.5, 0.1):
        arc = cv2.arcLength(largest, True)
        if arc == 0:
            break
        approx = cv2.approxPolyDP(largest, eps_factor * arc, True)
        pts = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
        if len(pts) >= 5:
            return pts
    # Full contour as last resort (no simplification)
    pts = [[float(pt[0][0]), float(pt[0][1])] for pt in largest]
    if len(pts) >= 3:
        return pts
    x, y, w, h = cv2.boundingRect(largest)
    return [
        [float(x),     float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x),     float(y + h)],
    ]


def _grabcut_segment(
    image: np.ndarray,
    bbox: BoundingBox,
    iterations: int = 5,
) -> np.ndarray:
    """
    GrabCut segmentation with localized ROI processing for high performance:
    - Crops image to the bounding box region with small contextual padding
    - CLAHE preprocessing for edge contrast
    - Morphological closing post-processing
    - Translates binary mask back to image dimensions
    """
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
    x2, y2 = min(w_img, int(bbox.x2)), min(h_img, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h_img, w_img), dtype=np.uint8)

    bw, bh = x2 - x1, y2 - y1
    pad_x = max(2, int(bw * 0.15))
    pad_y = max(2, int(bh * 0.15))

    roi_x1 = max(0, x1 - pad_x)
    roi_y1 = max(0, y1 - pad_y)
    roi_x2 = min(w_img, x2 + pad_x)
    roi_y2 = min(h_img, y2 + pad_y)

    crop = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    if crop.dtype != np.uint8:
        img_min, img_max = float(crop.min()), float(crop.max())
        if img_max > img_min:
            crop = ((crop.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            crop = np.zeros_like(crop, dtype=np.uint8)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Local rect inside cropped ROI
    local_x1 = x1 - roi_x1
    local_y1 = y1 - roi_y1
    local_w = bw
    local_h = bh
    rect = (local_x1, local_y1, local_w, local_h)

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    mask_gc = np.zeros(crop.shape[:2], dtype=np.uint8)

    try:
        cv2.grabCut(crop, mask_gc, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        binary_crop = np.where((mask_gc == 1) | (mask_gc == 3), 1, 0).astype(np.uint8)
        if binary_crop.sum() == 0:
            binary_crop[local_y1:local_y1 + local_h, local_x1:local_x1 + local_w] = 1
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_crop = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, kernel, iterations=1)
    except cv2.error:
        binary_crop = np.zeros(crop.shape[:2], dtype=np.uint8)
        binary_crop[local_y1:local_y1 + local_h, local_x1:local_x1 + local_w] = 1

    out = np.zeros((h_img, w_img), dtype=np.uint8)
    out[roi_y1:roi_y2, roi_x1:roi_x2] = binary_crop
    return out


def segment_with_boxes(
    image: np.ndarray,
    boxes: List[BoundingBox],
    labels: Optional[List[str]] = None,
    multimask: bool = False,
) -> SegmentationResult:
    t0 = time.perf_counter()

    if not boxes:
        return SegmentationResult(masks=[], count=0, model="none", metadata={})

    labels = labels or [f"object_{i}" for i in range(len(boxes))]
    use_sam2 = _probe_sam2()
    model_name = "sam2.1-tiny" if use_sam2 else "grabcut-fallback"

    masks_out: List[Mask] = []
    total_area = 0.0

    img_rgb = image
    if image.ndim == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 2:
        img_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    predictor = None
    if use_sam2:
        try:
            predictor = _load_sam2()
            # Set image once for all bounding boxes
            predictor.set_image(img_rgb)
        except (ImportError, FileNotFoundError, Exception) as exc:
            logger.warning(f"SAM2 fallback to GrabCut: {exc}")
            use_sam2 = False
            predictor = None
            model_name = "grabcut-fallback"

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if use_sam2 and predictor is not None:
            try:
                np_box = np.array([[box.x1, box.y1, box.x2, box.y2]], dtype=np.float32)
                sam_masks, scores, _ = predictor.predict(
                    box=np_box,
                    multimask_output=multimask,
                )
                best_idx = int(np.argmax(scores))
                binary_mask = sam_masks[best_idx].astype(np.uint8)
            except Exception as exc:
                logger.warning(f"SAM2 predict failed for box {i}: {exc}; using GrabCut")
                binary_mask = _grabcut_segment(image, box)
        else:
            binary_mask = _grabcut_segment(image, box)

        area_px = float(binary_mask.sum())
        polygon = _mask_to_polygon(binary_mask)
        total_area += area_px

        masks_out.append(
            Mask(
                id=f"mask_{i:04d}",
                label=label,
                polygon=polygon,
                bbox=box,
                area_px=area_px,
                confidence=None,
                model=model_name,
            )
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"Segmented {len(masks_out)} masks in {elapsed_ms}ms using {model_name}")

    return SegmentationResult(
        masks=masks_out,
        count=len(masks_out),
        total_area_px=total_area,
        model=model_name,
        metadata={"processing_ms": elapsed_ms, "sam2_used": use_sam2},
    )
