"""
SatQuery AI — Change Detection Pipeline
Full pipeline: Image A + Image B → ORB/RANSAC → ChangeFormer → Statistics

Steps:
  1. Load both images
  2. ORB + RANSAC registration (align image2 to image1)
  3. ChangeFormer (or image-diff fallback)
  4. Connected-component analysis for change regions
  5. Statistics
"""

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.logging import get_logger
from app.schemas.vision import BoundingBox, ChangeDetectionResult, ChangeRegion

logger = get_logger("satquery.pipelines.change_detection")


def _extract_change_regions(
    change_mask: np.ndarray,
    min_area_px: int = 25,
) -> List[ChangeRegion]:
    """
    Label connected components in the binary change mask.
    Returns a list of ChangeRegion objects sorted by area (descending).
    """
    n_labels, labels_im = cv2.connectedComponents(change_mask)
    regions: List[ChangeRegion] = []

    for lbl in range(1, n_labels):
        comp = (labels_im == lbl).astype(np.uint8)
        area_px = float(comp.sum())
        if area_px < min_area_px:
            continue

        ys, xs = np.where(comp)
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())

        regions.append(
            ChangeRegion(
                region_id=f"chg_{lbl:04d}",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                changed_pixels=int(area_px),
                area_px=area_px,
                area_m2=None,  # Only set if georeferencing is available
                change_type="unclassified",
                model="",  # Set by caller
            )
        )

    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions


def run_change_detection(
    image1: np.ndarray,
    image2: np.ndarray,
    skip_registration: bool = False,
    change_threshold: Optional[float] = None,
) -> ChangeDetectionResult:
    """
    Full change detection pipeline.

    Args:
        image1:              Reference image (T1).
        image2:              Target image (T2) — will be registered to image1.
        skip_registration:   If True, assumes image2 is already aligned.
        change_threshold:    Optional manual threshold for image-diff fallback [0,1].

    Returns:
        ChangeDetectionResult with change mask shape, statistics, and regions.
    """
    t0 = time.perf_counter()
    registration_metrics: Dict[str, Any] = {}

    # ── Step 1: Registration ──────────────────────────────────────────────────
    if not skip_registration:
        from vision.registration import register_images
        reg = register_images(image1, image2)
        registration_metrics = {
            "success": reg["success"],
            "keypoints1": reg["keypoints1"],
            "keypoints2": reg["keypoints2"],
            "matches_good": reg["matches_good"],
            "inliers": reg["inliers"],
            "registration_ms": reg["processing_ms"],
        }
        if reg["success"]:
            image2_aligned = reg["aligned_image"]
            logger.info(
                f"Registration OK: {reg['inliers']} inliers, "
                f"{reg['matches_good']} good matches"
            )
        else:
            logger.warning(f"Registration failed ({reg['error']}); using unregistered image2.")
            # Convert to BGR uint8 for consistency
            if image2.dtype != np.uint8:
                mn, mx = float(image2.min()), float(image2.max())
                if mx > mn:
                    image2_aligned = ((image2.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    image2_aligned = np.zeros_like(image2, dtype=np.uint8)
            else:
                image2_aligned = image2.copy()
            if image2_aligned.ndim == 2:
                image2_aligned = cv2.cvtColor(image2_aligned, cv2.COLOR_GRAY2BGR)
    else:
        image2_aligned = image2

    # Enforce exact spatial alignment with image1 frame
    h1, w1 = image1.shape[:2]
    if image2_aligned.shape[:2] != (h1, w1):
        logger.info(f"Resizing image2 ({image2_aligned.shape[:2]}) to match image1 ({h1}, {w1})")
        image2_aligned = cv2.resize(image2_aligned, (w1, h1), interpolation=cv2.INTER_LINEAR)

    # ── Step 2: Change Detection ──────────────────────────────────────────────
    try:
        from models.changeformer import detect_changes
    except (ImportError, AttributeError):
        from backend.models.changeformer import detect_changes
    cd_result = detect_changes(image1, image2_aligned, threshold=change_threshold)
    change_mask: np.ndarray = cd_result["change_mask"]
    model_name: str = cd_result["model"]
    fallback_used: bool = cd_result["fallback_used"]

    # ── Step 3: Statistics ────────────────────────────────────────────────────
    total_pixels = int(change_mask.size)
    changed_pixels = int(change_mask.sum())
    change_pct = round(100.0 * changed_pixels / max(total_pixels, 1), 3)

    # ── Step 4: Regions ───────────────────────────────────────────────────────
    regions = _extract_change_regions(change_mask)
    for r in regions:
        r.model = model_name

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"Change detection: {changed_pixels}/{total_pixels} px changed "
        f"({change_pct}%), {len(regions)} regions, model={model_name}, "
        f"total={elapsed_ms}ms"
    )

    return ChangeDetectionResult(
        change_mask_shape=list(change_mask.shape),
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        change_percentage=change_pct,
        change_regions=regions,
        registration_metrics=registration_metrics,
        model=model_name,
        fallback_used=fallback_used,
        metadata={
            "processing_ms": elapsed_ms,
            "skip_registration": skip_registration,
        },
    )
