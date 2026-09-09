"""
SatQuery AI - Semantic Change Intelligence Pipeline

Full pipeline: two images -> registration -> change detection ->
object detection -> overlap attribution -> SAM2 refinement ->
geospatial analysis -> structured evidence.

This is the definitive pipeline for queries like:
  "Which buildings changed?"
  "What is the largest changed building?"
  "Show me all structures that appeared or disappeared."

Architecture:
    image_A + image_B
         |
    ORB + RANSAC registration
         |
    ChangeFormer (or image-diff fallback)
         |
    Change regions
         |
    SpaceNet building detection (on image_A)
         |
    Object/change overlap attribution
         |
    SAM2 mask refinement (top-N changed objects)
         |
    Geospatial analysis (if CRS available)
         |
    Evidence aggregation + structured result
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings, load_models_config
from app.core.logging import get_logger

logger = get_logger("satquery.core.semantic_change_intelligence")


def run_semantic_change_pipeline(
    image1: np.ndarray,
    image2: np.ndarray,
    object_type: str = "building",
    sam_refine_top_n: int = 5,
    geo_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full semantic change intelligence pipeline.

    Args:
        image1:           Reference image (T1), BGR uint8.
        image2:           Target image (T2), BGR uint8.
        object_type:      Object class to attribute changes to ("building", "vehicle", etc.).
        sam_refine_top_n: Refine top-N changed objects with SAM2 for precise masks.
        geo_meta:         Optional dict with CRS / affine transform for georeferencing.

    Returns:
        Structured evidence dict — see docstring output contract below.

    Output contract:
    {
        "task": "changed_objects",
        "models_used": [...],
        "registration": {"success": bool, "inlier_ratio": float, ...},
        "change": {"changed_pixels": int, "change_percentage": float, "model": str},
        "objects": [
            {
                "id": str,
                "label": str,
                "change_overlap": float,   # fraction of object area that changed
                "change_percentage": float, # percent
                "area_before_pixels": float,
                "bbox": [x1,y1,x2,y2],
                "centroid": [cx,cy],
                "mask_polygon": [[x,y],...],
                "sam_refined": bool,
            }
        ],
        "geospatial": {"available": bool, ...},
        "evidence": [...],
        "warnings": [...],
        "processing_time_s": float,
    }
    """
    t0 = time.perf_counter()
    models_used: List[str] = []
    warnings: List[str] = []
    evidence: List[str] = []

    # ── Step 1: ORB + RANSAC Registration ────────────────────────────────────
    logger.info("Step 1/5: ORB+RANSAC registration")
    from vision.registration import register_images
    reg = register_images(image1, image2)
    t_reg = time.perf_counter()

    registration_result = {
        "success": reg["success"],
        "keypoints1": reg["keypoints1"],
        "keypoints2": reg["keypoints2"],
        "matched_features": reg["matches_total"],
        "inlier_matches": reg["inliers"],
        "inlier_ratio": (
            round(reg["inliers"] / max(reg["matches_good"], 1), 3)
            if reg["success"] else 0.0
        ),
        "homography": (
            reg["homography"].tolist()
            if reg.get("homography") is not None else None
        ),
        "registration_ms": reg["processing_ms"],
    }

    if reg["success"]:
        image2_aligned = reg["aligned_image"]
        models_used.append("ORB-RANSAC")
        evidence.append(
            f"Registration: {reg['inliers']} inliers from {reg['matches_good']} "
            f"good matches (inlier_ratio={registration_result['inlier_ratio']:.2f})."
        )
    else:
        image2_aligned = _to_bgr_uint8(image2)
        warnings.append(
            f"Registration failed: {reg.get('error', 'unknown')}. "
            "Using unregistered image2 — change results may contain misalignment artefacts."
        )
        evidence.append(f"Registration FAILED: {reg.get('error')}.")

    # Enforce exact spatial alignment with image1 frame
    h1, w1 = image1.shape[:2]
    if image2_aligned.shape[:2] != (h1, w1):
        logger.info(f"Resizing image2 ({image2_aligned.shape[:2]}) to match image1 ({h1}, {w1})")
        image2_aligned = cv2.resize(image2_aligned, (w1, h1), interpolation=cv2.INTER_LINEAR)

    # ── Step 2: ChangeFormer ──────────────────────────────────────────────────
    logger.info("Step 2/5: Change detection (ChangeFormer)")
    from models.changeformer import detect_changes
    cd = detect_changes(image1, image2_aligned)
    t_cd = time.perf_counter()

    change_mask: np.ndarray = cd["change_mask"]  # H x W, uint8 0/1
    total_px = int(change_mask.size)
    changed_px = int(change_mask.sum())
    change_pct = round(100.0 * changed_px / max(total_px, 1), 3)

    models_used.append(cd["model"])
    if cd.get("fallback_used"):
        warnings.append(f"ChangeFormer not available; used {cd['model']} fallback.")

    change_result = {
        "changed_pixels": changed_px,
        "total_pixels": total_px,
        "change_percentage": change_pct,
        "model": cd["model"],
        "fallback_used": cd.get("fallback_used", False),
        "changeformer_ms": round((t_cd - t_reg) * 1000, 2),
    }
    evidence.append(
        f"Change detection ({cd['model']}): {changed_px}/{total_px} px "
        f"changed ({change_pct}%)."
    )

    # ── Step 3: Object Detection ──────────────────────────────────────────────
    logger.info(f"Step 3/5: {object_type} detection")
    object_result = _detect_objects(image1, object_type)
    t_obj = time.perf_counter()
    models_used.append(object_result["model"])

    objects_detected = object_result["buildings"]
    masks_detected   = object_result["masks"]
    evidence.append(
        f"Object detection ({object_result['model']}): "
        f"{len(objects_detected)} {object_type}(s) detected."
    )

    # ── Step 4: Attribution (overlap analysis) ────────────────────────────────
    logger.info("Step 4/5: Change attribution")
    cfg_change = load_models_config().get("change", {})
    # Read threshold from models.yaml change section
    # Load the full YAML to get the 'change' key (not inside 'models')
    try:
        import yaml, os
        cfg_path = settings.MODELS_CONFIG_PATH
        with open(cfg_path) as f:
            full_cfg = yaml.safe_load(f) or {}
        overlap_threshold = full_cfg.get("change", {}).get("object_overlap_threshold", 0.30)
    except Exception:
        overlap_threshold = 0.30

    changed_objects = _attribute_changes(
        change_mask=change_mask,
        buildings=objects_detected,
        masks=masks_detected,
        overlap_threshold=overlap_threshold,
    )
    t_attr = time.perf_counter()

    evidence.append(
        f"Attribution (threshold={overlap_threshold}): "
        f"{len(changed_objects)} {object_type}(s) flagged as changed."
    )

    # ── Step 5: SAM2 Refinement (top-N changed objects) ──────────────────────
    logger.info(f"Step 5/5: SAM2 refinement (top {sam_refine_top_n})")
    # Sort by change_overlap descending; refine top-N
    changed_objects.sort(key=lambda o: o["change_overlap"], reverse=True)
    to_refine = changed_objects[:sam_refine_top_n]

    sam_refined_count = 0
    if to_refine:
        try:
            from models.sam2_segmenter import segment_with_boxes
            from app.schemas.vision import BoundingBox

            boxes = [
                BoundingBox(
                    x1=obj["bbox"][0], y1=obj["bbox"][1],
                    x2=obj["bbox"][2], y2=obj["bbox"][3],
                )
                for obj in to_refine
            ]
            labels = [obj["label"] for obj in to_refine]
            seg_result = segment_with_boxes(image1, boxes, labels)

            for i, (obj, mask) in enumerate(zip(to_refine, seg_result.masks)):
                obj["mask_polygon"] = mask.polygon
                obj["area_before_pixels"] = mask.area_px
                obj["sam_refined"] = True
                sam_refined_count += 1

            if seg_result.model not in models_used:
                models_used.append(seg_result.model)
            evidence.append(
                f"SAM2 refinement: {sam_refined_count} object(s) mask-refined "
                f"using {seg_result.model}."
            )
        except Exception as exc:
            warnings.append(f"SAM2 refinement skipped: {exc}")
            logger.warning(f"SAM2 refinement failed: {exc}")

    t_end = time.perf_counter()

    # ── Geospatial analysis ───────────────────────────────────────────────────
    geo_result = _geospatial_analysis(changed_objects, geo_meta)

    # ── Timing summary ────────────────────────────────────────────────────────
    timing = {
        "registration_ms": registration_result["registration_ms"],
        "change_detection_ms": change_result["changeformer_ms"],
        "object_detection_ms": round((t_obj - t_cd) * 1000, 2),
        "attribution_ms": round((t_attr - t_obj) * 1000, 2),
        "sam_refinement_ms": round((t_end - t_attr) * 1000, 2),
        "total_ms": round((t_end - t0) * 1000, 2),
    }
    logger.info(
        f"Semantic change pipeline complete: "
        f"{len(changed_objects)} changed {object_type}(s), "
        f"total={timing['total_ms']}ms"
    )

    return {
        "task": "changed_objects",
        "object_type": object_type,
        "models_used": models_used,
        "registration": registration_result,
        "change": change_result,
        "objects": changed_objects,
        "geospatial": geo_result,
        "evidence": evidence,
        "warnings": warnings,
        "timing": timing,
        "processing_time_s": round((t_end - t0), 3),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        mn, mx = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - mn) / max(mx - mn, 1e-6) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _detect_objects(image: np.ndarray, object_type: str) -> Dict[str, Any]:
    """Route object detection to appropriate model based on object_type."""
    object_type_lower = object_type.lower()

    if "building" in object_type_lower or "structure" in object_type_lower:
        from models.building_detector import detect_buildings
        return detect_buildings(image)

    # Generic YOLO for other object types
    from models.yolo_detector import detect as yolo_detect
    det = yolo_detect(image, classes=[object_type_lower], conf=0.20)
    buildings_list = []
    masks_list = []
    for i, d in enumerate(det.detections):
        bid = f"obj_{i:04d}"
        buildings_list.append({
            "building_id": bid,
            "label": d.label,
            "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
            "confidence": d.confidence,
            "centroid": d.centroid,
            "area_px": d.area_px,
            "model": d.model,
        })
        masks_list.append({
            "building_id": bid,
            "polygon": [],
            "area_px": d.area_px or 0.0,
        })
    return {
        "buildings": buildings_list,
        "masks": masks_list,
        "count": len(buildings_list),
        "model": det.model,
        "dataset": det.dataset,
    }


def _attribute_changes(
    change_mask: np.ndarray,
    buildings: List[Dict[str, Any]],
    masks: List[Dict[str, Any]],
    overlap_threshold: float,
) -> List[Dict[str, Any]]:
    """
    For each detected object, compute overlap with change_mask.

    overlap_ratio = changed_pixels_inside_object / object_pixels

    Objects with overlap_ratio >= overlap_threshold are flagged as changed.
    The threshold is configurable (models.yaml change.object_overlap_threshold).
    It is NOT scientifically validated — treat as a tunable heuristic.
    """
    h, w = change_mask.shape
    changed_objs = []

    for bld, msk in zip(buildings, masks):
        bb = bld.get("bbox", [0, 0, 1, 1])
        x1, y1, x2, y2 = (
            max(0, int(bb[0])), max(0, int(bb[1])),
            min(w, int(bb[2])), min(h, int(bb[3])),
        )
        if x2 <= x1 or y2 <= y1:
            continue

        # Build binary object mask from polygon or bbox
        obj_mask = np.zeros_like(change_mask)
        poly = msk.get("polygon", [])
        if poly and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(obj_mask, [pts], 1)
        else:
            obj_mask[y1:y2, x1:x2] = 1

        obj_area = float(obj_mask.sum())
        if obj_area < 1:
            continue

        overlap_px = float((obj_mask & change_mask).sum())
        overlap_ratio = overlap_px / obj_area

        if overlap_ratio < overlap_threshold:
            continue

        # Change IoU with local change window (not global, to avoid division by zero)
        local_change = float(change_mask[y1:y2, x1:x2].sum())
        change_iou = overlap_px / max(obj_area + local_change - overlap_px, 1e-6)

        changed_objs.append({
            "id": bld.get("building_id", f"obj_{len(changed_objs):04d}"),
            "label": bld.get("label", "object"),
            "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
            "centroid": bld.get("centroid"),
            "area_before_pixels": msk.get("area_px", obj_area),
            "changed_pixels_in_object": int(overlap_px),
            "change_overlap": round(overlap_ratio, 4),
            "change_iou": round(change_iou, 4),
            "change_percentage": round(overlap_ratio * 100.0, 2),
            "mask_polygon": poly,
            "sam_refined": False,
            "model": bld.get("model", "unknown"),
        })

    return changed_objs


def _geospatial_analysis(
    changed_objects: List[Dict[str, Any]],
    geo_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Enrich changed objects with georeferenced measurements if CRS is available.
    Returns a geo_result dict.
    """
    if not geo_meta or not geo_meta.get("is_geotiff") or not geo_meta.get("crs"):
        return {
            "available": False,
            "reason": "No valid CRS/geotransform in image metadata.",
            "area_available": False,
        }

    try:
        from app.geospatial.raster_ops import pixel_to_crs, pixel_area_to_m2
        crs = geo_meta.get("crs")
        transform = geo_meta.get("transform")
        if not transform:
            return {"available": False, "reason": "No affine transform in metadata.", "area_available": False}

        for obj in changed_objects:
            cx, cy = obj.get("centroid") or [None, None]
            if cx is not None and cy is not None:
                try:
                    lon, lat = pixel_to_crs(cx, cy, transform, crs)
                    obj["geo_centroid"] = {"lon": round(lon, 6), "lat": round(lat, 6)}
                except Exception:
                    pass

            area_px = obj.get("area_before_pixels", 0)
            if area_px:
                try:
                    area_m2 = pixel_area_to_m2(area_px, transform, crs)
                    obj["area_m2"] = round(area_m2, 1)
                except Exception:
                    pass

        return {
            "available": True,
            "crs": str(crs),
            "area_available": True,
        }

    except Exception as exc:
        logger.warning(f"Geospatial enrichment failed: {exc}")
        return {
            "available": False,
            "reason": f"Geospatial analysis error: {exc}",
            "area_available": False,
        }
