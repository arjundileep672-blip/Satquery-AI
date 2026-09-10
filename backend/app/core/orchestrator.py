"""
SatQuery AI — Task Orchestrator
Coordinates query routing, model dispatch, and answer synthesis.
Returns a consistent AnalysisResponse regardless of task type.
"""

import io
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure backend root takes precedence on sys.path to prevent root-level ./models directory
# from shadowing backend/models as an implicit namespace package.
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir in sys.path:
    sys.path.remove(_backend_dir)
sys.path.insert(0, _backend_dir)

if "models" in sys.modules and not hasattr(sys.modules["models"], "__file__"):
    del sys.modules["models"]

import cv2
import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.core.query_router import TaskType, route_query
from app.schemas.vision import (
    AnalysisResponse,
    BoundingBox,
    ChangeRegion,
    ChangedObject,
    Detection,
    Mask,
    MultitemporalReport,
    VisualizationRef,
)


logger = get_logger("satquery.core.orchestrator")


def _bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: try PIL
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _summarize_detections(detections: List[Detection]) -> str:
    if not detections:
        return "No objects detected."
    by_label: Dict[str, int] = {}
    for d in detections:
        by_label[d.label] = by_label.get(d.label, 0) + 1
    parts = [f"{count} {label}(s)" for label, count in by_label.items()]
    return f"Detected {len(detections)} objects: {', '.join(parts)}."


def _summarize_masks(masks: List[Mask]) -> str:
    if not masks:
        return "No segments produced."
    total_area = sum(m.area_px for m in masks)
    return f"Segmented {len(masks)} regions, total area {total_area:.0f} px."


def _summarize_changes(result) -> str:
    return (
        f"{result.changed_pixels} pixels changed "
        f"({result.change_percentage:.2f}% of image). "
        f"{len(result.change_regions)} change region(s) identified. "
        f"Model: {result.model}."
    )


def _summarize_changed_objects(objs: List[ChangedObject]) -> str:
    if not objs:
        return "No changed objects identified."
    return (
        f"{len(objs)} changed object(s). "
        "Average change overlap: "
        f"{sum(o.change_overlap for o in objs)/len(objs):.1%}."
    )


class Orchestrator:
    """
    Main orchestration controller.
    Receives image bytes + query, dispatches to correct pipeline,
    synthesises NL answer, returns AnalysisResponse.
    """

    def analyze(
        self,
        image1_bytes: bytes,
        query: str,
        image2_bytes: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        additional_images_bytes: Optional[List[bytes]] = None,
        additional_metadata: Optional[List[Dict[str, Any]]] = None,
        explicit_dates: Optional[List[str]] = None,
    ) -> AnalysisResponse:
        t0 = time.perf_counter()
        req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        meta = metadata or {}
        warnings: List[str] = []

        task, rationale = route_query(query)
        logger.info(f"[{req_id}] Query routed to task={task.value} | {rationale}")

        # Decode image(s)
        image1 = _bytes_to_cv2(image1_bytes)
        image2 = _bytes_to_cv2(image2_bytes) if image2_bytes else None

        # ── EuroSAT Scene Classification (adaptive: patch or single-label) ────
        # Runs before pipeline dispatch so scene context informs detection + VLM.
        # Images ≥ 448px on their longest side → patch-based multi-label.
        # Smaller images (thumbnails, test patches) → whole-image single-label.
        scene_ctx = ""
        scene_stats: Dict[str, Any] = {}
        try:
            from models.eurosat_classifier import get_classifier
            clf = get_classifier()
            img_h, img_w = image1.shape[:2]

            if max(img_h, img_w) >= 448:
                # ── Patch-based multi-label path ──────────────────────────────
                patch_result = clf.classify_patches(image1)
                if patch_result.available and patch_result.labels:
                    top_labels = [
                        lbl for lbl in patch_result.labels if lbl.coverage_pct >= 5.0
                    ][:4]  # cap display at 4 labels for brevity
                    if not top_labels:
                        top_labels = patch_result.labels[:1]
                    composition = ", ".join(
                        f"{lbl.class_name} {lbl.coverage_pct}%"
                        for lbl in top_labels
                    )
                    scene_ctx = f"Scene composition: {composition}. "
                    # Union of YOLO hints for all labels with ≥ 10% coverage
                    from models.eurosat_classifier import SCENE_TO_YOLO_HINTS
                    yolo_hint_set: set = set()
                    for lbl in patch_result.labels:
                        if lbl.coverage_pct >= 10.0:
                            yolo_hint_set.update(SCENE_TO_YOLO_HINTS.get(lbl.class_name, []))
                    scene_stats = {
                        "patch_scene_classification": {
                            "labels": [
                                {
                                    "class_name": lbl.class_name,
                                    "coverage_pct": lbl.coverage_pct,
                                    "tile_count": lbl.tile_count,
                                    "mean_confidence": lbl.mean_confidence,
                                    "rank": lbl.rank,
                                }
                                for lbl in patch_result.labels
                            ],
                            "total_tiles": patch_result.total_tiles,
                            "tile_size": patch_result.tile_size,
                            "stride": patch_result.stride,
                            "inference_ms": patch_result.inference_ms,
                            "model": patch_result.model,
                        }
                    }
                    logger.info(
                        f"[{req_id}] Patch scene ({patch_result.total_tiles} tiles): "
                        f"{composition} in {patch_result.inference_ms}ms"
                    )
            else:
                # ── Single-label whole-image path (small images / thumbnails) ─
                scene = clf.classify(image1)
                if scene.available:
                    conf_pct = round(scene.confidence * 100, 1)
                    scene_ctx = f"Scene type: {scene.top_class} ({conf_pct}% confidence). "
                    scene_stats = {
                        "scene_classification": {
                            "top_class": scene.top_class,
                            "confidence": scene.confidence,
                            "top3": [
                                {"class": p.class_name, "probability": round(p.probability, 4)}
                                for p in scene.top3
                            ],
                            "model": scene.model,
                            "inference_ms": scene.inference_ms,
                        }
                    }
                    logger.info(
                        f"[{req_id}] Scene: {scene.top_class} ({conf_pct}%) "
                        f"in {scene.inference_ms}ms"
                    )
        except Exception as exc:
            logger.debug(f"[{req_id}] EuroSAT scene classifier skipped: {exc}")

        # ── Dispatch ──────────────────────────────────────────────────────────
        detections: List[Detection] = []
        masks: List[Mask] = []
        changes: List[ChangeRegion] = []
        changed_objects: List[ChangedObject] = []
        multitemporal_report: Optional[MultitemporalReport] = None
        statistics: Dict[str, Any] = {}
        models_used: List[str] = []
        results_summary = ""


        try:
            if task == TaskType.DETECTION:
                dr = self._run_detection(image1)
                detections = dr.detections
                statistics = {"total_detections": dr.count}
                models_used = [dr.model]
                results_summary = _summarize_detections(detections)

            elif task == TaskType.REMOTE_DETECTION:
                dr = self._run_remote_detection(image1, query)
                detections = dr.detections
                statistics = {"total_detections": dr.count}
                models_used = [dr.model]
                results_summary = _summarize_detections(detections)

            elif task == TaskType.BUILDING_DETECTION:
                dr, sr = self._run_building_detection(image1)
                detections = dr
                masks = sr
                statistics = {"buildings_count": len(dr)}
                models_used = list({d.model for d in dr} | {m.model for m in sr})
                results_summary = f"Detected {len(dr)} building(s)."

            elif task == TaskType.COUNTING:
                dr = self._run_detection(image1)
                detections = dr.detections
                statistics = {"count": dr.count}
                models_used = [dr.model]
                results_summary = _summarize_detections(detections)

            elif task == TaskType.SEGMENTATION:
                sr = self._run_segmentation(image1)
                masks = sr.masks
                statistics = {"segments": sr.count, "total_area_px": sr.total_area_px}
                models_used = [sr.model]
                results_summary = _summarize_masks(masks)

            elif task == TaskType.CHANGE_DETECTION:
                if image2 is None:
                    warnings.append("Change detection requires two images. Provide image2.")
                    results_summary = "Change detection could not run: image2 not provided."
                else:
                    cd = self._run_change_detection(image1, image2)
                    changes = cd.change_regions
                    statistics = {
                        "changed_pixels": cd.changed_pixels,
                        "total_pixels": cd.total_pixels,
                        "change_percentage": cd.change_percentage,
                        "change_regions": len(cd.change_regions),
                    }
                    if cd.fallback_used:
                        warnings.append(f"ChangeFormer not available; used {cd.model} fallback.")
                    models_used = [cd.model]
                    results_summary = _summarize_changes(cd)

            elif task == TaskType.CHANGED_OBJECTS:
                if image2 is None:
                    warnings.append("Changed-objects detection requires two images. Provide image2.")
                    results_summary = "Changed-objects pipeline could not run: image2 not provided."
                else:
                    objs, cd = self._run_changed_objects(image1, image2)
                    changed_objects = objs
                    changes = cd.change_regions
                    statistics = {
                        "changed_objects": len(objs),
                        "changed_pixels": cd.changed_pixels,
                        "change_percentage": cd.change_percentage,
                    }
                    if cd.fallback_used:
                        warnings.append(f"ChangeFormer not available; used {cd.model} fallback.")
                    models_used = [cd.model, "building_detector"]
                    results_summary = _summarize_changed_objects(objs)

            elif task == TaskType.MULTITEMPORAL_ANALYSIS:
                if image2 is None:
                    warnings.append("Multitemporal analysis requires at least two satellite images.")
                    results_summary = "Multitemporal parameter analysis could not run: second image not provided."
                else:
                    imgs = [image1, image2]
                    metas = [meta, meta.get("image2_meta", {}) if isinstance(meta, dict) else {}]
                    if additional_images_bytes:
                        for add_b in additional_images_bytes:
                            imgs.append(_bytes_to_cv2(add_b))
                        if additional_metadata:
                            metas.extend(additional_metadata)
                        else:
                            metas.extend([{} for _ in additional_images_bytes])

                    rep, chg_reg, chg_obj, m_stats = self._run_multitemporal_analysis(
                        images=imgs,
                        images_meta=metas,
                        query=query,
                        explicit_dates=explicit_dates,
                    )
                    multitemporal_report = rep
                    changes = chg_reg
                    changed_objects = chg_obj
                    statistics.update(m_stats)
                    models_used = ["ORB-RANSAC", "building_detector", "multitemporal_analyzer"]
                    results_summary = rep.executive_summary

            elif task == TaskType.SPATIAL_ANALYSIS:
                results_summary = self._run_spatial_analysis(image1, meta)
                statistics = {"spatial_analysis": True}

            else:  # IMAGE_UNDERSTANDING
                results_summary = "Scene-level image understanding requested."


        except Exception as exc:
            logger.error(f"[{req_id}] Pipeline error for task={task.value}: {exc}")
            warnings.append(f"Pipeline error: {exc}")
            results_summary = f"Analysis encountered an error: {exc}"

        # Merge scene classification into statistics and prepend to summary
        if scene_stats:
            statistics.update(scene_stats)
        if scene_ctx and results_summary:
            results_summary = scene_ctx + results_summary
        elif scene_ctx:
            results_summary = scene_ctx
        # Track eurosat model in models_used when available
        if scene_stats:
            models_used = ["eurosat_efficientnet_b0"] + models_used

        # ── VLM synthesis ─────────────────────────────────────────────────────
        try:
            try:
                from models.vlm import get_vlm_bridge
            except (ImportError, AttributeError):
                from backend.models.vlm import get_vlm_bridge
            bridge = get_vlm_bridge()
            answer = bridge.synthesize_answer(
                query=query,
                task=task.value,
                results_summary=results_summary,
                image_bytes=image1_bytes,
                mime_type="image/jpeg",
                metadata=meta,
            )
        except Exception as exc:
            logger.warning(f"[{req_id}] VLM synthesis failed: {exc}")
            answer = results_summary
            warnings.append(f"VLM answer synthesis unavailable: {exc}")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return AnalysisResponse(
            request_id=req_id,
            task=task.value,
            answer=answer,
            detections=detections,
            masks=masks,
            changes=changes,
            changed_objects=changed_objects,
            multitemporal_report=multitemporal_report,
            statistics=statistics,
            visualizations=[],
            metadata={
                "image1": meta,
                "image2_provided": image2 is not None,
                "models_used": models_used,
                "device": "cpu",
                "processing_ms": elapsed_ms,
                "query_route_rationale": rationale,
            },
            warnings=warnings,
            # Phase-1 compat fields
            operation=task.value,
            confidence=None,
            evidence=[],
            tools_used=models_used,
            models_used=models_used,
            analysis_trace=[
                f"Query received: '{query}'",
                f"Task classified: {task.value}",
                rationale,
                f"Models: {', '.join(models_used) or 'none'}",
                f"Completed in {elapsed_ms}ms",
            ],
        )

    # ── Private pipeline methods ──────────────────────────────────────────────

    def _run_multitemporal_analysis(
        self,
        images: List[np.ndarray],
        images_meta: List[Optional[Dict[str, Any]]],
        query: str,
        explicit_dates: Optional[List[str]] = None,
    ) -> Tuple[MultitemporalReport, List[ChangeRegion], List[ChangedObject], Dict[str, Any]]:
        from app.core.multitemporal_analysis import MultitemporalAnalyzer

        analyzer = MultitemporalAnalyzer()
        report = analyzer.analyze_sequence(
            images=images,
            images_meta=images_meta,
            query=query,
            explicit_dates=explicit_dates,
        )

        change_regions: List[ChangeRegion] = []
        changed_objects: List[ChangedObject] = []

        for obj in report.object_changes:
            bb = obj.bbox_pixel or [0, 0, 0, 0]
            if obj.status != "UNCHANGED":
                change_type_map = {
                    "NEW": "new_building",
                    "REMOVED": "removed_building",
                    "EXPANDED": "expanded_building",
                    "CONTRACTED": "contracted_building",
                }
                area_m2 = obj.area_date2 or obj.area_date1 or 10.0
                change_regions.append(
                    ChangeRegion(
                        region_id=obj.object_id,
                        bbox=BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]),
                        changed_pixels=int(area_m2 / 0.25),
                        area_px=float(area_m2 / 0.25),
                        area_m2=area_m2,
                        change_type=change_type_map.get(obj.status, "unclassified"),
                        model="multitemporal_analyzer",
                    )
                )

            changed_objects.append(
                ChangedObject(
                    object_id=obj.object_id,
                    label=f"{obj.object_class} ({obj.status})",
                    bbox=BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]),
                    mask_polygon=obj.polygon_pixel,
                    centroid=obj.centroid_pixel,
                    change_overlap=abs(obj.percentage_change or 0.0) / 100.0,
                    change_percentage=abs(obj.percentage_change or 0.0),
                    confidence=obj.confidence,
                )
            )

        stats: Dict[str, Any] = {
            "dates": report.dates,
            "registration_quality": report.registration_quality,
            "spatial_summary": report.spatial_summary,
        }
        for p in report.parameters:
            slug = p.parameter.lower().replace(" ", "_").replace("-", "_")
            stats[f"{slug}_date1"] = p.value_1
            stats[f"{slug}_date2"] = p.value_2
            stats[f"{slug}_abs_change"] = p.absolute_change
            stats[f"{slug}_pct_change"] = p.percentage_change

        return report, change_regions, changed_objects, stats

    def _run_detection(self, image: np.ndarray):
        # Prefer remote sensing detector (trained on DOTA aerial imagery) to prevent ground-level COCO false positives
        try:
            from models.remote_detector import detect_remote
            dr = detect_remote(image)
            if dr.model != "yolo_unavailable" and len(dr.detections) > 0:
                return dr
        except Exception:
            pass

        from models.yolo_detector import detect, AERIAL_COCO_CLASSES
        return detect(image, classes=AERIAL_COCO_CLASSES)

    def _run_remote_detection(self, image: np.ndarray, query: str):
        from models.remote_detector import detect_remote, DOTA_V1_CLASSES
        # Extract any DOTA class names mentioned in the query
        q = query.lower()
        filter_classes = [c for c in DOTA_V1_CLASSES if c in q]
        return detect_remote(image, classes=filter_classes or None)

    def _run_building_detection(self, image: np.ndarray):
        from models.building_detector import detect_buildings
        from app.schemas.vision import BoundingBox, Detection, Mask
        result = detect_buildings(image)
        dets: List[Detection] = []
        msks: List[Mask] = []
        for b in result.get("buildings", []):
            bb = b.get("bbox", [0, 0, 1, 1])
            dets.append(Detection(
                id=b["building_id"],
                label=b.get("label", "building"),
                confidence=b.get("confidence"),
                bbox=BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]),
                centroid=b.get("centroid"),
                area_px=b.get("area_px"),
                model=b.get("model", "building_detector"),
                dataset=result.get("dataset", "SpaceNet7/COCO"),
            ))
        for m in result.get("masks", []):
            msks.append(Mask(
                id=m["building_id"],
                label="building",
                polygon=m.get("polygon", []),
                area_px=m.get("area_px", 0.0),
                model=result.get("model", "building_detector"),
            ))
        return dets, msks

    def _run_segmentation(self, image: np.ndarray):
        # Generic segmentation: detect objects then SAM
        from models.yolo_detector import detect
        from models.sam2_segmenter import segment_with_boxes
        det_r = detect(image, conf=0.20)
        if not det_r.detections:
            from app.schemas.vision import SegmentationResult
            return SegmentationResult(masks=[], count=0, model="none")
        boxes = [d.bbox for d in det_r.detections]
        labels = [d.label for d in det_r.detections]
        return segment_with_boxes(image, boxes, labels)

    def _run_change_detection(self, image1: np.ndarray, image2: np.ndarray):
        from pipelines.change_detection import run_change_detection
        return run_change_detection(image1, image2)

    def _run_changed_objects(self, image1: np.ndarray, image2: np.ndarray):
        """
        Pipeline:
        1. Change detection (registration + ChangeFormer/diff)
        2. Building detection on image1
        3. Overlap building masks with change mask
        4. Return changed buildings
        """
        import numpy as np
        from pipelines.change_detection import run_change_detection
        from models.building_detector import detect_buildings
        from app.schemas.vision import ChangedObject, BoundingBox
        import cv2

        cd = run_change_detection(image1, image2)
        change_mask = np.zeros(image1.shape[:2], dtype=np.uint8)
        # Rebuild mask from region data
        for region in cd.change_regions:
            if region.bbox:
                x1, y1 = int(region.bbox.x1), int(region.bbox.y1)
                x2, y2 = int(region.bbox.x2), int(region.bbox.y2)
                change_mask[y1:y2, x1:x2] = 1

        bld_result = detect_buildings(image1)
        changed_objs: List[ChangedObject] = []

        for i, (bld, msk) in enumerate(
            zip(bld_result.get("buildings", []), bld_result.get("masks", []))
        ):
            bb = bld.get("bbox", [0, 0, 1, 1])
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            h, w = change_mask.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Build binary building mask for overlap
            bld_mask = np.zeros_like(change_mask)
            poly = msk.get("polygon", [])
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(bld_mask, [pts], 1)
            else:
                bld_mask[y1:y2, x1:x2] = 1

            bld_area = float(bld_mask.sum())
            if bld_area < 1:
                continue

            overlap_px = float((bld_mask & change_mask).sum())
            change_pct = round(overlap_px / bld_area * 100.0, 2)
            change_iou = round(overlap_px / (bld_area + float(change_mask[y1:y2, x1:x2].sum()) - overlap_px + 1e-6), 4)

            if change_pct < 5.0:  # Skip < 5% changed objects
                continue

            changed_objs.append(ChangedObject(
                object_id=bld.get("building_id", f"obj_{i:04d}"),
                label=bld.get("label", "building"),
                bbox=BoundingBox(x1=float(bb[0]), y1=float(bb[1]), x2=float(bb[2]), y2=float(bb[3])),
                mask_polygon=poly,
                centroid=bld.get("centroid"),
                change_overlap=change_iou,
                change_percentage=change_pct,
                confidence=None,  # Not fabricated
            ))

        return changed_objs, cd

    def _run_spatial_analysis(self, image: np.ndarray, meta: Dict[str, Any]) -> str:
        h, w = image.shape[:2]
        total_px = h * w
        summary = f"Image dimensions: {w}×{h} pixels ({total_px:,} total pixels)."
        if meta.get("is_geotiff") and meta.get("crs"):
            summary += f" CRS: {meta['crs']}."
            if meta.get("bounds"):
                bounds = meta["bounds"]
                summary += f" Spatial extent: {bounds}."
        else:
            summary += " No geospatial CRS available; metrics are pixel-relative."
        return summary


# Module-level singleton
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
