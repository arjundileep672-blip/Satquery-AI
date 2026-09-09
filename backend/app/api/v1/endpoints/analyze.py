"""
Analysis Endpoint for SatQuery AI
Processes multipart/form-data image upload and natural language query.

Phase 2: Routes advanced vision tasks (YOLO, SAM2, ChangeFormer, changed-objects)
through the new Orchestrator. Falls back to Phase-1 Planner for VQA/understanding.
"""

import asyncio
import uuid
from typing import Any, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.agents.planner import SatQueryPlanner
from app.core.config import settings
from app.core.logging import get_logger
from app.core.query_router import TaskType, route_query
from app.models.gemini_adapter import ModelProviderUnavailableError
from app.schemas.analysis import AnalysisResult
from app.services.image_service import (
    ImageProcessingError,
    extract_metadata,
    prepare_image_for_vlm,
    sanitize_filename,
    validate_file,
)
from app.tools.registry import registry

router = APIRouter()
logger = get_logger("satquery.api.analyze")
planner = SatQueryPlanner(registry)

# Tasks handled by the new Orchestrator (Phase 2 pipelines)
_ORCHESTRATOR_TASKS = {
    TaskType.DETECTION,
    TaskType.REMOTE_DETECTION,
    TaskType.BUILDING_DETECTION,
    TaskType.COUNTING,
    TaskType.SEGMENTATION,
    TaskType.CHANGE_DETECTION,
    TaskType.CHANGED_OBJECTS,
    TaskType.MULTITEMPORAL_ANALYSIS,
    TaskType.SPATIAL_ANALYSIS,
}



def _is_valid_upload(f: Any) -> bool:
    return f is not None and bool(getattr(f, "filename", None))


async def _process_upload(upload_file: UploadFile):
    filename = sanitize_filename(upload_file.filename or "uploaded_raster.png")
    file_bytes = await upload_file.read()
    validate_file(filename, upload_file.content_type, len(file_bytes))
    metadata = extract_metadata(file_bytes, filename)
    mime_type = upload_file.content_type or "image/png"
    vlm_bytes, vlm_mime = prepare_image_for_vlm(file_bytes, filename, mime_type)
    return file_bytes, vlm_bytes, vlm_mime, metadata


def _bbox_to_pixel_list(bbox: dict) -> list:
    """Convert BoundingBox dict {x1,y1,x2,y2} to [x1,y1,x2,y2] pixel list."""
    if bbox is None:
        return [0, 0, 0, 0]
    return [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]


def _polygon_to_geojson_ring(polygon: list) -> dict | None:
    """
    Convert a flat [[x,y], ...] polygon (pixel coords) to a GeoJSON-like
    geometry dict that the frontend ImageViewer expects:
      { type: "Polygon", coordinates: [[[x,y], ...]] }
    Returns None if polygon is invalid.
    """
    if not polygon or len(polygon) < 3:
        return None
    # Close the ring if not already closed
    ring = [list(pt) for pt in polygon]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _change_bbox_to_geojson(bbox: dict | None) -> dict | None:
    """Convert ChangeRegion bbox to a rectangular GeoJSON polygon."""
    if bbox is None:
        return None
    x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
    x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
    if x1 == x2 or y1 == y2:
        return None
    ring = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
    return {"type": "Polygon", "coordinates": [ring]}


def _orchestrator_result_to_analysis(orch_result) -> AnalysisResult:
    """
    Convert Orchestrator's AnalysisResponse to the AnalysisResult schema
    used by Phase-1 endpoints so the response_model is satisfied.

    Applies coordinate translation so the frontend ImageViewer can render
    bounding boxes (bbox_pixel list) and masks (geometry.coordinates GeoJSON).
    """
    # ── Detections: add bbox_pixel for frontend overlay rendering ──────────────
    detections = []
    for d in orch_result.detections:
        d_dict = d.model_dump()
        bbox = d_dict.get("bbox") or {}
        d_dict["bbox_pixel"] = _bbox_to_pixel_list(bbox)
        # Flatten oriented bbox to the shape the frontend expects
        obb = d_dict.get("oriented_bbox")
        if obb:
            d_dict["oriented_bbox"] = {
                "width_px": obb.get("width", 0),
                "height_px": obb.get("height", 0),
                "angle_degrees": obb.get("angle_deg", 0),
            }
        detections.append(d_dict)

    # ── Masks: convert polygon → GeoJSON geometry.coordinates ─────────────────
    masks = []
    for m in orch_result.masks:
        m_dict = m.model_dump()
        polygon = m_dict.get("polygon") or []
        geojson_geom = _polygon_to_geojson_ring(polygon)
        if geojson_geom:
            m_dict["geometry"] = geojson_geom
            m_dict["coordinates"] = geojson_geom["coordinates"]
        masks.append(m_dict)

    # ── Changes: convert bbox → geometry_geojson.coordinates ──────────────────
    changes = []
    for c in orch_result.changes:
        c_dict = c.model_dump()
        bbox = c_dict.get("bbox")
        geojson_geom = _change_bbox_to_geojson(bbox)
        if geojson_geom:
            c_dict["geometry_geojson"] = geojson_geom
        changes.append(c_dict)

    # ── Multitemporal Report: serialize if present ─────────────────────────────
    multi_report_dict = (
        orch_result.multitemporal_report.model_dump()
        if getattr(orch_result, "multitemporal_report", None)
        else None
    )

    return AnalysisResult(
        request_id=orch_result.request_id,
        answer=orch_result.answer,
        operation=orch_result.task,
        task=orch_result.task,
        confidence=orch_result.confidence,
        detections=detections,
        masks=masks,
        changes=changes,
        multitemporal_report=multi_report_dict,
        statistics=orch_result.statistics,
        visualizations=[v.model_dump() for v in orch_result.visualizations],
        evidence=orch_result.evidence,
        tools_used=orch_result.tools_used,
        models_used=orch_result.models_used,
        analysis_trace=orch_result.analysis_trace,
        metadata=orch_result.metadata,
        warnings=orch_result.warnings,
    )


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    responses={
        400: {"description": "Invalid request or corrupted image"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Validation error"},
        500: {"description": "Internal processing error"},
        503: {"description": "Model provider unavailable"},
    },
)
async def analyze_image(
    image: Optional[UploadFile] = File(None, description="Primary satellite raster (alias 1)"),
    image1: Optional[UploadFile] = File(None, description="Primary satellite raster (alias 2)"),
    image2: Optional[UploadFile] = File(None, description="Secondary raster for change detection"),
    dates: Optional[str] = Form(None, description="Optional comma-separated or JSON list of dates"),
    query: str = Form(..., description="Natural language query or question"),
):
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    clean_query = query.strip() if query else ""

    # 1. Validate query
    if not clean_query:
        logger.warning(f"Empty query received (Request ID: {req_id})")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or whitespace.",
        )

    # 2. Select primary image
    primary_file = image if _is_valid_upload(image) else (image1 if _is_valid_upload(image1) else None)
    if not primary_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Primary image is required. Please upload an image file.",
        )

    second_file = image2 if _is_valid_upload(image2) else None

    # 3. Route the query (Phase-2 router)
    task_type, rationale = route_query(clean_query)
    has_second = second_file is not None

    # Require two images for change and multitemporal tasks
    if task_type in (TaskType.CHANGE_DETECTION, TaskType.CHANGED_OBJECTS, TaskType.MULTITEMPORAL_ANALYSIS) and not has_second:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Change and multitemporal analysis require two images. Please upload both Image 1 and Image 2.",
        )

    # Parse explicit dates if provided
    parsed_dates = None
    if isinstance(dates, str) and dates.strip():
        try:
            import json
            parsed_dates = json.loads(dates) if dates.strip().startswith("[") else [d.strip() for d in dates.split(",")]
        except Exception:
            parsed_dates = [d.strip() for d in dates.split(",")]

    try:
        # 4. Process uploads
        raw_bytes1, vlm_bytes1, vlm_mime1, metadata1 = await _process_upload(primary_file)
        raw_bytes2 = vlm_bytes2 = None
        meta2 = None
        if second_file is not None:
            raw_bytes2, vlm_bytes2, _, meta2 = await _process_upload(second_file)

        # 5. Dispatch: Orchestrator (Phase-2) or Planner (Phase-1 fallback)
        if task_type in _ORCHESTRATOR_TASKS:
            from app.core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            orch_result = await asyncio.to_thread(
                orch.analyze,
                image1_bytes=raw_bytes1,
                query=clean_query,
                image2_bytes=raw_bytes2,
                metadata=metadata1.model_dump(),
                request_id=req_id,
                explicit_dates=parsed_dates,
            )
            return _orchestrator_result_to_analysis(orch_result)

        # Phase-1 fallback: VQA / image_understanding
        result = await asyncio.to_thread(
            planner.plan_and_execute,
            query=clean_query,
            image_bytes=vlm_bytes1,
            mime_type=vlm_mime1,
            metadata=metadata1,
            request_id=req_id,
            second_image_bytes=vlm_bytes2,
            second_metadata=meta2,
        )
        return result


    except ImageProcessingError as exc:
        logger.warning(f"Image validation failed: {exc.message} (Request ID: {req_id})")
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    except ModelProviderUnavailableError as exc:
        logger.error(f"Model provider error: {exc.message} (Request ID: {req_id})")
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(f"Unexpected processing error (Request ID: {req_id}): {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during satellite image analysis (Request ID: {req_id}).",
        )


@router.post("/detect", response_model=AnalysisResult)
async def detect_objects(
    image: Optional[UploadFile] = File(None),
    image1: Optional[UploadFile] = File(None),
    query: Optional[str] = Form("Detect all objects and vehicles in this satellite scene."),
):
    """Detect objects using YOLO (generic) or YOLOv8n-OBB (aerial)."""
    return await analyze_image(image=image, image1=image1, query=query or "Detect all vehicles and objects.")


@router.post("/segment", response_model=AnalysisResult)
async def segment_imagery(
    image: Optional[UploadFile] = File(None),
    image1: Optional[UploadFile] = File(None),
    query: Optional[str] = Form("Find and segment all buildings and infrastructure zones."),
):
    """Segment buildings and structures using SAM 2.1."""
    return await analyze_image(image=image, image1=image1, query=query or "Find and segment all buildings.")


@router.post("/change-detection", response_model=AnalysisResult)
async def detect_changes(
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    query: Optional[str] = Form("What changed between these images?"),
):
    """ORB+RANSAC registration → ChangeFormer (or image-diff fallback) → statistics."""
    primary = image1 or image
    return await analyze_image(image=primary, image2=image2, query=query or "What changed between these images?")


@router.post("/changed-objects", response_model=AnalysisResult)
async def detect_changed_objects(
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    query: Optional[str] = Form("Which buildings have changed?"),
):
    """Full changed-objects pipeline: registration → change mask → building overlap."""
    primary = image1 or image
    return await analyze_image(image=primary, image2=image2, query=query or "Which buildings have changed?")


@router.post("/multitemporal-analysis", response_model=AnalysisResult)
async def multitemporal_analysis(
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    dates: Optional[str] = Form(None),
    query: Optional[str] = Form("What parameters changed between these images?"),
):
    """Full multitemporal parameter analysis: urban, roads, vegetation, water, and transition matrix."""
    primary = image1 or image
    return await analyze_image(
        image=primary,
        image2=image2,
        dates=dates,
        query=query or "What parameters changed between these images?",
    )

