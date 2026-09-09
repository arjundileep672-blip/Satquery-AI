"""
SatQuery AI — Comprehensive Tests for Vision, ML Models & Orchestration (Phase 2)
Tests all vision pipelines, models, query routing, registration, and endpoints.
"""

import io
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.core.query_router import TaskType, route_query
from app.core.orchestrator import Orchestrator, get_orchestrator
from app.schemas.vision import BoundingBox, DetectionResult
from vision.registration import register_images
from pipelines.change_detection import run_change_detection
from models.yolo_detector import detect as yolo_detect
from models.remote_detector import detect_remote, DOTA_V1_CLASSES
from models.building_detector import detect_buildings
from models.sam2_segmenter import segment_with_boxes
from models.changeformer import detect_changes
from models.vlm import get_vlm_bridge

client = TestClient(app)


def create_synthetic_image(width: int = 256, height: int = 256, color=(120, 140, 160)) -> np.ndarray:
    """Helper to create a numpy RGB image."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Add some geometric shapes so features can be detected
    img[30:70, 30:80] = [220, 50, 50]       # Red rectangle
    img[100:150, 120:180] = [50, 220, 50]   # Green rectangle
    img[180:220, 60:110] = [50, 50, 220]    # Blue rectangle
    return img


def create_png_bytes(width: int = 256, height: int = 256, color=(120, 140, 160)) -> bytes:
    """Helper to create PNG bytes from synthetic image."""
    arr = create_synthetic_image(width, height, color)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ── Test 1: Query Router ──────────────────────────────────────────────────────
def test_01_query_router_classification():
    q_bld, _ = route_query("Find all buildings")
    assert q_bld == TaskType.BUILDING_DETECTION

    q_veh, _ = route_query("Detect all vehicles")
    assert q_bld in (TaskType.BUILDING_DETECTION, TaskType.DETECTION)

    q_air, _ = route_query("Find all aircraft")
    assert q_air == TaskType.REMOTE_DETECTION

    q_seg, _ = route_query("Segment the buildings")
    assert q_seg in (TaskType.SEGMENTATION, TaskType.BUILDING_DETECTION)

    q_cnt, _ = route_query("How many vehicles are present?")
    assert q_cnt == TaskType.COUNTING

    q_chg, _ = route_query("What changed between these two images?")
    assert q_chg == TaskType.CHANGE_DETECTION

    q_chg_bld, _ = route_query("Which buildings have changed?")
    assert q_chg_bld == TaskType.CHANGED_OBJECTS

    q_pct, _ = route_query("What percentage changed?")
    assert q_pct == TaskType.CHANGE_DETECTION


# ── Test 2: ORB + RANSAC Registration ─────────────────────────────────────────
def test_02_orb_ransac_registration():
    img1 = create_synthetic_image(300, 300)
    # Slightly translated copy
    img2 = np.roll(img1, shift=5, axis=1)

    reg = register_images(img1, img2)
    assert "aligned_image" in reg
    assert "keypoints1" in reg
    assert "keypoints2" in reg
    assert "processing_ms" in reg
    assert reg["processing_ms"] >= 0


# ── Test 3: ChangeFormer / Image-Diff Fallback ────────────────────────────────
def test_03_change_detection_module():
    img1 = create_synthetic_image(256, 256)
    img2 = img1.copy()
    # Introduce an obvious changed block
    img2[80:140, 80:140] = [255, 255, 255]

    res = detect_changes(img1, img2)
    assert "change_mask" in res
    assert res["change_mask"].shape == (256, 256)
    assert "model" in res
    assert res["change_mask"].sum() > 0


# ── Test 4: Change Detection Full Pipeline ────────────────────────────────────
def test_04_change_detection_pipeline():
    img1 = create_synthetic_image(256, 256)
    img2 = img1.copy()
    img2[50:120, 50:120] = [240, 240, 240]

    cd = run_change_detection(img1, img2, skip_registration=True)
    assert cd.total_pixels == 256 * 256
    assert cd.changed_pixels > 0
    assert cd.change_percentage > 0.0
    assert isinstance(cd.change_regions, list)
    assert len(cd.change_regions) > 0


# ── Test 5: YOLO Generic Detector ─────────────────────────────────────────────
def test_05_yolo_detector():
    img = create_synthetic_image(256, 256)
    result = yolo_detect(img, conf=0.1)
    assert isinstance(result, DetectionResult)
    assert "COCO" in result.dataset
    assert result.metadata["processing_ms"] >= 0


# ── Test 6: Remote Sensing Detector (DOTA-v1) ─────────────────────────────────
def test_06_remote_detector():
    img = create_synthetic_image(256, 256)
    result = detect_remote(img, conf=0.1)
    assert isinstance(result, DetectionResult)
    assert result.model in ("yolo26n-obb", "yolov8n-obb")
    assert result.dataset == "DOTA-v1"
    assert "plane" in DOTA_V1_CLASSES
    assert "ship" in DOTA_V1_CLASSES


# ── Test 7: Building Detector ─────────────────────────────────────────────────
def test_07_building_detector():
    img = create_synthetic_image(256, 256)
    result = detect_buildings(img)
    assert "buildings" in result
    assert "masks" in result
    assert "count" in result
    assert "metadata" in result


# ── Test 8: SAM2 Segmentation / GrabCut Fallback ──────────────────────────────
def test_08_sam2_segmentation():
    img = create_synthetic_image(256, 256)
    boxes = [BoundingBox(x1=30, y1=30, x2=80, y2=70)]
    result = segment_with_boxes(img, boxes, labels=["building"])
    assert result.count == 1
    assert len(result.masks) == 1
    assert result.masks[0].area_px > 0
    assert len(result.masks[0].polygon) >= 3


# ── Test 9: VLM Bridge Synthesis ──────────────────────────────────────────────
def test_09_vlm_bridge():
    bridge = get_vlm_bridge()
    answer = bridge.synthesize_answer(
        query="How many vehicles are present?",
        task="counting",
        results_summary="Detected 4 vehicles across the road intersection.",
    )
    assert len(answer) > 0


# ── Test 10: Task Orchestrator Full Flow ──────────────────────────────────────
def test_10_orchestrator():
    orchestrator = get_orchestrator()
    png_bytes = create_png_bytes(256, 256)
    png_bytes2 = create_png_bytes(256, 256, color=(140, 160, 180))

    # Single-image query
    resp1 = orchestrator.analyze(
        image1_bytes=png_bytes,
        query="Detect all vehicles",
        metadata={"width": 256, "height": 256, "format": "PNG"},
    )
    assert resp1.request_id.startswith("req_")
    assert len(resp1.answer) > 0
    assert resp1.task in ("detection", "remote_detection", "counting", "building_detection")

    # Dual-image change query
    resp2 = orchestrator.analyze(
        image1_bytes=png_bytes,
        image2_bytes=png_bytes2,
        query="What changed between these two images?",
        metadata={"width": 256, "height": 256, "format": "PNG"},
    )
    assert resp2.task == "change_detection"
    assert len(resp2.answer) > 0


# ── Test 11: End-to-End API Endpoints ─────────────────────────────────────────
def test_11_endpoints_e2e():
    png_bytes1 = create_png_bytes(256, 256)
    png_bytes2 = create_png_bytes(256, 256, color=(140, 150, 160))

    # /detect
    r_det = client.post(
        "/api/v1/detect",
        files={"image": ("img.png", png_bytes1, "image/png")},
        data={"query": "Detect all vehicles"},
    )
    assert r_det.status_code == 200
    assert "detections" in r_det.json()

    # /segment
    r_seg = client.post(
        "/api/v1/segment",
        files={"image": ("img.png", png_bytes1, "image/png")},
        data={"query": "Find and segment all buildings"},
    )
    assert r_seg.status_code == 200
    assert "masks" in r_seg.json()

    # /change-detection
    r_cd = client.post(
        "/api/v1/change-detection",
        files={
            "image1": ("img1.png", png_bytes1, "image/png"),
            "image2": ("img2.png", png_bytes2, "image/png"),
        },
        data={"query": "What changed between these images?"},
    )
    assert r_cd.status_code == 200
    assert "changes" in r_cd.json()

    # /changed-objects
    r_co = client.post(
        "/api/v1/changed-objects",
        files={
            "image1": ("img1.png", png_bytes1, "image/png"),
            "image2": ("img2.png", png_bytes2, "image/png"),
        },
        data={"query": "Which buildings have changed?"},
    )
    assert r_co.status_code == 200
    assert "changes" in r_co.json()
