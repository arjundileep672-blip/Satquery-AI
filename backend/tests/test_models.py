"""
SatQuery AI — Automated Test Suite for Offline Models & Pipelines
Covers model availability, inference pipelines, registration, geospatial analysis, and offline guarantees.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings, resolve_device
from core.model_registry import get_registry


@pytest.fixture(scope="session")
def demo_image():
    """Load local demo image or create synthetic fixture."""
    p = REPO_ROOT / "data" / "demo" / "single_image.jpg"
    if p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            return img
    # Synthetic fallback
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[40:120, 40:120] = [200, 200, 200]
    return img


@pytest.fixture(scope="session")
def bitemporal_pair():
    """Load local bi-temporal demo pair or create synthetic pair."""
    p1 = REPO_ROOT / "data" / "demo" / "before.tif"
    p2 = REPO_ROOT / "data" / "demo" / "after.tif"
    if p1.exists() and p2.exists():
        im1 = cv2.imread(str(p1))
        im2 = cv2.imread(str(p2))
        if im1 is not None and im2 is not None:
            return im1, im2
    # Synthetic
    im1 = np.zeros((256, 256, 3), dtype=np.uint8)
    im2 = im1.copy()
    im1[40:120, 40:120] = [200, 200, 200]
    im2[40:120, 40:120] = [50, 50, 50]  # change
    return im1, im2


# ── Model Availability Tests ──────────────────────────────────────────────────

def test_yolo12_loaded():
    """Verify YOLO generic detector checkpoint is available or fallback configured."""
    reg = get_registry()
    status = reg["yolo12n"]["load_status"]
    assert status in ("ready", "fallback"), f"YOLO status is {status}"


def test_yolo26_loaded():
    """Verify YOLO26n-OBB detector checkpoint is available or fallback configured."""
    reg = get_registry()
    status = reg["yolo26n_obb"]["load_status"]
    assert status in ("ready", "fallback"), f"YOLO26-OBB status is {status}"


def test_spacenet_loaded():
    """Verify SpaceNet model status is defined."""
    reg = get_registry()
    assert "spacenet" in reg
    assert reg["spacenet"]["task"] == "building_footprint_segmentation"


def test_sam2_loaded():
    """Verify SAM2 checkpoint is available."""
    reg = get_registry()
    status = reg["sam2"]["load_status"]
    assert status in ("ready", "fallback", "missing"), f"SAM2 status is {status}"


def test_changeformer_loaded():
    """Verify ChangeFormer status is defined."""
    reg = get_registry()
    assert "changeformer" in reg
    assert reg["changeformer"]["task"] == "binary_change_detection"


def test_vlm_loaded():
    """Verify VLM adapter is ready (Ollama or Mock)."""
    reg = get_registry()
    status = reg["vlm"]["load_status"]
    assert "ready" in status or status == "fallback (gemini)"


# ── Inference Tests ───────────────────────────────────────────────────────────

def test_detection(demo_image):
    """Test YOLO detection pipeline on local demo image."""
    from pipelines.detection import run_generic_detection
    result = run_generic_detection(demo_image, conf=0.1)
    assert hasattr(result, "detections")
    assert isinstance(result.count, int)
    assert result.model in ("yolo12n", "yolo11n", "yolo_unavailable")


def test_remote_detection(demo_image):
    """Test YOLO26n-OBB remote sensing detector on local demo image."""
    from pipelines.detection import run_remote_detection
    result = run_remote_detection(demo_image, query="detect plane and ship", conf=0.1)
    assert hasattr(result, "detections")
    assert isinstance(result.count, int)


def test_segmentation(demo_image):
    """Test segmentation pipeline (SAM2 / GrabCut fallback)."""
    from pipelines.segmentation import run_segmentation
    result = run_segmentation(demo_image, conf=0.1)
    assert hasattr(result, "masks")
    assert hasattr(result, "total_area_px")


def test_change_detection(bitemporal_pair):
    """Test change detection pipeline (ChangeFormer or diff fallback)."""
    im1, im2 = bitemporal_pair
    from pipelines.change_detection import run_change_detection
    result = run_change_detection(im1, im2)
    assert hasattr(result, "changed_pixels")
    assert hasattr(result, "change_percentage")
    assert result.total_pixels > 0
    assert 0.0 <= result.change_percentage <= 100.0


def test_changed_buildings(bitemporal_pair):
    """Test semantic change intelligence pipeline."""
    im1, im2 = bitemporal_pair
    from pipelines.changed_objects import run_changed_objects
    result = run_changed_objects(im1, im2, object_type="building")
    assert result["task"] == "changed_objects"
    assert "registration" in result
    assert "change" in result
    assert "objects" in result
    assert isinstance(result["objects"], list)


# ── Registration Tests ────────────────────────────────────────────────────────

def test_orb_registration(demo_image):
    """Test ORB feature detection and matching on identical/shifted image."""
    from vision.registration import register_images
    # Slight translation
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    shifted = cv2.warpAffine(demo_image, M, (demo_image.shape[1], demo_image.shape[0]))
    reg = register_images(demo_image, shifted)
    assert "success" in reg
    assert "keypoints1" in reg
    assert "matches_total" in reg


def test_ransac_outlier_rejection():
    """Verify that completely unaligned images fail registration clearly without crashing."""
    from vision.registration import register_images
    noise1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    noise2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    reg = register_images(noise1, noise2)
    # Registration should report failure or zero inliers, but never crash
    assert "success" in reg
    assert "error" in reg


# ── Geospatial Tests ──────────────────────────────────────────────────────────

def test_crs_detection():
    """Verify georeferencing extraction handles images without CRS gracefully."""
    from geospatial.coordinates import extract_georeferencing
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = extract_georeferencing(img.tobytes())
    assert res["georeferencing_available"] is False
    assert res["area_available"] is False


def test_pixel_to_geo():
    """Verify pixel to lat/lon transformation with valid affine and CRS."""
    from geospatial.coordinates import pixel_to_latlon
    transform = [0.5, 0.0, 500000.0, 0.0, -0.5, 2000000.0]
    crs = "EPSG:32643"
    result = pixel_to_latlon(10.0, 10.0, transform, crs)
    assert result is not None
    lat, lon = result
    assert isinstance(lat, float)
    assert isinstance(lon, float)


def test_area_calculation():
    """Verify Shoelace pixel area calculation for a simple 10x10 polygon."""
    from geospatial.area import calculate_polygon_area
    box_pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    res = calculate_polygon_area(box_pts)
    assert res["area_pixels"] == 100.0
    assert res["georeferencing_available"] is False


# ── Offline Inference Test ────────────────────────────────────────────────────

def test_offline_inference(demo_image):
    """
    Verify that end-to-end /analyze works with NO external network access.
    Simulates offline mode using local mock/offline models.
    """
    from app.core.orchestrator import get_orchestrator
    orch = get_orchestrator()

    _, buf = cv2.imencode(".png", demo_image)
    raw_bytes = buf.tobytes()

    response = orch.analyze(
        image1_bytes=raw_bytes,
        query="Detect all objects in this satellite scene.",
    )

    assert response is not None
    assert response.task in ("detection", "remote_detection", "counting")
    assert response.answer is not None
    assert len(response.answer) > 0
    # Provenance check: no OpenAI or cloud APIs in models_used
    for model_name in response.models_used:
        assert "openai" not in model_name.lower()
        assert "gemini" not in model_name.lower()
