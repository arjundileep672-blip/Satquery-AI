"""
Comprehensive Automated Tests for SatQuery AI Phase 1
Tests all 14 required specifications without requiring a live Gemini API key.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.agents.planner import SatQueryPlanner
from app.core.config import settings
from app.models.mock_adapter import MockVLMAdapter
from app.schemas.analysis import AnalysisResult
from app.schemas.evidence import Evidence
from app.schemas.metadata import ImageMetadata
from app.services.image_service import (
    ImageProcessingError,
    extract_metadata,
    sanitize_filename,
    validate_file,
)
from app.tools.image_understanding import ImageUnderstandingTool
from app.tools.registry import ToolRegistry, ToolNotFoundError
from app.tools.visual_question_answering import VisualQuestionAnsweringTool

client = TestClient(app)


def create_test_png(width: int = 100, height: int = 100, color=(100, 150, 200)) -> bytes:
    """Helper to create valid in-memory PNG bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


# 1. Health Endpoint Test
def test_01_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "ok")
    assert data["service"] == "SatQuery AI"
    assert "tools_registered" in data
    assert "image_understanding" in data["tools_registered"]
    assert "visual_question_answering" in data["tools_registered"]


# 2. Valid Image Upload & Metadata Extraction
def test_02_valid_image_metadata():
    png_bytes = create_test_png(200, 150)
    meta = extract_metadata(png_bytes, "test_raster.png")
    assert meta.width == 200
    assert meta.height == 150
    assert meta.channels == 3
    assert meta.format == "PNG"
    assert meta.is_geotiff is False

    # Test with GeoTIFF sample
    demo_tif = Path("data/demo/sample_coastal_geotiff.tif")
    if demo_tif.exists():
        with open(demo_tif, "rb") as f:
            tif_bytes = f.read()
        geo_meta = extract_metadata(tif_bytes, "sample_coastal_geotiff.tif")
        assert geo_meta.is_geotiff is True
        assert geo_meta.crs == "EPSG:4326"
        assert geo_meta.band_count == 4
        assert geo_meta.bounds is not None


# 3. Invalid Image Upload (Corrupt Bytes)
def test_03_invalid_image_upload():
    corrupt_bytes = b"NOT_A_VALID_IMAGE_FILE_RANDOM_GARBAGE"
    response = client.post(
        "/api/v1/analyze",
        files={"image": ("corrupt.png", corrupt_bytes, "image/png")},
        data={"query": "What is in this image?"},
    )
    assert response.status_code == 400
    assert "Corrupt or invalid image file" in response.json()["detail"]


# 4. Unsupported File Type
def test_04_unsupported_file_type():
    txt_bytes = b"Hello world, this is a plain text file."
    response = client.post(
        "/api/v1/analyze",
        files={"image": ("document.txt", txt_bytes, "text/plain")},
        data={"query": "What objects are visible?"},
    )
    assert response.status_code == 415
    assert "Unsupported file extension" in response.json()["detail"]


# 5. Oversized File (>25MB)
def test_05_oversized_file():
    oversized_size = settings.MAX_FILE_SIZE_BYTES + 1024
    with pytest.raises(ImageProcessingError) as exc_info:
        validate_file("huge_satellite_scene.tif", "image/tiff", oversized_size)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum allowed limit" in exc_info.value.message


# 6. Empty Query
def test_06_empty_query():
    png_bytes = create_test_png()
    response = client.post(
        "/api/v1/analyze",
        files={"image": ("scene.png", png_bytes, "image/png")},
        data={"query": "   "},
    )
    assert response.status_code == 400
    assert "cannot be empty or whitespace" in response.json()["detail"]


# 7. Planner Intent Classification
def test_07_planner_classification():
    planner = SatQueryPlanner()

    op1, _ = planner.classify_intent("Provide a general overview and scene summary of this terrain.")
    assert op1 == "image_understanding"

    op2, _ = planner.classify_intent("Describe the land cover composition and scene.")
    assert op2 == "image_understanding"

    op3, _ = planner.classify_intent("Is there a river or water channel in the western part?")
    assert op3 == "visual_question_answering"

    op4, _ = planner.classify_intent("How many cargo ships are berthed at the pier?")
    assert op4 == "visual_question_answering"


# 8. Controlled Tool Registry
def test_08_tool_registry():
    custom_reg = ToolRegistry()
    tool = VisualQuestionAnsweringTool(MockVLMAdapter())
    custom_reg.register(tool)

    assert custom_reg.has_tool("visual_question_answering") is True
    assert custom_reg.get("visual_question_answering").name == "visual_question_answering"

    # Verify exception on missing tool (no arbitrary execution)
    with pytest.raises(ToolNotFoundError):
        custom_reg.get("arbitrary_python_executor")


# 9. VQA Tool Execution
def test_09_vqa_tool_execution():
    tool = VisualQuestionAnsweringTool(MockVLMAdapter())
    png_bytes = create_test_png()
    out = tool.execute(
        image_bytes=png_bytes,
        mime_type="image/png",
        query="Are there ships visible?",
        metadata={"width": 100, "height": 100},
    )
    assert out.status == "success"
    assert out.tool_name == "visual_question_answering"
    assert len(out.answer) > 0
    assert out.confidence is None  # Qualitative / uncalibrated per spec


# 10. Image Understanding Tool Execution
def test_10_image_understanding_execution():
    tool = ImageUnderstandingTool(MockVLMAdapter())
    png_bytes = create_test_png()
    out = tool.execute(
        image_bytes=png_bytes,
        mime_type="image/png",
        query="Describe the scene.",
        metadata={"width": 100, "height": 100},
    )
    assert out.status == "success"
    assert out.tool_name == "image_understanding"
    assert len(out.answer) > 0
    assert out.confidence is None


# 11. Mock VLM Deterministic Output
def test_11_mock_vlm_adapter():
    vlm = MockVLMAdapter()
    resp = vlm.generate_response(
        image_bytes=b"dummy",
        mime_type="image/png",
        prompt="What objects are visible in this satellite scene?",
        metadata={"width": 512, "height": 512, "is_geotiff": False},
    )
    assert resp.model_name == "mock-rs-vlm-v1"
    assert "Maritime Vessels" in resp.text or "commercial ships" in resp.text
    assert resp.confidence is None


# 12. Evidence Schema
def test_12_evidence_schema():
    ev = Evidence(
        evidence_id="ev_test_1",
        source_asset="harbor.png",
        observation="Water body detected in west quadrant",
        spatial_extent=None,
        confidence=None,
        model_tool_used="visual_question_answering:mock-rs-vlm-v1",
        processing_steps=["Ingest", "Inspect"],
        measurements={"width_px": 512, "height_px": 512},
        geometry=None,
    )
    assert ev.confidence is None
    assert ev.spatial_extent is None
    assert ev.geometry is None
    assert ev.measurements["width_px"] == 512


# 13. AnalysisResult Schema
def test_13_analysis_result_schema():
    res = AnalysisResult(
        request_id="req_123",
        answer="Analysis indicates active port.",
        operation="image_understanding",
        confidence=None,
        evidence=[],
        tools_used=["image_understanding"],
        models_used=["mock-rs-vlm-v1"],
        analysis_trace=["Image validated", "Tool selected"],
        metadata={"format": "PNG"},
        warnings=["Model confidence is qualitative/un-calibrated."],
    )
    assert res.request_id == "req_123"
    assert res.confidence is None
    assert len(res.analysis_trace) == 2


# 14. Full End-to-End API Response Test
def test_14_api_analyze_e2e():
    png_bytes = create_test_png(256, 256)
    response = client.post(
        "/api/v1/analyze",
        files={"image": ("demo_port.png", png_bytes, "image/png")},
        data={"query": "What objects are visible in this image?"},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify structured fields
    assert "request_id" in data
    assert "answer" in data
    assert data["operation"] in ("image_understanding", "visual_question_answering")
    assert data["confidence"] is None  # Uncalibrated / qualitative
    assert "evidence" in data
    assert len(data["evidence"]) > 0
    assert "tools_used" in data
    assert "models_used" in data
    assert "analysis_trace" in data
    assert len(data["analysis_trace"]) >= 4
    assert "metadata" in data
    assert data["metadata"]["width"] == 256
    assert data["metadata"]["height"] == 256
    assert "warnings" in data
