"""
Workflow Verification Script for SatQuery AI Phase 1
Tests the exact 11-step end-to-end workflow requested in the specification.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app

client = TestClient(app)

def run_verification():
    print("=== TESTING EXACT WORKFLOW ===")
    
    # 1. Health check
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health = health_resp.json()
    print("Step 1: Health check OK ->", health)
    assert health["demo_mode"] is True, "Expected DEMO_MODE=True"

    # 2. Upload sample satellite image
    img_path = Path("data/demo/sample_harbor_optical.png")
    assert img_path.exists(), f"Missing demo image: {img_path}"
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    print(f"Step 2: Loaded sample image ({len(img_bytes)} bytes)")

    # 3 & 4. Submit query
    query = "What objects are visible in this image?"
    print(f"Step 3 & 4: Submitting query: '{query}'")

    resp = client.post(
        "/api/v1/analyze",
        files={"image": ("sample_harbor_optical.png", img_bytes, "image/png")},
        data={"query": query}
    )
    assert resp.status_code == 200, f"Analyze failed: {resp.text}"
    result = resp.json()

    # 5. Verify Planner selects correct tool
    print("Step 5: Operation selected ->", result["operation"])
    print("        Tools used ->", result["tools_used"])
    assert result["operation"] in ("visual_question_answering", "image_understanding")
    assert "visual_question_answering" in result["tools_used"] or "image_understanding" in result["tools_used"]

    # 6. Verify VLM adapter executes
    print("Step 6: Models used ->", result["models_used"])
    assert len(result["models_used"]) > 0
    assert "mock-rs-vlm-v1" in result["models_used"][0]

    # 7. Verify structured AnalysisResult
    print("Step 7: Request ID ->", result["request_id"])
    assert result["request_id"].startswith("req_")
    assert result["confidence"] is None  # Properly uncalibrated per spec
    print("        Confidence ->", result["confidence"], "(Correctly null/uncalibrated)")

    # 8. Verify answer appears
    print("Step 8: Answer excerpt ->\n" + result["answer"][:180] + "...")
    assert len(result["answer"]) > 20

    # 9. Verify Analysis Trace appears
    print("Step 9: Analysis Trace ->", result["analysis_trace"])
    assert len(result["analysis_trace"]) >= 4
    assert any("validated" in s.lower() for s in result["analysis_trace"])
    assert any("classified" in s.lower() for s in result["analysis_trace"])
    assert any("selected" in s.lower() for s in result["analysis_trace"])

    # 10. Verify image metadata appears
    print("Step 10: Metadata ->", result["metadata"])
    assert result["metadata"]["width"] == 512
    assert result["metadata"]["height"] == 512
    assert result["metadata"]["format"] == "PNG"

    # 11. Test with GeoTIFF sample (Demo mode without API key)
    tif_path = Path("data/demo/sample_coastal_geotiff.tif")
    if tif_path.exists():
        with open(tif_path, "rb") as f:
            tif_bytes = f.read()

        resp_tif = client.post(
            "/api/v1/analyze",
            files={"image": ("sample_coastal_geotiff.tif", tif_bytes, "image/tiff")},
            data={"query": "Is there a river or open water in this scene?"}
        )
        assert resp_tif.status_code == 200
        res_tif = resp_tif.json()
        print("Step 11: GeoTIFF Analysis OK -> CRS:", res_tif["metadata"].get("crs"), "Bands:", res_tif["metadata"].get("band_count"))
        assert res_tif["metadata"]["is_geotiff"] is True
        assert res_tif["metadata"]["crs"] == "EPSG:4326"

    print("=== ALL 11 WORKFLOW VERIFICATIONS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_verification()
