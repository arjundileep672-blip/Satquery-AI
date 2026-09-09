"""
Tests for EuroSAT Performance Metrics & Confusion Matrix endpoint
"""

from fastapi.testclient import TestClient
from app.main import app
from app.api.v1.endpoints.metrics import get_eurosat_metrics

client = TestClient(app)


def test_eurosat_metrics_function():
    """Verify get_eurosat_metrics direct python function returns expected structure."""
    data = get_eurosat_metrics()
    assert "overall_accuracy" in data
    assert data["overall_accuracy"] > 0
    assert "top_3_accuracy" in data
    assert "confusion_matrix" in data
    assert len(data["confusion_matrix"]) == 10
    assert len(data["confusion_matrix"][0]) == 10
    assert "normalized_confusion_matrix" in data
    assert len(data["normalized_confusion_matrix"]) == 10
    assert "class_names" in data
    assert len(data["class_names"]) == 10
    assert "per_class_metrics" in data
    assert "AnnualCrop" in data["per_class_metrics"]
    ac_metric = data["per_class_metrics"]["AnnualCrop"]
    assert "precision" in ac_metric
    assert "recall" in ac_metric
    assert "f1" in ac_metric
    assert "support" in ac_metric


def test_eurosat_metrics_endpoint():
    """Verify GET /api/v1/metrics/eurosat HTTP endpoint returns HTTP 200."""
    response = client.get("/api/v1/metrics/eurosat")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["model"] == "EuroSAT EfficientNet-B0"
    assert json_data["overall_accuracy"] >= 70.0
    assert len(json_data["confusion_matrix"]) == 10


def test_fleet_metrics_endpoint():
    """Verify GET /api/v1/metrics/models returns a score for every registry model."""
    response = client.get("/api/v1/metrics/models")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    names = {row["id"] for row in payload["models"]}
    assert "eurosat_efficientnet_b0" in names
    assert "yolo12n" in names
    assert "yolo26n_obb" in names
    for row in payload["models"]:
        assert row["score"] > 0
        assert row["metric"]
