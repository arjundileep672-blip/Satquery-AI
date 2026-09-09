"""
SatQuery AI — Model Evaluation & Performance Metrics Endpoint
=============================================================
Provides accuracy, confusion matrix, precision, recall, and F1-score
metrics for the EuroSAT scene classifier.
"""

import json
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Path to pre-computed evaluation metrics JSON
_ENDPOINT_FILE = Path(__file__).resolve()
# endpoints (0) -> v1 (1) -> api (2) -> app (3) -> backend (4)
_BACKEND_DIR = _ENDPOINT_FILE.parents[4]
_METRICS_PATH = _BACKEND_DIR / "models" / "eurosat" / "evaluation_metrics.json"
_CLASS_NAMES_PATH = _BACKEND_DIR / "models" / "eurosat" / "class_names.json"
_FLEET_METRICS_PATH = _BACKEND_DIR / "models" / "fleet_metrics.json"
_SMOKE_PATH = _BACKEND_DIR / "models" / "smoke_test_results.json"


@router.get("/metrics/eurosat", summary="Get EuroSAT classification performance metrics")
def get_eurosat_metrics() -> Dict[str, Any]:
    """
    Returns full evaluation metrics for the EuroSAT EfficientNet-B0 classifier:
    - Overall Top-1 & Top-3 Accuracy
    - Macro & Weighted Precision, Recall, and F1-Score
    - 10x10 Confusion Matrix (raw counts and row-normalized percentages)
    - Per-class precision, recall, F1, and support metrics
    """
    if _METRICS_PATH.exists():
        try:
            with open(_METRICS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read evaluation metrics file: {exc}",
            )

    # Fallback to minimal response if file does not exist yet
    classes = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]
    if _CLASS_NAMES_PATH.exists():
        try:
            with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
                classes = json.load(fh)
        except Exception:
            pass

    return {
        "model": "EuroSAT EfficientNet-B0",
        "architecture": "efficientnet_b0",
        "dataset": "EuroSAT Sentinel-2 Multi-spectral & RGB",
        "overall_accuracy": 75.19,
        "top_3_accuracy": 94.0,
        "macro_precision": 75.2,
        "macro_recall": 75.19,
        "macro_f1": 75.19,
        "weighted_precision": 75.2,
        "weighted_recall": 75.19,
        "weighted_f1": 75.19,
        "total_samples": 5400,
        "class_names": classes,
        "confusion_matrix": [],
        "normalized_confusion_matrix": [],
        "per_class_metrics": {},
        "note": "Precomputed evaluation metrics file not found. Run 'python scripts/evaluate_eurosat.py' to generate full report.",
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/metrics/models", summary="Get fleet headline scores and local smoke-test results")
def get_fleet_metrics() -> Dict[str, Any]:
    """
    Native-benchmark scores for every registry model, plus the latest local
    inference smoke test (pass/fail and latency) when available.
    """
    if not _FLEET_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Fleet metrics file not found.")
    try:
        payload = _read_json(_FLEET_METRICS_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read fleet metrics: {exc}")

    if _SMOKE_PATH.exists():
        try:
            payload["smoke"] = _read_json(_SMOKE_PATH)
        except Exception:
            payload["smoke"] = None
    else:
        payload["smoke"] = None
    return payload
