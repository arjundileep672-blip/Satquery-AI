"""
SatQuery AI - Updated Health Check Endpoint

Returns full model status, device, and offline mode flag.
"""

from fastapi import APIRouter
from app.core.config import settings, resolve_device

router = APIRouter()


def _model_status_table() -> dict:
    """Build a per-model status dict from the registry."""
    try:
        import sys, os
        # Add backend to path so model_registry can import app.*
        backend_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        ))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        from core.model_registry import refresh_registry
        reg = refresh_registry()
        return {
            name: entry["load_status"]
            for name, entry in reg.items()
        }
    except Exception as exc:
        return {"error": f"Registry unavailable: {exc}"}


@router.get("/health")
def health_check():
    """
    Returns application health, model availability, device, and offline status.

    Response contract:
    {
      "status": "ok" | "degraded",
      "service": str,
      "version": str,
      "offline_mode": true,
      "device": "cuda" | "cpu",
      "demo_mode": bool,
      "models": {
        "yolo12n":     "ready" | "fallback" | "missing",
        "yolo26n_obb": "...",
        "spacenet":    "...",
        "sam2":        "...",
        "changeformer":"...",
        "vlm":         "...",
      }
    }
    """
    device = resolve_device()
    models = _model_status_table()

    # Compute overall status
    critical_models = ["yolo12n", "yolo26n_obb", "sam2"]
    status = "healthy"
    for m in critical_models:
        ms = models.get(m, "missing")
        if ms not in ("ready", "fallback", "ready (mock)"):
            status = "degraded"
            break

    from app.tools.registry import registry
    return {
        "status": status,
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "offline_mode": True,
        "device": device,
        "gpu_available": device == "cuda",
        "models_status": "Ready" if status == "healthy" else "Degraded",
        "demo_mode": settings.DEMO_MODE,
        "models": models,
        "tools_registered": [t["name"] for t in registry.list_tools()],
    }
