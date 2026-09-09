"""
SatQuery AI - Central Model Registry

Provides MODEL_REGISTRY: a dict mapping model names to metadata and load status.
Used by /health endpoint and check_models.py.

Every model exposes:
  - name: str
  - task: str
  - checkpoint: str (relative to repo root)
  - fallback_checkpoint: str | None
  - device: str
  - load_status: str ("ready" | "missing" | "unavailable" | "fallback")
  - source: str
  - license: str
  - dataset: str | None
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings, load_models_config

# Repo root = one level above backend/
_REPO_ROOT: Path = settings.BASE_DIR.parent


def _ckpt_status(rel_path: str, fallback_rel: Optional[str] = None) -> str:
    """
    Returns "ready" if primary checkpoint exists,
    "fallback" if only fallback exists,
    "missing" otherwise.
    """
    if rel_path and (_REPO_ROOT / rel_path).exists():
        return "ready"
    if fallback_rel and (_REPO_ROOT / fallback_rel).exists():
        return "fallback"
    return "missing"


def _ckpt_size_mb(rel_path: str) -> Optional[float]:
    """File size in MB, or None if not found."""
    p = _REPO_ROOT / rel_path
    if p.exists():
        return round(p.stat().st_size / (1024 * 1024), 2)
    return None


def build_registry() -> Dict[str, Dict[str, Any]]:
    """Build the model registry from models.yaml + filesystem state."""
    cfg = load_models_config()

    yolo_cfg   = cfg.get("yolo", {})
    remote_cfg = cfg.get("remote_detector", {})
    sn_cfg     = cfg.get("building_detector", {})
    sam2_cfg   = cfg.get("sam2", {})
    cf_cfg     = cfg.get("changeformer", {})
    vlm_cfg    = cfg.get("vlm", {})

    yolo_ckpt    = yolo_cfg.get("checkpoint", "models/yolo/yolo12n.pt")
    yolo_fb      = yolo_cfg.get("fallback_checkpoint", "checkpoints/yolo11n.pt")
    remote_ckpt  = remote_cfg.get("checkpoint", "models/yolo26_obb/yolo26n-obb.pt")
    remote_fb    = remote_cfg.get("fallback_checkpoint", "checkpoints/yolov8n-obb.pt")
    sn_ckpt      = sn_cfg.get("checkpoint", "models/spacenet/spacenet_rio.pt")
    sam2_ckpt    = sam2_cfg.get("checkpoint", "models/sam2/sam2.1_hiera_tiny.pt")
    cf_ckpt      = cf_cfg.get("checkpoint", "models/changeformer/changeformer_levir.pt")

    # VLM: availability is checked via Ollama ping, not file
    vlm_status = _probe_vlm(vlm_cfg)

    registry = {
        "yolo12n": {
            "name": "YOLO12n",
            "task": "generic_object_detection",
            "checkpoint": yolo_ckpt,
            "fallback_checkpoint": yolo_fb,
            "load_status": _ckpt_status(yolo_ckpt, yolo_fb),
            "size_mb": _ckpt_size_mb(yolo_ckpt) or _ckpt_size_mb(yolo_fb),
            "device": yolo_cfg.get("device", "auto"),
            "dataset": "COCO",
            "source": "https://github.com/ultralytics/ultralytics",
            "license": "AGPL-3.0",
            "version": "YOLO12n",
        },
        "yolo26n_obb": {
            "name": "YOLO26n-OBB (DOTA-v1)",
            "task": "oriented_object_detection",
            "checkpoint": remote_ckpt,
            "fallback_checkpoint": remote_fb,
            "load_status": _ckpt_status(remote_ckpt, remote_fb),
            "size_mb": _ckpt_size_mb(remote_ckpt) or _ckpt_size_mb(remote_fb),
            "device": remote_cfg.get("device", "auto"),
            "dataset": "DOTA-v1",
            "source": "https://github.com/ultralytics/ultralytics",
            "license": "AGPL-3.0",
            "version": "YOLOv8/26n-OBB",
        },
        "spacenet": {
            "name": "SpaceNet Rio Building Detector",
            "task": "building_footprint_segmentation",
            "checkpoint": sn_ckpt,
            "fallback_checkpoint": None,
            "load_status": _ckpt_status(sn_ckpt),
            "size_mb": _ckpt_size_mb(sn_ckpt),
            "device": sn_cfg.get("device", "auto"),
            "dataset": "SpaceNet-Rio",
            "source": "https://huggingface.co/harshinde/spacenet-models",
            "license": "Apache-2.0",
            "version": "U-Net SpaceNet Rio",
        },
        "sam2": {
            "name": "SAM 2.1 Hiera Tiny",
            "task": "interactive_segmentation",
            "checkpoint": sam2_ckpt,
            "fallback_checkpoint": None,
            "load_status": _ckpt_status(sam2_ckpt),
            "size_mb": _ckpt_size_mb(sam2_ckpt),
            "device": sam2_cfg.get("device", "auto"),
            "dataset": "SA-1B",
            "source": "https://github.com/facebookresearch/sam2",
            "license": "Apache-2.0",
            "version": "SAM 2.1 Hiera Tiny",
        },
        "changeformer": {
            "name": "ChangeFormer V6 (LEVIR-CD)",
            "task": "binary_change_detection",
            "checkpoint": cf_ckpt,
            "fallback_checkpoint": None,
            "load_status": _ckpt_status(cf_ckpt),
            "size_mb": _ckpt_size_mb(cf_ckpt),
            "device": cf_cfg.get("device", "auto"),
            "dataset": "LEVIR-CD",
            "source": "https://github.com/wgcban/ChangeFormer",
            "license": "MIT",
            "version": "ChangeFormer V6",
        },
        "vlm": {
            "name": f"Local LLM via Ollama ({vlm_cfg.get('model', 'mistral')})",
            "task": "vision_language_reasoning",
            "checkpoint": f"ollama://{vlm_cfg.get('model', 'mistral')}",
            "fallback_checkpoint": "mock_adapter",
            "load_status": vlm_status,
            "size_mb": None,
            "device": "cpu",
            "dataset": None,
            "source": "https://ollama.com",
            "license": "varies by model",
            "version": vlm_cfg.get("model", "mistral"),
        },
    }
    return registry


def _probe_vlm(vlm_cfg: Dict[str, Any]) -> str:
    """Probe Ollama availability. Returns 'ready', 'fallback', or 'missing'."""
    import os
    from app.core.config import settings

    # In DEMO_MODE, MockVLM is always available
    if settings.DEMO_MODE:
        return "ready (mock)"

    base_url = vlm_cfg.get("ollama_base_url", "http://localhost:11434")
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                return "ready"
    except Exception:
        pass

    # Gemini fallback
    if os.getenv("GEMINI_API_KEY"):
        return "fallback (gemini)"

    return "missing"


# Module-level singleton — built once on import
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def get_registry() -> Dict[str, Dict[str, Any]]:
    """Return (or lazily build) the model registry."""
    global MODEL_REGISTRY
    if not MODEL_REGISTRY:
        MODEL_REGISTRY = build_registry()
    return MODEL_REGISTRY


def refresh_registry() -> Dict[str, Dict[str, Any]]:
    """Force-rebuild registry (e.g. after download_models.py runs)."""
    global MODEL_REGISTRY
    MODEL_REGISTRY = build_registry()
    return MODEL_REGISTRY
