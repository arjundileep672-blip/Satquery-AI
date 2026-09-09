"""
SatQuery AI — ChangeFormer Change Detector
Wraps ChangeFormer for binary bi-temporal change detection.

Checkpoint source: https://github.com/wgcban/ChangeFormer
Pretrained dataset: LEVIR-CD / S2Looking

FALLBACK (always available — no PyTorch required):
  Image differencing + Otsu threshold + morphological cleanup.
  Clearly labeled in all response metadata.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings, load_models_config, resolve_device
from app.core.logging import get_logger

logger = get_logger("satquery.models.changeformer")

_changeformer_model = None
_changeformer_available: Optional[bool] = None


class ChangeFormerDetector:
    """
    Standard interface for ChangeFormer bi-temporal change detection.
    """

    def __init__(self, checkpoint: Optional[str] = None, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    def load(self):
        if _probe_changeformer():
            self.model = _get_changeformer_model()
        return self

    def predict(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        patch_size: int = 256,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Accepts:
            image_before: Reference T1 image (numpy array)
            image_after:  Target T2 image (numpy array, registered)
        Returns:
            {
              "change_mask": ...,
              "change_probability": ...,
              "changed_area_pixels": 123456,
              "change_percentage": 12.4
            }
        """
        raw = detect_changes(image_before, image_after, patch_size=patch_size, threshold=threshold)
        mask = raw["change_mask"]
        total_pixels = int(mask.size)
        changed_pixels = int(mask.sum())
        change_pct = round(100.0 * changed_pixels / max(total_pixels, 1), 3)

        # Average change score across changed pixels
        change_score = round(float(changed_pixels / max(total_pixels, 1)), 4)

        return {
            "change_mask": mask,
            "change_probability": change_score,  # Heuristic change score (uncalibrated)
            "change_score": change_score,
            "changed_area_pixels": changed_pixels,
            "change_percentage": change_pct,
            "model": raw.get("model", "ChangeFormer"),
            "fallback_used": raw.get("fallback_used", False),
            "processing_ms": raw.get("processing_ms", 0.0),
        }


def _probe_changeformer() -> bool:
    """Check if ChangeFormer + PyTorch are available."""
    global _changeformer_available
    if _changeformer_available is not None:
        return _changeformer_available

    cfg = load_models_config().get("changeformer", {})
    ckpt_rel = cfg.get("checkpoint", "models/changeformer/changeformer_levir.pt")
    if not ckpt_rel:
        _changeformer_available = False
        return False

    ckpt_path = settings.BASE_DIR.parent / ckpt_rel
    if not ckpt_path.exists() or ckpt_path.stat().st_size < 1000:
        _changeformer_available = False
        return False

    try:
        import torch  # noqa: F401
        _changeformer_available = True
    except ImportError:
        _changeformer_available = False

    return _changeformer_available


def _image_diff_change_mask(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: Optional[float] = None,
    morph_kernel: int = 5,
) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    if img2.shape[:2] != (h1, w1):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)

    def to_float_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            gray = np.mean(img.astype(np.float32), axis=2)
        else:
            gray = img.astype(np.float32)
        g_min, g_max = gray.min(), gray.max()
        if g_max > g_min:
            return (gray - g_min) / (g_max - g_min)
        return np.zeros_like(gray, dtype=np.float32)

    g1 = to_float_gray(img1)
    g2 = to_float_gray(img2)

    diff = np.abs(g1 - g2)
    diff_u8 = (diff * 255).astype(np.uint8)

    if threshold is not None:
        thr = int(threshold * 255)
        _, binary = cv2.threshold(diff_u8, thr, 1, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(diff_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened.astype(np.uint8)


def detect_changes(
    image1: np.ndarray,
    image2: np.ndarray,
    patch_size: int = 256,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    h1, w1 = image1.shape[:2]
    if image2.shape[:2] != (h1, w1):
        image2 = cv2.resize(image2, (w1, h1), interpolation=cv2.INTER_LINEAR)

    use_cf = _probe_changeformer()

    if use_cf:
        try:
            result_mask = _run_changeformer(image1, image2, patch_size)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "change_mask": result_mask,
                "model": "ChangeFormer",
                "fallback_used": False,
                "processing_ms": elapsed_ms,
            }
        except Exception as exc:
            logger.warning(f"ChangeFormer inference failed, using fallback: {exc}")

    change_mask = _image_diff_change_mask(image1, image2, threshold=threshold)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"Change detection (image-diff fallback) completed in {elapsed_ms}ms")

    return {
        "change_mask": change_mask,
        "model": "image_diff_otsu",
        "fallback_used": True,
        "processing_ms": elapsed_ms,
    }


def _run_changeformer(
    img1: np.ndarray,
    img2: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    import torch

    model = _get_changeformer_model()
    device = resolve_device()
    model = model.to(device).eval()

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)

    def prep(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                img = ((img.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r1 = cv2.resize(prep(img1), (patch_size, patch_size))
    r2 = cv2.resize(prep(img2), (patch_size, patch_size))

    def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_rgb).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return t.to(device)

    with torch.no_grad():
        pred = model(to_tensor(r1), to_tensor(r2))
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        if pred.shape[1] > 1:
            pred = torch.softmax(pred, dim=1)[:, 1:2, :, :]
        else:
            pred = torch.sigmoid(pred)

        pred_np = pred.squeeze().cpu().numpy()

    binary = (pred_np > 0.5).astype(np.uint8)
    mask_full = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full


def _get_changeformer_model():
    global _changeformer_model
    if _changeformer_model is not None:
        return _changeformer_model

    import sys
    import torch

    cfg = load_models_config().get("changeformer", {})
    ckpt_rel = cfg.get("checkpoint", "models/changeformer/changeformer_levir.pt")
    ckpt_path = settings.BASE_DIR.parent / ckpt_rel

    arch_candidates = [
        settings.BASE_DIR.parent / "models" / "changeformer" / "changeformer_arch",
        settings.BASE_DIR.parent / "backend" / "models" / "changeformer_arch",
    ]
    for d in arch_candidates:
        if d.exists() and str(d) not in sys.path:
            sys.path.insert(0, str(d))

    embed_dim = cfg.get("embed_dim", 256)
    try:
        from ChangeFormer import ChangeFormerV6  # type: ignore
        try:
            model = ChangeFormerV6(embed_dim=embed_dim)
        except TypeError:
            model = ChangeFormerV6()
    except ImportError as exc:
        raise ImportError(f"ChangeFormer architecture import failed: {exc}")

    if ckpt_path.exists():
        try:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict):
            if "model_G_state_dict" in state:
                state = state["model_G_state_dict"]
            elif "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        # Strip potential 'module.' prefix from DDP checkpoints
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                clean_k = k[7:] if k.startswith("module.") else k
                cleaned[clean_k] = v
            state = cleaned
        model.load_state_dict(state, strict=False)

    _changeformer_model = model
    return model
