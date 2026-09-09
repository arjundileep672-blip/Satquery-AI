"""
SatQuery AI — EuroSAT Scene Classifier (Inference)
===================================================
Loads the trained EfficientNet-B0 checkpoint and classifies satellite
image scenes into one of 10 EuroSAT land-use/land-cover categories.

EuroSAT Classes:
    AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
    Pasture, PermanentCrop, Residential, River, SeaLake

Usage (module-level singleton via get_classifier()):
    from models.eurosat_classifier import get_classifier, SceneClassification
    clf = get_classifier()
    result: SceneClassification = clf.classify(image_np)
    print(result.top_class, result.confidence)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger("satquery.models.eurosat")

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODULE_DIR = Path(__file__).resolve().parent
_CHECKPOINT_PATH = _MODULE_DIR / "eurosat" / "eurosat_classifier.pt"
_CLASS_NAMES_PATH = _MODULE_DIR / "eurosat" / "class_names.json"

# Fallback class list (matches training order)
_DEFAULT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Scene → relevant YOLO COCO class names (used by orchestrator for filtering)
SCENE_TO_YOLO_HINTS = {
    "AnnualCrop":            ["tractor", "truck", "car"],
    "Forest":                [],               # No vehicles expected
    "HerbaceousVegetation":  [],
    "Highway":               ["car", "truck", "bus", "motorcycle", "bicycle"],
    "Industrial":            ["truck", "car", "building"],
    "Pasture":               [],
    "PermanentCrop":         ["tractor"],
    "Residential":           ["car", "building", "person", "bicycle"],
    "River":                 ["boat"],
    "SeaLake":               ["boat", "ship"],
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ClassPrediction:
    """Single class prediction with probability."""
    class_name: str
    probability: float   # 0.0–1.0
    rank: int            # 1 = top, 2 = second, etc.


@dataclass
class SceneClassification:
    """
    Full scene classification result.
    top_class: highest-confidence EuroSAT label
    confidence: probability of top_class (0.0–1.0)
    top3: top-3 predictions with probabilities
    model: source model identifier
    inference_ms: inference time in milliseconds
    available: False if checkpoint is missing (graceful degradation)
    """
    top_class: str = "unknown"
    confidence: float = 0.0
    top3: List[ClassPrediction] = field(default_factory=list)
    model: str = "eurosat_efficientnet_b0"
    inference_ms: float = 0.0
    available: bool = True
    yolo_hints: List[str] = field(default_factory=list)  # COCO classes to prioritize


@dataclass
class PatchLabel:
    """Single class label from patch-based voting aggregation."""
    class_name: str
    coverage_pct: float     # % of tiles where this class was the top-1 winner
    tile_count: int         # number of tiles that voted for this class
    mean_confidence: float  # mean softmax probability across winning tiles
    rank: int               # 1 = dominant class by coverage


@dataclass
class PatchSceneResult:
    """
    Multi-label scene classification result from patch-based tiling.
    labels: ranked list of scene types present in the image (≥ min_coverage_pct).
    total_tiles: total number of tiles evaluated.
    tile_size: pixel side-length of each tile (224).
    stride: pixel step between tile origins (112 = 50% overlap).
    """
    labels: List[PatchLabel] = field(default_factory=list)
    total_tiles: int = 0
    tile_size: int = 224
    stride: int = 112
    inference_ms: float = 0.0
    model: str = "eurosat_efficientnet_b0"
    available: bool = True


# ── Lazy Singleton ────────────────────────────────────────────────────────────

_classifier_instance: Optional["EuroSATClassifier"] = None


def get_classifier() -> "EuroSATClassifier":
    """Return the module-level EuroSATClassifier singleton (lazy init)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EuroSATClassifier()
    return _classifier_instance


# ── Classifier Class ──────────────────────────────────────────────────────────

class EuroSATClassifier:
    """
    Inference wrapper for the trained EfficientNet-B0 EuroSAT classifier.
    Thread-safe after load(). load() is called lazily on first classify().
    """

    def __init__(self):
        self._model = None
        self._class_names: List[str] = []
        self._loaded: bool = False
        self._available: bool = False
        self._transform = None

    def _load(self) -> None:
        """Attempt to load checkpoint. Sets self._available=False if missing."""
        if self._loaded:
            return

        self._loaded = True  # Mark to avoid repeated attempts even on failure

        if not _CHECKPOINT_PATH.exists():
            logger.info(
                f"EuroSAT checkpoint not found at {_CHECKPOINT_PATH}. "
                "Run scripts/train_eurosat.py to train the model. "
                "Scene classification will be unavailable until then."
            )
            self._available = False
            return

        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import efficientnet_b0

            # Load class names
            if _CLASS_NAMES_PATH.exists():
                with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as fh:
                    self._class_names = json.load(fh)
            else:
                self._class_names = _DEFAULT_CLASSES

            # Load checkpoint
            ckpt = torch.load(str(_CHECKPOINT_PATH), map_location="cpu", weights_only=False)
            num_classes = ckpt.get("num_classes", len(self._class_names))

            # Build same architecture as training
            model = efficientnet_b0(weights=None)
            import torch.nn as nn
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # Override class names from checkpoint if present
            if "class_names" in ckpt and ckpt["class_names"]:
                self._class_names = ckpt["class_names"]

            self._model = model
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._available = True
            val_acc = ckpt.get("val_acc", "?")
            trained_epoch = ckpt.get("epoch", "?")
            logger.info(
                f"EuroSAT classifier loaded: {num_classes} classes | "
                f"Checkpoint epoch={trained_epoch} val_acc={val_acc:.1f}%"
                if isinstance(val_acc, float) else
                f"EuroSAT classifier loaded: {num_classes} classes"
            )

        except Exception as exc:
            logger.warning(f"EuroSAT classifier failed to load: {exc}")
            self._available = False

    def classify(self, image: np.ndarray) -> SceneClassification:
        """
        Classify satellite image into one of the EuroSAT land-use classes.

        Args:
            image: BGR numpy array (any size — resized internally).

        Returns:
            SceneClassification with top_class, confidence, top3, yolo_hints.
            If checkpoint is missing, returns available=False with unknown class.
        """
        self._load()

        if not self._available or self._model is None:
            return SceneClassification(
                top_class="unknown",
                confidence=0.0,
                top3=[],
                available=False,
                model="eurosat_efficientnet_b0 (not loaded)",
            )

        t0 = time.perf_counter()
        try:
            import torch
            # Convert BGR → RGB for transform
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            if img_rgb.ndim == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

            tensor = self._transform(img_rgb).unsqueeze(0)  # [1, 3, 224, 224]

            with torch.no_grad():
                logits = self._model(tensor)                 # [1, num_classes]
                probs = torch.softmax(logits, dim=1)[0]      # [num_classes]

            probs_np = probs.numpy()
            top_indices = np.argsort(probs_np)[::-1][:3]

            top3 = [
                ClassPrediction(
                    class_name=self._class_names[idx] if idx < len(self._class_names) else str(idx),
                    probability=float(probs_np[idx]),
                    rank=rank + 1,
                )
                for rank, idx in enumerate(top_indices)
            ]

            top_class = top3[0].class_name
            confidence = top3[0].probability
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            return SceneClassification(
                top_class=top_class,
                confidence=confidence,
                top3=top3,
                model="eurosat_efficientnet_b0",
                inference_ms=elapsed_ms,
                available=True,
                yolo_hints=SCENE_TO_YOLO_HINTS.get(top_class, []),
            )

        except Exception as exc:
            logger.warning(f"EuroSAT inference failed: {exc}")
            return SceneClassification(
                top_class="unknown",
                confidence=0.0,
                top3=[],
                available=False,
                model="eurosat_efficientnet_b0 (error)",
            )

    def classify_patches(
        self,
        image: np.ndarray,
        tile_size: int = 224,
        stride: int = 112,
        batch_size: int = 32,
        min_coverage_pct: float = 1.0,
    ) -> PatchSceneResult:
        """
        Tile the image into overlapping patches and classify each independently.
        Aggregates per-class tile votes into a coverage-percentage ranked list.

        Args:
            image: BGR numpy array of any size.
            tile_size: side length of each square tile in pixels (default 224 — matches EfficientNet input).
            stride: step between tile origins (default 112 = 50% overlap).
            batch_size: max tiles per forward pass to bound RAM usage (default 32).
            min_coverage_pct: suppress classes below this % in the output (default 1.0).

        Returns:
            PatchSceneResult with ranked multi-label coverage. Falls back to
            available=False (no crash) if the checkpoint is not loaded.

        Time complexity: O(N_tiles / batch_size) forward passes, where
            N_tiles ≈ ceil(H/stride) * ceil(W/stride).
        Space complexity: O(batch_size * 3 * 224 * 224 * 4 bytes) ≈ 20 MB peak at batch_size=32.
        """
        self._load()

        if not self._available or self._model is None:
            return PatchSceneResult(available=False, model="eurosat_efficientnet_b0 (not loaded)")

        t0 = time.perf_counter()
        try:
            import torch

            # Convert BGR → RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            if img_rgb.ndim == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

            h, w = img_rgb.shape[:2]

            # Pad image with replicated border so every tile is exactly tile_size × tile_size.
            pad_h = max(0, tile_size - h)
            pad_w = max(0, tile_size - w)
            if pad_h > 0 or pad_w > 0:
                img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                h, w = img_rgb.shape[:2]

            # Collect tile top-left origins
            ys = list(range(0, h - tile_size + 1, stride))
            xs = list(range(0, w - tile_size + 1, stride))
            # Ensure last row/column is included
            if not ys or ys[-1] + tile_size < h:
                ys.append(h - tile_size)
            if not xs or xs[-1] + tile_size < w:
                xs.append(w - tile_size)

            origins: List[Tuple[int, int]] = [(y, x) for y in ys for x in xs]
            total_tiles = len(origins)

            if total_tiles == 0:
                return PatchSceneResult(available=False, model="eurosat_efficientnet_b0 (no tiles)")

            # ── Batched inference ─────────────────────────────────────────────
            num_classes = len(self._class_names)
            # vote_counts[c] = number of tiles where class c was top-1
            vote_counts: List[int] = [0] * num_classes
            # conf_sums[c] = sum of winning softmax probabilities for class c
            conf_sums: List[float] = [0.0] * num_classes

            for batch_start in range(0, total_tiles, batch_size):
                batch_origins = origins[batch_start: batch_start + batch_size]
                tensors = []
                for (y, x) in batch_origins:
                    tile = img_rgb[y: y + tile_size, x: x + tile_size]
                    tensors.append(self._transform(tile))

                batch_tensor = torch.stack(tensors, dim=0)  # [B, 3, 224, 224]
                with torch.no_grad():
                    logits = self._model(batch_tensor)                    # [B, num_classes]
                    probs = torch.softmax(logits, dim=1).numpy()          # [B, num_classes]

                for tile_probs in probs:
                    winner_idx = int(np.argmax(tile_probs))
                    vote_counts[winner_idx] += 1
                    conf_sums[winner_idx] += float(tile_probs[winner_idx])

            # ── Aggregate into PatchLabel list ────────────────────────────────
            raw_labels = []
            for idx, count in enumerate(vote_counts):
                if count == 0:
                    continue
                cov_pct = round(count / total_tiles * 100, 1)
                mean_conf = round(conf_sums[idx] / count, 4)
                cls_name = self._class_names[idx] if idx < len(self._class_names) else str(idx)
                raw_labels.append((cov_pct, count, mean_conf, cls_name))

            # Sort descending by coverage, filter by threshold, assign rank
            raw_labels.sort(key=lambda t: t[0], reverse=True)
            labels = [
                PatchLabel(
                    class_name=cls_name,
                    coverage_pct=cov_pct,
                    tile_count=count,
                    mean_confidence=mean_conf,
                    rank=rank + 1,
                )
                for rank, (cov_pct, count, mean_conf, cls_name) in enumerate(raw_labels)
                if cov_pct >= min_coverage_pct
            ]

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                f"EuroSAT patch classification: {total_tiles} tiles, "
                f"top label={labels[0].class_name if labels else 'none'} "
                f"({labels[0].coverage_pct if labels else 0}%), {elapsed_ms}ms"
            )

            return PatchSceneResult(
                labels=labels,
                total_tiles=total_tiles,
                tile_size=tile_size,
                stride=stride,
                inference_ms=elapsed_ms,
                model="eurosat_efficientnet_b0",
                available=True,
            )

        except Exception as exc:
            logger.warning(f"EuroSAT patch classification failed: {exc}")
            return PatchSceneResult(available=False, model="eurosat_efficientnet_b0 (error)")

    def is_available(self) -> bool:
        """Return True if checkpoint is loaded and ready."""
        self._load()
        return self._available

    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        self._load()
        return self._class_names or _DEFAULT_CLASSES
