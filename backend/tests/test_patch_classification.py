"""Quick runner for patch classification unit tests (no pytest needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

def _make_image(h, w, channels=3):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, channels), dtype=np.uint8)

import torch
import torch.nn as nn
import torchvision.transforms as T

from models.eurosat_classifier import EuroSATClassifier, PatchSceneResult

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

class _Zeros(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 10)

class _Rotating(nn.Module):
    def forward(self, x):
        b = x.shape[0]
        logits = torch.zeros(b, 10)
        for i in range(b):
            logits[i, i % 3] = 10.0
        return logits

class _AllZero(nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 10)
        logits[:, 0] = 100.0
        return logits

def _make_clf():
    c = EuroSATClassifier()
    c._loaded = True
    c._available = True
    c._class_names = CLASS_NAMES[:]
    c._model = _Zeros()
    c._transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])
    return c

def test_unavailable_no_crash():
    clf = EuroSATClassifier()
    clf._loaded = True
    clf._available = False
    clf._model = None
    r = clf.classify_patches(_make_image(976, 1805))
    assert isinstance(r, PatchSceneResult)
    assert r.available is False
    assert r.labels == []

def test_tile_count_448():
    r2 = _make_clf().classify_patches(_make_image(448, 448), tile_size=224, stride=112)
    assert r2.available is True
    assert r2.total_tiles == 9

def test_coverage_sums_to_100():
    clf3 = _make_clf()
    clf3._model = _Rotating()
    r3 = clf3.classify_patches(_make_image(976, 1805), min_coverage_pct=0.0)
    total_cov = sum(l.coverage_pct for l in r3.labels)
    assert r3.available is True
    assert abs(total_cov - 100.0) <= 1.0

def test_large_tile_count():
    r4 = _make_clf().classify_patches(_make_image(976, 1805))
    assert r4.total_tiles >= 50

def test_min_coverage_filter():
    clf5 = _make_clf()
    clf5._model = _AllZero()
    r5 = clf5.classify_patches(_make_image(448, 448), min_coverage_pct=5.0)
    assert len(r5.labels) == 1
    assert r5.labels[0].class_name == "AnnualCrop"
    assert r5.labels[0].coverage_pct == 100.0
    assert r5.labels[0].rank == 1

def test_real_checkpoint_if_available():
    real_clf = EuroSATClassifier()
    real_clf._load()
    if real_clf.is_available():
        r6 = real_clf.classify_patches(_make_image(976, 1805))
        assert r6.available is True
        assert len(r6.labels) > 0
        assert r6.tile_size == 224
        cov_sum = sum(l.coverage_pct for l in r6.labels)
        assert cov_sum >= 90.0

if __name__ == "__main__":
    test_unavailable_no_crash()
    test_tile_count_448()
    test_coverage_sums_to_100()
    test_large_tile_count()
    test_min_coverage_filter()
    test_real_checkpoint_if_available()
    print("All pytest-compatible patch classification tests PASSED")

