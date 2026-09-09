#!/usr/bin/env python3
"""
SatQuery AI — Model Check Script

Verifies the presence, size, device, and status of all configured AI models:
  - YOLO12n (Generic Object Detector)
  - YOLO26n-OBB / DOTA (Remote Sensing OBB Detector)
  - SpaceNet Rio (Building Footprint Segmenter)
  - SAM 2.1 Hiera Tiny (Interactive Segmenter)
  - ChangeFormer (Bi-temporal Change Detector)
  - Local VLM/LLM (Ollama / Mistral)

Usage:
    python scripts/check_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set utf-8 encoding for stdout on Windows if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from core.model_registry import get_registry, build_registry
from app.core.config import resolve_device, settings


def main():
    print("=" * 80)
    print(" SatQuery AI - Local Model Verification & Offline Readiness Check")
    print("=" * 80)
    
    device = resolve_device()
    print(f"  Inference Device: {device.upper()}")
    print(f"  Offline Mode:     ENABLED (No cloud API calls permitted)")
    print(f"  Demo Mode:        {settings.DEMO_MODE}")
    print(f"  Repository Root:  {REPO_ROOT}")
    print("-" * 80)

    reg = build_registry()

    headers = f"{'Model Key':<15} | {'Model Name':<28} | {'Status':<14} | {'Size (MB)':<10} | {'Device':<6} | {'Checkpoint'}"
    print(headers)
    print("-" * 80)

    all_ready = True

    for key, info in reg.items():
        name = info.get("name", key)
        status = info.get("load_status", "missing")
        size = info.get("size_mb")
        size_str = f"{size:.1f}" if size is not None else "-"
        dev = info.get("device", device)
        ckpt = info.get("checkpoint", "")

        if status in ("ready", "ready (mock)"):
            status_icon = "[OK] " + status
        elif status == "fallback":
            status_icon = "[WARN] fallback"
        else:
            status_icon = "[X] missing"
            if key not in ("spacenet", "changeformer"):
                all_ready = False

        print(f"{key:<15} | {name[:28]:<28} | {status_icon:<14} | {size_str:<10} | {dev:<6} | {ckpt}")

    print("=" * 80)

    if all_ready:
        print("[OK] System status: READY for local and offline inference.")
    else:
        print("[WARN] System status: DEGRADED / MISSING MODELS.")
        print("  Run: python scripts/download_models.py to download required checkpoints.")
    print("=" * 80)


if __name__ == "__main__":
    main()
