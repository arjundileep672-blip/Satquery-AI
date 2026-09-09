#!/usr/bin/env python3
"""
Run local inference smoke tests for every SatQuery vision model and write
latency / pass-fail results for the metrics page graphs.

Usage:
    python scripts/smoke_test_models.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = BACKEND / "models" / "smoke_test_results.json"


def _synth_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    img[:, :] = (42, 96, 58)
    cv2.rectangle(img, (36, 40), (150, 155), (190, 190, 185), -1)
    cv2.rectangle(img, (170, 70), (290, 210), (80, 85, 110), -1)
    cv2.circle(img, (90, 240), 28, (210, 200, 40), -1)
    noise = rng.integers(0, 18, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _record(test_id: str, name: str, fn, load_status: str) -> dict:
    t0 = time.perf_counter()
    try:
        summary = fn()
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "id": test_id,
            "name": name,
            "passed": True,
            "latency_ms": latency,
            "load_status": load_status,
            "output_summary": str(summary),
            "error": None,
        }
    except Exception as exc:
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "id": test_id,
            "name": name,
            "passed": False,
            "latency_ms": latency,
            "load_status": load_status,
            "output_summary": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_pytest() -> dict:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(BACKEND / "tests" / "test_models.py"),
        "-q",
        "--tb=line",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
    passed = failed = skipped = 0
    # pytest -q summary like: "18 passed, 1 skipped in 12.3s"
    for token in stdout.replace(",", " ").split():
        pass
    import re

    m = re.search(r"(\d+)\s+passed", stdout)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", stdout)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+)\s+skipped", stdout)
    if m:
        skipped = int(m.group(1))
    return {
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
        "latency_ms": elapsed,
        "summary_line": stdout.strip().splitlines()[-1] if stdout.strip() else "",
    }


def main() -> int:
    from app.core.config import resolve_device
    from core.model_registry import build_registry

    device = resolve_device()
    reg = build_registry()
    image = _synth_image(1)
    before = _synth_image(1)
    after = _synth_image(2)
    after[40:155, 36:150] = (40, 40, 40)

    tests = []

    def eurosat():
        from models.eurosat_classifier import get_classifier

        clf = get_classifier()
        result = clf.classify(image)
        if not result.available:
            raise RuntimeError(result.model)
        return f"{result.top_class} ({result.confidence:.2f})"

    tests.append(
        _record(
            "eurosat",
            "EuroSAT EfficientNet-B0",
            eurosat,
            "ready" if (BACKEND / "models" / "eurosat" / "eurosat_classifier.pt").exists() else "missing",
        )
    )

    def yolo12():
        from pipelines.detection import run_generic_detection

        result = run_generic_detection(image, conf=0.1)
        if not hasattr(result, "detections"):
            raise RuntimeError("no detections attribute")
        return f"{result.count} detections · {result.model}"

    tests.append(
        _record("yolo12n", "YOLO12n", yolo12, reg.get("yolo12n", {}).get("load_status", "missing"))
    )

    def yolo26():
        from pipelines.detection import run_remote_detection

        result = run_remote_detection(image, query="detect plane and ship", conf=0.1)
        if not hasattr(result, "detections"):
            raise RuntimeError("no detections attribute")
        return f"{result.count} detections · {getattr(result, 'model', 'yolo26n_obb')}"

    tests.append(
        _record(
            "yolo26n_obb",
            "YOLO26n-OBB",
            yolo26,
            reg.get("yolo26n_obb", {}).get("load_status", "missing"),
        )
    )

    def spacenet():
        from pipelines.segmentation import run_building_segmentation

        result = run_building_segmentation(image)
        count = result.get("count", 0)
        model = result.get("model", "unknown")
        return f"{count} buildings · {model}"

    tests.append(
        _record("spacenet", "SpaceNet Rio / buildings", spacenet, reg.get("spacenet", {}).get("load_status", "missing"))
    )

    def sam2():
        from pipelines.segmentation import run_segmentation

        result = run_segmentation(image, conf=0.1)
        if not hasattr(result, "masks"):
            raise RuntimeError("no masks attribute")
        return f"{result.count} masks · {result.model}"

    tests.append(_record("sam2", "SAM 2.1 Tiny", sam2, reg.get("sam2", {}).get("load_status", "missing")))

    def changeformer():
        from pipelines.change_detection import run_change_detection

        result = run_change_detection(before, after)
        if not hasattr(result, "change_percentage"):
            raise RuntimeError("no change_percentage")
        return f"{result.change_percentage:.2f}% changed · {result.model}"

    tests.append(
        _record(
            "changeformer",
            "ChangeFormer V6",
            changeformer,
            reg.get("changeformer", {}).get("load_status", "missing"),
        )
    )

    print("Running pytest backend/tests/test_models.py ...")
    pytest_info = _run_pytest()

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "tests": tests,
        "pytest": pytest_info,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    passed_n = sum(1 for t in tests if t["passed"])
    print(f"Smoke tests: {passed_n}/{len(tests)} passed  device={device}")
    for t in tests:
        flag = "PASS" if t["passed"] else "FAIL"
        print(f"  [{flag}] {t['name']:<28} {t['latency_ms']:>8} ms  {t['output_summary'] or t['error']}")
    print(f"Pytest: {pytest_info.get('summary_line')}")
    print(f"Wrote {OUT_PATH}")
    return 0 if passed_n == len(tests) and pytest_info.get("exit_code", 1) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
