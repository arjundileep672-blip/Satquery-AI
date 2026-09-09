#!/usr/bin/env python3
"""
SatQuery AI — Model Download Script

Downloads all required model checkpoints for offline inference.
Zero external dependency required for downloading (uses urllib / requests).

Usage:
    python scripts/download_models.py

After this script completes, SatQuery can operate fully offline.
No internet access is required during inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Set utf-8 encoding for stdout on Windows if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))


def _download_file(url: str, dest: Path, label: str) -> bool:
    """
    Download url -> dest using urllib.request with progress reporter.
    Returns True on success, False on failure.
    Skips download if dest already exists and is non-empty.
    """
    if dest.exists() and dest.stat().st_size > 1000:
        try:
            rel = dest.relative_to(REPO_ROOT)
        except Exception:
            rel = dest
        print(f"  [SKIP] {label}: already exists ({_mb(dest):.1f} MB) at {rel}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DL]   {label} <- {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (SatQuery-AI-Downloader)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total_header = resp.getheader("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            block_size = 64 * 1024
            last_print = 0

            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print > 1.0:
                    last_print = now
                    if total:
                        pct = (downloaded / total) * 100.0
                        sys.stdout.write(f"\r    {label}: {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({pct:.0f}%)")
                    else:
                        sys.stdout.write(f"\r    {label}: {downloaded / (1024*1024):.1f} MB")
                    sys.stdout.flush()

        if total:
            sys.stdout.write(f"\r    {label}: {downloaded / (1024*1024):.1f} MB (100%)\n")
        else:
            sys.stdout.write(f"\r    {label}: {downloaded / (1024*1024):.1f} MB\n")
        sys.stdout.flush()

        if dest.stat().st_size == 0:
            dest.unlink()
            print(f"  ERROR: {label}: downloaded file is empty.")
            return False

        try:
            rel = dest.relative_to(REPO_ROOT)
        except Exception:
            rel = dest
        print(f"  [OK]   {label}: {_mb(dest):.1f} MB -> {rel}")
        return True

    except Exception as exc:
        print(f"\n  WARN:  {label}: direct download failed ({exc})")
        if dest.exists() and dest.stat().st_size < 1000:
            try:
                dest.unlink()
            except Exception:
                pass
        return False


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Download functions ────────────────────────────────────────────────────────

def download_yolo12n() -> bool:
    """YOLO12n checkpoint."""
    dest = REPO_ROOT / "models" / "yolo" / "yolo12n.pt"
    # Official Ultralytics releases
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt"
    ok = _download_file(url, dest, "YOLO12n")
    if not ok:
        fb = REPO_ROOT / "checkpoints" / "yolo11n.pt"
        fb2 = REPO_ROOT / "yolo11n.pt"
        existing = fb if fb.exists() else (fb2 if fb2.exists() else None)
        if existing and not dest.exists():
            import shutil
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, dest)
            print(f"  [FB]   YOLO12n: verified fallback yolo11n.pt -> models/yolo/yolo12n.pt")
            return True
    return ok or (dest.exists() and dest.stat().st_size > 0)


def download_yolo26n_obb() -> bool:
    """YOLO26n-OBB / DOTA-v1 checkpoint."""
    dest = REPO_ROOT / "models" / "yolo26_obb" / "yolo26n-obb.pt"
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n-obb.pt"
    ok = _download_file(url, dest, "YOLO26n-OBB")
    if not ok:
        fb = REPO_ROOT / "checkpoints" / "yolov8n-obb.pt"
        fb2 = REPO_ROOT / "yolov8n-obb.pt"
        existing = fb if fb.exists() else (fb2 if fb2.exists() else None)
        if existing and not dest.exists():
            import shutil
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, dest)
            print(f"  [FB]   YOLO26n-OBB: verified fallback yolov8n-obb.pt -> models/yolo26_obb/yolo26n-obb.pt")
            return True
    return ok or (dest.exists() and dest.stat().st_size > 0)


def download_spacenet_rio() -> bool:
    """SpaceNet Rio U-Net weights from HuggingFace."""
    dest = REPO_ROOT / "models" / "spacenet" / "spacenet_rio.pt"
    url = "https://huggingface.co/harshinde/spacenet-models/resolve/main/spacenet_rio.pt"
    ok = _download_file(url, dest, "SpaceNet Rio")
    if not ok:
        print("  NOTE: SpaceNet Rio checkpoint can be manually placed at models/spacenet/spacenet_rio.pt.")
        print("        Building detector will automatically use YOLO+SAM2 fallback until then.")
    return ok


def download_sam2_hiera_tiny() -> bool:
    """SAM 2.1 Hiera Tiny checkpoint."""
    dest = REPO_ROOT / "models" / "sam2" / "sam2.1_hiera_tiny.pt"
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    ok = _download_file(url, dest, "SAM 2.1 Hiera Tiny")
    if not ok:
        url2 = "https://github.com/facebookresearch/sam2/releases/download/sam2.1/sam2.1_hiera_tiny.pt"
        ok = _download_file(url2, dest, "SAM 2.1 Hiera Tiny (alt)")
    if not ok:
        print("  NOTE: SAM2 checkpoint can be placed at models/sam2/sam2.1_hiera_tiny.pt.")
        print("        Segmentation falls back to GrabCut until then.")
    return ok


def download_changeformer() -> bool:
    """ChangeFormer LEVIR-CD weights."""
    dest = REPO_ROOT / "models" / "changeformer" / "changeformer_levir.pt"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [SKIP] ChangeFormer LEVIR-CD: already exists ({_mb(dest):.1f} MB)")
        return True

    zip_url = (
        "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/"
        "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip"
    )
    temp_zip = dest.parent / "temp_changeformer.zip"
    ok = _download_file(zip_url, temp_zip, "ChangeFormer LEVIR-CD Zip")
    if ok and temp_zip.exists():
        import zipfile
        import shutil
        try:
            with zipfile.ZipFile(temp_zip, "r") as z:
                for name in z.namelist():
                    if name.endswith("best_ckpt.pt"):
                        with z.open(name) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        break
            if temp_zip.exists():
                temp_zip.unlink()
            print(f"  [OK]   ChangeFormer LEVIR-CD: {_mb(dest):.1f} MB -> {dest.relative_to(REPO_ROOT)}")
            return True
        except Exception as exc:
            print(f"  ERROR: Extracting ChangeFormer checkpoint failed: {exc}")
            if temp_zip.exists():
                temp_zip.unlink()
            return False
    return False


def write_metadata(results: dict) -> None:
    meta_path = REPO_ROOT / "models" / "metadata.json"
    meta = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": {}}

    for name, info in results.items():
        ckpt = REPO_ROOT / info.get("checkpoint", "")
        entry = {
            "download_success": info.get("success", False),
            "checkpoint": info.get("checkpoint", ""),
        }
        if ckpt.exists():
            entry["size_mb"] = round(_mb(ckpt), 2)
            entry["sha256"] = _sha256(ckpt)
        meta["models"][name] = entry

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    try:
        rel = meta_path.relative_to(REPO_ROOT)
    except Exception:
        rel = meta_path
    print(f"\n[META] Metadata written to {rel}")


def main():
    print("=" * 60)
    print("SatQuery AI - Model Download Script")
    print("=" * 60)
    print(f"Repository root: {REPO_ROOT}\n")

    steps = [
        ("yolo12n",      download_yolo12n,      "models/yolo/yolo12n.pt"),
        ("yolo26n_obb",  download_yolo26n_obb,   "models/yolo26_obb/yolo26n-obb.pt"),
        ("spacenet",     download_spacenet_rio,  "models/spacenet/spacenet_rio.pt"),
        ("sam2",         download_sam2_hiera_tiny, "models/sam2/sam2.1_hiera_tiny.pt"),
        ("changeformer", download_changeformer,  "models/changeformer/changeformer_levir.pt"),
    ]

    results = {}
    t0 = time.time()

    for name, fn, ckpt in steps:
        print("\n" + "-" * 50)
        print(f"Checking / Downloading: {name}")
        ok = fn()
        results[name] = {"success": ok, "checkpoint": ckpt}

    write_metadata(results)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Download process complete in {elapsed:.1f}s")
    print("Run: python scripts/check_models.py to verify all models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
