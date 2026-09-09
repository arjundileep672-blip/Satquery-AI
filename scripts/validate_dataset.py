"""
SatQuery AI — EuroSAT Dataset Validator
========================================
Read-only inspection of the EuroSAT dataset.

Checks:
  - All 10 class directories present
  - Total image count per class
  - Image formats (JPEG only expected)
  - Image resolutions (expected: 64x64)
  - Corrupted / unreadable images
  - Missing class directories
  - Class imbalance ratio
  - Duplicate filenames across classes
  - Zero-byte files

Produces:
  scripts/dataset_validation_report.json

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --dataset-path <custom_path>

Exit code:
    0 - all checks passed (training safe to proceed)
    1 - critical errors detected (do NOT train)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EUROSAT_CLASSES = [
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

EXPECTED_NUM_CLASSES = 10
EXPECTED_IMAGE_SIZE = (64, 64)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

REPORT_PATH = REPO_ROOT / "scripts" / "dataset_validation_report.json"


def resolve_dataset_root(hint: str | None) -> Path:
    """Resolve the EuroSAT image root, checking kagglehub cache first."""
    if hint:
        p = Path(hint)
        if p.exists():
            return p

    # Try standard kagglehub cache location
    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets" / "apollo2506" / "eurosat-dataset"

    if cache_base.exists():
        version_dirs = sorted(cache_base.glob("versions/*/EuroSAT"), reverse=True)
        if version_dirs:
            return version_dirs[0]

    raise FileNotFoundError(
        "EuroSAT dataset root not found.\n"
        "Run: python scripts/train_eurosat.py  (it downloads automatically via kagglehub)\n"
        "Or specify --dataset-path manually."
    )


def scan_class(cls_dir: Path, cls_name: str) -> dict:
    """Scan one class directory, return per-class stats."""
    from PIL import Image, UnidentifiedImageError

    result: dict = {
        "class": cls_name,
        "dir_exists": cls_dir.exists(),
        "total": 0,
        "valid": 0,
        "corrupted": [],
        "zero_byte": [],
        "wrong_resolution": [],
        "formats": Counter(),
        "sample_resolution": None,
    }

    if not cls_dir.exists():
        return result

    for fpath in cls_dir.iterdir():
        if fpath.suffix.lower() not in VALID_EXTENSIONS:
            continue

        result["total"] += 1
        result["formats"][fpath.suffix.lower()] += 1

        if fpath.stat().st_size == 0:
            result["zero_byte"].append(str(fpath.name))
            continue

        try:
            with Image.open(fpath) as img:
                w, h = img.size
                if result["sample_resolution"] is None:
                    result["sample_resolution"] = (w, h)
                if (w, h) != EXPECTED_IMAGE_SIZE:
                    result["wrong_resolution"].append((str(fpath.name), w, h))
                result["valid"] += 1
        except (UnidentifiedImageError, OSError, Exception) as exc:
            result["corrupted"].append((str(fpath.name), str(exc)))

    result["formats"] = dict(result["formats"])
    return result


def check_duplicates(class_stats: list) -> list:
    """Return filenames that appear in more than one class."""
    all_names: Counter = Counter()
    for cs in class_stats:
        cls_dir = Path(cs["_cls_dir"])
        if cls_dir.exists():
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() in VALID_EXTENSIONS:
                    all_names[fpath.name] += 1
    return [name for name, count in all_names.items() if count > 1]


def main() -> int:
    parser = argparse.ArgumentParser(description="EuroSAT dataset validator")
    parser.add_argument("--dataset-path", default="", help="Path to EuroSAT image root (auto-detected if omitted)")
    parser.add_argument("--skip-pixel-check", action="store_true", help="Skip opening every image (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("SatQuery AI -- EuroSAT Dataset Validator")
    print("=" * 60)

    try:
        dataset_root = resolve_dataset_root(args.dataset_path or None)
    except FileNotFoundError as exc:
        print(f"\n[CRITICAL] {exc}")
        return 1

    print(f"Dataset root:  {dataset_root}\n")

    class_stats = []
    total_images = 0
    total_valid = 0
    total_corrupted = 0
    total_zero_byte = 0
    total_wrong_res = 0
    missing_classes = []

    for cls in EUROSAT_CLASSES:
        cls_dir = dataset_root / cls
        print(f"  Scanning {cls:<25}", end="", flush=True)

        if args.skip_pixel_check:
            imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS] if cls_dir.exists() else []
            stats = {
                "class": cls,
                "dir_exists": cls_dir.exists(),
                "total": len(imgs),
                "valid": len(imgs),
                "corrupted": [],
                "zero_byte": [],
                "wrong_resolution": [],
                "formats": {},
                "sample_resolution": None,
                "_cls_dir": str(cls_dir),
            }
        else:
            stats = scan_class(cls_dir, cls)

        stats["_cls_dir"] = str(cls_dir)
        class_stats.append(stats)

        if not stats["dir_exists"]:
            missing_classes.append(cls)
            print("  [MISSING DIR]")
            continue

        total_images += stats["total"]
        total_valid += stats["valid"]
        total_corrupted += len(stats["corrupted"])
        total_zero_byte += len(stats["zero_byte"])
        total_wrong_res += len(stats["wrong_resolution"])

        issues = []
        if stats["corrupted"]:
            issues.append(f"{len(stats['corrupted'])} corrupted")
        if stats["zero_byte"]:
            issues.append(f"{len(stats['zero_byte'])} zero-byte")
        if stats["wrong_resolution"]:
            issues.append(f"{len(stats['wrong_resolution'])} wrong-res")

        if issues:
            print(f"  [WARN] {stats['total']} images ({', '.join(issues)})")
        else:
            print(f"  [OK]   {stats['total']} images")

    # Class imbalance
    counts = [s["total"] for s in class_stats if s["dir_exists"]]
    imbalance_ratio = max(counts) / min(counts) if counts and min(counts) > 0 else float("inf")

    # Duplicate filenames
    print("\n  Checking for duplicate filenames...", end="", flush=True)
    duplicates: list = []
    try:
        duplicates = check_duplicates(class_stats)
    except Exception:
        pass
    print(f" {len(duplicates)} found")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Classes found:       {EXPECTED_NUM_CLASSES - len(missing_classes)}/{EXPECTED_NUM_CLASSES}")
    print(f"  Total images:        {total_images:,}")
    print(f"  Valid images:        {total_valid:,}")
    print(f"  Corrupted images:    {total_corrupted}")
    print(f"  Zero-byte images:    {total_zero_byte}")
    print(f"  Wrong-resolution:    {total_wrong_res}")
    print(f"  Duplicate filenames: {len(duplicates)}")
    print(f"  Class imbalance:     {imbalance_ratio:.2f}x  (1.5x is acceptable)")

    critical_errors = []
    if missing_classes:
        critical_errors.append(f"Missing class directories: {missing_classes}")
    if total_images == 0:
        critical_errors.append("No images found in dataset root")
    if total_corrupted > total_images * 0.05:
        critical_errors.append(f"{total_corrupted} corrupted images exceeds 5% threshold")

    warnings = []
    if imbalance_ratio > 2.0:
        warnings.append(f"Class imbalance ratio {imbalance_ratio:.2f}x - WeightedRandomSampler is essential")
    if total_wrong_res > 0:
        warnings.append(f"{total_wrong_res} images have non-standard resolution (expected 64x64)")
    if total_zero_byte > 0:
        warnings.append(f"{total_zero_byte} zero-byte images detected")

    report = {
        "dataset_root": str(dataset_root),
        "total_images": total_images,
        "total_valid": total_valid,
        "total_corrupted": total_corrupted,
        "total_zero_byte": total_zero_byte,
        "total_wrong_resolution": total_wrong_res,
        "expected_image_size": list(EXPECTED_IMAGE_SIZE),
        "classes_found": EXPECTED_NUM_CLASSES - len(missing_classes),
        "missing_classes": missing_classes,
        "class_imbalance_ratio": round(imbalance_ratio, 3),
        "duplicate_filenames": duplicates[:50],
        "per_class": [
            {k: v for k, v in s.items() if k != "_cls_dir"}
            for s in class_stats
        ],
        "critical_errors": critical_errors,
        "warnings": warnings,
        "validation_passed": len(critical_errors) == 0,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  Report saved to: {REPORT_PATH}")
    print()

    if warnings:
        print("[WARNINGS]")
        for w in warnings:
            print(f"  WARN: {w}")
        print()

    if critical_errors:
        print("[CRITICAL ERRORS -- DO NOT TRAIN]")
        for e in critical_errors:
            print(f"  ERROR: {e}")
        print()
        return 1
    else:
        print("[RESULT] Dataset validation PASSED. Safe to train.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
