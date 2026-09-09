"""
SatQuery AI — Robust EuroSAT Training Script
=============================================
Fine-tunes EfficientNet-B0 on EuroSAT (10-class land-use/land-cover).

FEATURES
--------
- Runs entirely standalone from a terminal — no agent required
- Resumes from latest checkpoint automatically
- Saves best checkpoint AND latest checkpoint separately
- Mixed precision (AMP) when GPU is available
- WeightedRandomSampler for class imbalance (1.5x ratio)
- CosineAnnealingLR scheduler
- Graceful SIGINT/SIGTERM handling (saves state on Ctrl+C)
- Full JSON metrics log per epoch
- Sanity-check mode (--sanity-check, 3 epochs, 10% data)
- Hard abort with CUDA OOM recovery hint

GPU NOTE
--------
Detected environment: Intel Arc 140V GPU (8 GB shared).
PyTorch currently installed as CPU-only build (2.14.0+cpu).
CUDA availability: False

To enable GPU acceleration, you MUST install a CUDA or Intel XPU build of
PyTorch. Until then, training runs on CPU and will be slow (~10-20 min/epoch).

To install PyTorch with CUDA 12.x (for NVIDIA GPUs):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

For Intel Arc GPU via Intel Extension for PyTorch (IPEX):
    pip install intel-extension-for-pytorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

See README for full instructions.

USAGE
-----
    # Sanity check only (3 epochs, 10% data — runs quickly on CPU):
    python scripts/train_robust.py --sanity-check

    # Full training (CPU):
    python scripts/train_robust.py --epochs 15

    # Full training with custom batch size:
    python scripts/train_robust.py --epochs 15 --batch-size 16

    # Resume from existing checkpoint:
    python scripts/train_robust.py --epochs 15 --resume

OUTPUT
------
    runs/eurosat/
        best_checkpoint.pt      -- best validation accuracy
        latest_checkpoint.pt    -- most recent epoch (for resuming)
        training_config.json    -- hyperparameters and paths used
        metrics.json            -- per-epoch loss/accuracy log
        train.log               -- full text log (append mode)
    backend/models/eurosat/eurosat_classifier.pt  -- prod checkpoint (best)
    backend/models/eurosat/class_names.json       -- class index mapping
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs" / "eurosat"
PROD_DIR = REPO_ROOT / "backend" / "models" / "eurosat"
BEST_CKPT = RUNS_DIR / "best_checkpoint.pt"
LATEST_CKPT = RUNS_DIR / "latest_checkpoint.pt"
CONFIG_PATH = RUNS_DIR / "training_config.json"
METRICS_PATH = RUNS_DIR / "metrics.json"
LOG_PATH = RUNS_DIR / "train.log"
PROD_CKPT = PROD_DIR / "eurosat_classifier.pt"
CLASS_NAMES_PATH = PROD_DIR / "class_names.json"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
PROD_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Dataset constants ─────────────────────────────────────────────────────────

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
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """Configure dual logging to stdout and persistent log file."""
    logger = logging.getLogger("satquery.train")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (append mode — survives restarts)
    fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Dataset ───────────────────────────────────────────────────────────────────


def resolve_dataset_root() -> Path:
    """Auto-locate EuroSAT images from kagglehub cache. Exits if not found."""
    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets" / "apollo2506" / "eurosat-dataset"
    if cache_base.exists():
        version_dirs = sorted(cache_base.glob("versions/*/EuroSAT"), reverse=True)
        if version_dirs:
            return version_dirs[0]
    return None


class EuroSATDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Corrupted image: return blank tensor matching expected shape
            img = Image.new("RGB", (64, 64), color=0)
        if self.transform:
            img = self.transform(img)
        return img, label


def build_samples(image_root: Path, logger: logging.Logger) -> Tuple[List, List]:
    """Scan class subdirectories and collect (path, label_idx) pairs."""
    found = sorted([
        d.name for d in image_root.iterdir()
        if d.is_dir() and d.name in EUROSAT_CLASSES
    ])
    if not found:
        logger.error("No EuroSAT class subdirectories found at %s", image_root)
        sys.exit(1)

    class_to_idx = {cls: idx for idx, cls in enumerate(found)}
    samples: List[Tuple[Path, int]] = []

    for cls_name, idx in class_to_idx.items():
        cls_dir = image_root / cls_name
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        samples.extend([(p, idx) for p in imgs])
        logger.info("  %s: %d images", cls_name, len(imgs))

    logger.info("Total samples: %d across %d classes", len(samples), len(found))
    return samples, found


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.Resize((96, 96)),
    T.RandomCrop((64, 64)),
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=90),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Load EfficientNet-B0 with ImageNet weights and replace classifier head."""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)
    return model


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_acc: float,
    best_val_acc: float,
    class_names: List[str],
    scaler,
) -> None:
    """Save full training state. Atomic: write to .tmp then rename."""
    tmp_path = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "num_classes": len(class_names),
        "architecture": "efficientnet_b0",
    }, str(tmp_path))
    tmp_path.replace(path)


def save_prod_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_acc: float,
    class_names: List[str],
) -> None:
    """Save production-format checkpoint (model weights + metadata only)."""
    tmp_path = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "class_names": class_names,
        "num_classes": len(class_names),
        "architecture": "efficientnet_b0",
    }, str(tmp_path))
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler,
    logger: logging.Logger,
) -> Tuple[int, float, float]:
    """Load training state, return (start_epoch, val_acc, best_val_acc)."""
    if not path.exists():
        return 0, 0.0, 0.0

    logger.info("Loading checkpoint: %s", path)
    try:
        state = torch.load(str(path), map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        epoch = state.get("epoch", 0)
        val_acc = state.get("val_acc", 0.0)
        best_val_acc = state.get("best_val_acc", val_acc)
        logger.info("Resumed from epoch %d  (val_acc=%.2f%%)", epoch, val_acc)
        return epoch, val_acc, best_val_acc
    except Exception as exc:
        logger.error("Failed to load checkpoint %s: %s — starting fresh", path, exc)
        return 0, 0.0, 0.0


# ── Training loop ─────────────────────────────────────────────────────────────


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler,
    logger: logging.Logger,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            sys.stdout.write(
                f"\r  Epoch {epoch}/{total_epochs} "
                f"[{batch_idx + 1}/{len(loader)}] "
                f"Loss: {total_loss / total:.4f}  "
                f"Acc: {correct / total * 100:.1f}%  "
            )
            sys.stdout.flush()

    print()
    return total_loss / total, correct / total * 100


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct: Dict[str, int] = {cls: 0 for cls in class_names}
    class_total: Dict[str, int] = {cls: 0 for cls in class_names}

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            cls = class_names[label] if label < len(class_names) else str(label)
            class_total[cls] = class_total.get(cls, 0) + 1
            if pred == label:
                class_correct[cls] = class_correct.get(cls, 0) + 1

    per_class_acc = {
        cls: (class_correct.get(cls, 0) / class_total.get(cls, 1)) * 100
        for cls in class_names
        if class_total.get(cls, 0) > 0
    }
    return total_loss / total, correct / total * 100, per_class_acc


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SatQuery AI — Robust EuroSAT Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs to train (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (default: 0, safe on Windows)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if it exists")
    parser.add_argument("--sanity-check", action="store_true", help="Run 3-epoch sanity check on 10%% of data")
    parser.add_argument("--subset", type=float, default=None, help="Fraction of dataset to use (e.g. 0.1 for 10%%)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision (AMP)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    # ── Sanity-check / subset overrides ─────────────────────────────────────
    sanity_mode = args.sanity_check
    subset_ratio = 1.0
    if sanity_mode:
        logger.info("*** SANITY-CHECK MODE: 3 epochs, 10%% of data ***")
        if args.epochs == 15:
            args.epochs = 3
        args.batch_size = min(args.batch_size, 16)
        subset_ratio = 0.1
    elif args.subset is not None:
        subset_ratio = max(0.01, min(1.0, args.subset))
        logger.info("*** SUBSET MODE: Using %.1f%% of data ***", subset_ratio * 100)
        args.batch_size = min(args.batch_size, 16)

    # ── Device detection ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp

    logger.info("=" * 60)
    logger.info("SatQuery AI -- EuroSAT Training")
    logger.info("=" * 60)
    logger.info("Device:        %s", device)
    logger.info("AMP enabled:   %s", use_amp)
    logger.info("Epochs:        %d", args.epochs)
    logger.info("Batch size:    %d", args.batch_size)
    logger.info("Workers:       %d", args.workers)
    logger.info("Resume:        %s", args.resume)

    if device.type == "cpu":
        logger.warning("=" * 60)
        logger.warning("WARNING: Training on CPU only.")
        logger.warning("PyTorch build detected: %s", torch.__version__)
        logger.warning("No CUDA GPU available. Each epoch may take 10-20 minutes.")
        logger.warning("For GPU acceleration, install a CUDA build of PyTorch:")
        logger.warning("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        logger.warning("=" * 60)

    # ── Dataset ─────────────────────────────────────────────────────────────
    logger.info("Locating dataset...")
    dataset_root = resolve_dataset_root()
    if dataset_root is None:
        logger.error("EuroSAT dataset not found in kagglehub cache.")
        logger.error("Run first: python scripts/train_eurosat.py  (downloads dataset).")
        logger.error("Or run: python scripts/validate_dataset.py  to check paths.")
        return 1

    logger.info("Dataset root: %s", dataset_root)
    all_samples, class_names = build_samples(dataset_root, logger)

    if not all_samples:
        logger.error("No images found. Aborting.")
        return 1

    # Sanity-check / subset subsample
    if subset_ratio < 1.0:
        rng_s = random.Random(SEED)
        rng_s.shuffle(all_samples)
        all_samples = all_samples[: max(200, int(len(all_samples) * subset_ratio))]
        logger.info("Subsample mode: using %d samples (%.1f%% of full dataset)", len(all_samples), subset_ratio * 100)

    # Shuffle and split
    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_split))
    n_train = len(all_samples) - n_val
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    logger.info("Train: %d  |  Val: %d", len(train_samples), len(val_samples))

    # WeightedRandomSampler — compensate for class imbalance
    label_counts: Dict[int, int] = {}
    for _, lbl in train_samples:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    sample_weights = [1.0 / label_counts.get(lbl, 1) for _, lbl in train_samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    train_ds = EuroSATDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds = EuroSATDataset(val_samples, transform=VAL_TRANSFORM)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    num_classes = len(class_names)
    logger.info("Building EfficientNet-B0 for %d classes...", num_classes)
    model = build_model(num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params: %s  |  Trainable: %s", f"{total_params:,}", f"{trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Differential learning rates: lower for frozen backbone, higher for head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpoint resume ───────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    current_val_acc = 0.0
    metrics_log: List[dict] = []

    if args.resume and LATEST_CKPT.exists():
        start_epoch, current_val_acc, best_val_acc = load_checkpoint(
            LATEST_CKPT, model, optimizer, scheduler, device, scaler, logger
        )
        # Load existing metrics log if present
        if METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, encoding="utf-8") as fh:
                    metrics_log = json.load(fh)
            except Exception:
                metrics_log = []
    elif not args.resume and LATEST_CKPT.exists():
        logger.info("Existing checkpoint found at %s (not resuming — use --resume to continue)", LATEST_CKPT)

    # ── Graceful interrupt handler ──────────────────────────────────────────
    interrupted = False

    def _handle_signal(signum, frame):
        nonlocal interrupted
        logger.warning("Signal %s received. Saving state and exiting cleanly...", signum)
        interrupted = True

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    # ── Save training config ─────────────────────────────────────────────────
    config = {
        "architecture": "efficientnet_b0",
        "dataset": "EuroSAT",
        "dataset_root": str(dataset_root),
        "num_classes": num_classes,
        "class_names": class_names,
        "total_images": len(all_samples),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "val_split": args.val_split,
        "use_amp": use_amp,
        "device": str(device),
        "workers": args.workers,
        "resume": args.resume,
        "sanity_mode": sanity_mode,
        "best_checkpoint": str(BEST_CKPT),
        "latest_checkpoint": str(LATEST_CKPT),
        "prod_checkpoint": str(PROD_CKPT),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Config saved: %s", CONFIG_PATH)

    # ── Verify checkpoint dir writable ──────────────────────────────────────
    test_file = RUNS_DIR / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as exc:
        logger.error("Checkpoint directory not writable: %s — %s", RUNS_DIR, exc)
        return 1

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info("Starting training from epoch %d / %d", start_epoch + 1, args.epochs)

    per_class_acc: Dict[str, float] = {}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if interrupted:
            logger.warning("Interrupted before epoch %d. State saved.", epoch)
            break

        t0 = time.time()

        try:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, args.epochs, scaler, logger
            )
            val_loss, val_acc, per_class_acc = validate(
                model, val_loader, criterion, device, class_names
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA Out-of-Memory at epoch %d!", epoch)
            logger.error("Reduce --batch-size (currently %d) and retry.", args.batch_size)
            save_checkpoint(
                LATEST_CKPT, model, optimizer, scheduler,
                epoch - 1, current_val_acc, best_val_acc, class_names, scaler
            )
            return 1

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d  TrainLoss: %.4f  TrainAcc: %.1f%%  "
            "ValLoss: %.4f  ValAcc: %.1f%%  [%.0fs]",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, elapsed
        )

        current_val_acc = val_acc

        # Always save latest checkpoint
        save_checkpoint(
            LATEST_CKPT, model, optimizer, scheduler,
            epoch, val_acc, best_val_acc, class_names, scaler
        )

        # Save best checkpoint if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                BEST_CKPT, model, optimizer, scheduler,
                epoch, val_acc, best_val_acc, class_names, scaler
            )
            # Sync to production path
            save_prod_checkpoint(PROD_CKPT, model, epoch, val_acc, class_names)
            logger.info("  >> New best! val_acc=%.2f%%  Checkpoints saved.", val_acc)

        # Append to metrics log
        metrics_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "elapsed_s": round(elapsed, 1),
            "per_class_acc": {k: round(v, 2) for k, v in per_class_acc.items()},
        })
        with open(METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump(metrics_log, fh, indent=2)

        if interrupted:
            logger.warning("Training interrupted after epoch %d. Checkpoints saved.", epoch)
            break

    # ── Final report ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  Best val_acc:      %.2f%%", best_val_acc)
    logger.info("  Best checkpoint:   %s", BEST_CKPT)
    logger.info("  Latest checkpoint: %s", LATEST_CKPT)
    logger.info("  Prod checkpoint:   %s", PROD_CKPT)
    logger.info("  Metrics log:       %s", METRICS_PATH)

    # Save class names JSON for inference
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    logger.info("  Class names:       %s", CLASS_NAMES_PATH)

    if sanity_mode:
        logger.info("")
        logger.info("SANITY CHECK PASSED.")
        logger.info("Full training command:")
        logger.info("  python scripts/train_robust.py --epochs 15")
        logger.info("Resume command (if interrupted):")
        logger.info("  python scripts/train_robust.py --epochs 15 --resume")

    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
