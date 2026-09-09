"""
SatQuery AI — EuroSAT Scene Classifier Training Script
=======================================================
Fine-tunes EfficientNet-B0 (ImageNet pretrained) on the EuroSAT satellite
image dataset for 10-class land-use / land-cover classification.

Usage:
    python scripts/train_eurosat.py
    python scripts/train_eurosat.py --epochs 3   # quick test
    python scripts/train_eurosat.py --epochs 15  # full accuracy run

Output:
    backend/models/eurosat/eurosat_classifier.pt   — best checkpoint
    backend/models/eurosat/class_names.json         — class index mapping

EuroSAT Classes (10):
    AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
    Pasture, PermanentCrop, Residential, River, SeaLake
"""

import argparse
import json
import os
import random
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
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "backend" / "models" / "eurosat"
CHECKPOINT_PATH = CHECKPOINT_DIR / "eurosat_classifier.pt"
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.json"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── EuroSAT Dataset ───────────────────────────────────────────────────────────

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
NUM_CLASSES = len(EUROSAT_CLASSES)


def download_dataset() -> Path:
    """Download EuroSAT dataset from Kaggle Hub and return root path."""
    print("\n[1/5] Downloading EuroSAT dataset via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
        print(f"      Dataset path: {path}")
        return Path(path)
    except Exception as exc:
        print(f"      ERROR: kagglehub download failed: {exc}")
        sys.exit(1)


def find_image_root(dataset_path: Path) -> Path:
    """
    Locate the root that contains subdirectories matching EuroSAT class names.
    kagglehub stores the files inside 'EuroSAT/'.
    """
    if all((dataset_path / cls).exists() for cls in EUROSAT_CLASSES[:3]):
        return dataset_path

    for candidate in ["EuroSAT", "2750", "eurosat"]:
        sub = dataset_path / candidate
        if sub.is_dir() and all((sub / cls).exists() for cls in EUROSAT_CLASSES[:3]):
            print(f"      Found image root at: {sub}")
            return sub

    for child in dataset_path.iterdir():
        if child.is_dir() and all((child / cls).exists() for cls in EUROSAT_CLASSES[:3]):
            print(f"      Found image root at: {child}")
            return child

    print("      WARNING: Could not locate class subdirs, using dataset_path as root")
    return dataset_path


class EuroSATDataset(Dataset):
    """
    Loads EuroSAT images from a folder structure:
        root/ClassName/image_XXXXX.jpg
    """

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
            img = Image.new("RGB", (64, 64), color=0)
        if self.transform:
            img = self.transform(img)
        return img, label


def build_samples(image_root: Path) -> List[Tuple[Path, int]]:
    """Scan class subdirectories and collect (path, label_idx) pairs."""
    samples: List[Tuple[Path, int]] = []
    class_to_idx: Dict[str, int] = {}

    # Build class index from existing subdirs matching EuroSAT class names
    found_classes = sorted([
        d.name for d in image_root.iterdir()
        if d.is_dir() and d.name in EUROSAT_CLASSES
    ])

    if not found_classes:
        # Fallback: use whatever subdirs exist
        found_classes = sorted([d.name for d in image_root.iterdir() if d.is_dir()])

    print(f"      Found {len(found_classes)} classes: {found_classes}")
    class_to_idx = {cls: idx for idx, cls in enumerate(found_classes)}

    for cls_name, idx in class_to_idx.items():
        cls_dir = image_root / cls_name
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
        samples.extend([(p, idx) for p in imgs])
        print(f"        {cls_name}: {len(imgs)} images")

    print(f"      Total samples: {len(samples)}")
    return samples, found_classes


# ── Transforms ────────────────────────────────────────────────────────────────
# EuroSAT patches are 64×64 — upscale to 224×224 for EfficientNet-B0

TRAIN_TRANSFORM = T.Compose([
    T.Resize((96, 96)),          # upscale small satellite patches
    T.RandomCrop((64, 64)),      # random crop for augmentation
    T.Resize((224, 224)),        # resize to EfficientNet input size
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
    print("\n[2/5] Building EfficientNet-B0 model...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the final classifier: 1280 → num_classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total params:     {total_params:,}")
    print(f"      Trainable params: {trainable_params:,}")
    return model


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
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
            print(
                f"      Epoch {epoch}/{total_epochs} "
                f"[{batch_idx + 1}/{len(loader)}] "
                f"Loss: {total_loss / total:.4f}  "
                f"Acc: {correct / total * 100:.1f}%",
                end="\r",
            )

    print()  # newline after \r
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
    class_correct = {cls: 0 for cls in class_names}
    class_total = {cls: 0 for cls in class_names}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            cls_name = class_names[label] if label < len(class_names) else str(label)
            class_total[cls_name] = class_total.get(cls_name, 0) + 1
            if pred == label:
                class_correct[cls_name] = class_correct.get(cls_name, 0) + 1

    per_class_acc = {
        cls: (class_correct.get(cls, 0) / class_total.get(cls, 1)) * 100
        for cls in class_names if class_total.get(cls, 0) > 0
    }
    return total_loss / total, correct / total * 100, per_class_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train EuroSAT scene classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SatQuery AI — EuroSAT Training")
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print("=" * 60)

    # 1. Download dataset
    dataset_path = download_dataset()
    image_root = find_image_root(dataset_path)

    # 2. Build sample list and datasets
    print("\n[3/5] Preparing datasets...")
    all_samples, class_names = build_samples(image_root)

    if not all_samples:
        print("ERROR: No images found. Check dataset path.")
        sys.exit(1)

    # Stratified split using indices
    n_val = int(len(all_samples) * args.val_split)
    n_train = len(all_samples) - n_val

    # Shuffle before split for reproducibility
    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    train_ds = EuroSATDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds = EuroSATDataset(val_samples, transform=VAL_TRANSFORM)
    print(f"      Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Weighted sampler for class balance
    label_counts = {}
    for _, lbl in train_samples:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    sample_weights = [1.0 / label_counts.get(lbl, 1) for _, lbl in train_samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Build model
    num_classes = len(class_names)
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Two-phase learning: lower LR for backbone, higher for head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training loop
    print(f"\n[4/5] Training for {args.epochs} epoch(s) on {device}...")
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc, per_class_acc = validate(
            model, val_loader, criterion, device, class_names
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:>2}/{args.epochs}  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  "
            f"[{elapsed:.0f}s]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "architecture": "efficientnet_b0",
                },
                CHECKPOINT_PATH,
            )
            print(f"      [OK] Best checkpoint saved (val_acc={val_acc:.2f}%)")

    total_time = time.time() - start_time

    # 5. Final report
    print(f"\n[5/5] Training complete!")
    print(f"      Total time:    {total_time / 60:.1f} min")
    print(f"      Best epoch:    {best_epoch} / {args.epochs}")
    print(f"      Best val acc:  {best_val_acc:.2f}%")
    print(f"      Checkpoint:    {CHECKPOINT_PATH}")
    print()

    # Per-class accuracy table from last val run
    print("  Per-class accuracy (last validation run):")
    print("  " + "-" * 45)
    for cls in class_names:
        acc = per_class_acc.get(cls, 0.0)
        bar = "#" * int(acc / 5)
        print(f"  {cls:<25} {acc:>6.1f}%  {bar}")
    print("  " + "-" * 45)

    # Save class names JSON
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    print(f"      Class names:   {CLASS_NAMES_PATH}")
    print()
    print("  Integration: checkpoint will be auto-loaded by SatQuery AI on next request.")
    print("  Run the backend server and upload a satellite image to see scene labels.")


if __name__ == "__main__":
    main()
