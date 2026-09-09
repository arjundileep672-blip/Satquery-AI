"""
Inspect the existing EuroSAT checkpoint without modifying it.
"""
import torch
from pathlib import Path

ckpt_path = Path(__file__).resolve().parent.parent / "backend" / "models" / "eurosat" / "eurosat_classifier.pt"
print(f"Path:   {ckpt_path}")
print(f"Exists: {ckpt_path.exists()}")

if ckpt_path.exists():
    sz = ckpt_path.stat().st_size / 1024 / 1024
    print(f"Size:   {sz:.2f} MB")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        print(f"Keys:          {list(ckpt.keys())}")
        print(f"epoch:         {ckpt.get('epoch')}")
        print(f"val_acc:       {ckpt.get('val_acc')}")
        print(f"num_classes:   {ckpt.get('num_classes')}")
        print(f"architecture:  {ckpt.get('architecture')}")
        print(f"class_names:   {ckpt.get('class_names')}")
    except Exception as e:
        print(f"Load error: {e}")
else:
    print("CHECKPOINT NOT FOUND.")
