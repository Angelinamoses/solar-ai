"""
SolarGuard AI — Model Training
================================
Provides quick-prototype and full training modes for YOLOv8n.

Usage
-----
Quick prototype (default, ~2 epochs, subset of data):
    python train_model.py

Full training:
    python train_model.py --mode full --epochs 50

Options
-------
    --mode      quick | full        (default: quick)
    --epochs    int                 (default: 2 for quick, 30 for full)
    --batch     int                 (default: 8 for quick, 16 for full)
    --imgsz     int                 (default: 320 for quick, 640 for full)
    --device    cpu | 0 | 0,1 ...   (default: auto-detect GPU)
    --output    path                (default: models/)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

EXPECTED_CLASSES = [
    "MultiByPassed", "MultiDiode", "MultiHotSpot", "SingleByPassed",
    "SingleDiode", "SingleHotSpot", "StringOpenCircuit", "StringReversedPolarity",
]


def auto_device() -> str:
    """Return '0' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def build_quick_subset(
    processed_dir: Path,
    subset_dir: Path,
    max_images: int = 1200,
    seed: int = 42,
) -> Path:
    """
    Create a class-aware subset of training images under subset_dir.
    Stratifies by class to preserve minority-class representation.
    Returns path to new data.yaml.
    """
    random.seed(seed)
    src_img_dir = processed_dir / "train" / "images"
    src_lbl_dir = processed_dir / "train" / "labels"

    # Group label files by their most-common class
    class_to_stems: dict[int, list[str]] = {i: [] for i in range(len(EXPECTED_CLASSES))}
    all_stems = [f.stem for f in sorted(src_lbl_dir.glob("*.txt"))]

    for stem in all_stems:
        lbl_path = src_lbl_dir / f"{stem}.txt"
        lines = lbl_path.read_text(encoding="utf-8", errors="replace").splitlines()
        class_counts: dict[int, int] = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cid = int(parts[0])
                    class_counts[cid] = class_counts.get(cid, 0) + 1
                except ValueError:
                    pass
        if class_counts:
            dominant = max(class_counts, key=class_counts.get)
            class_to_stems[dominant].append(stem)

    # Stratified sample
    per_class = max(1, max_images // len(EXPECTED_CLASSES))
    selected: list[str] = []
    for cid, stems in class_to_stems.items():
        random.shuffle(stems)
        selected.extend(stems[:per_class])
    selected = list(dict.fromkeys(selected))  # deduplicate, preserve order

    print(f"Quick subset: selected {len(selected)} images from {len(all_stems)} training images")

    # Copy to subset_dir
    dst_img = subset_dir / "train" / "images"
    dst_lbl = subset_dir / "train" / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for stem in selected:
        for ext in (".jpg", ".png"):
            src = src_img_dir / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, dst_img / src.name)
                break
        lbl_src = src_lbl_dir / f"{stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, dst_lbl / lbl_src.name)

    # Copy valid and test splits unchanged
    for split in ("valid", "test"):
        for subdir in ("images", "labels"):
            src = processed_dir / split / subdir
            dst = subset_dir / split / subdir
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    # Write data.yaml
    yaml_data = {
        "path": str(subset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(EXPECTED_CLASSES),
        "names": EXPECTED_CLASSES,
    }
    yaml_path = subset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return yaml_path


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Ensure processed dataset exists
    if not PROCESSED_DIR.exists() or not (PROCESSED_DIR / "data.yaml").exists():
        print(
            "ERROR: Processed dataset not found.\n"
            "Run first:  python scripts/prepare_dataset.py"
        )
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device else auto_device()
    print(f"Device: {device}")

    if args.mode == "quick":
        epochs = args.epochs if args.epochs else 2
        imgsz = args.imgsz if args.imgsz else 320
        batch = args.batch if args.batch else 8
        subset_dir = ROOT / "data" / "quick_subset"
        print(f"\nQuick mode: {epochs} epochs, imgsz={imgsz}, batch={batch}")
        print("Building class-aware training subset (~1,200 images)...")
        data_yaml = build_quick_subset(PROCESSED_DIR, subset_dir, max_images=1200)
    else:
        epochs = args.epochs if args.epochs else 30
        imgsz = args.imgsz if args.imgsz else 640
        batch = args.batch if args.batch else 16
        data_yaml = PROCESSED_DIR / "data.yaml"
        print(f"\nFull mode: {epochs} epochs, imgsz={imgsz}, batch={batch}")

    output_dir = Path(args.output) if args.output else MODEL_DIR
    project_dir = output_dir / "runs"
    run_name = f"solarguard_{args.mode}"

    print(f"\nLoading YOLOv8n pretrained weights...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training -> {project_dir / run_name}")
    print("=" * 60)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
        # Class-imbalance mitigations
        cls=0.5,
        # Deterministic for quick mode
        seed=42,
        deterministic=(args.mode == "quick"),
        # Save artifacts
        save=True,
        save_period=1,
        plots=True,
        val=True,
        patience=10,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=1 if args.mode == "quick" else 3,
    )

    # Copy best.pt to models/best.pt for easy access
    best_src = project_dir / run_name / "weights" / "best.pt"
    best_dst = MODEL_DIR / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\nBest weights copied -> {best_dst}")
    else:
        print(f"WARNING: best.pt not found at {best_src}")

    # Save training metadata
    meta = {
        "mode": args.mode,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "data_yaml": str(data_yaml),
        "best_weights": str(best_dst),
        "run_dir": str(project_dir / run_name),
    }
    meta_path = MODEL_DIR / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved -> {meta_path}")
    print("\nTraining complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SolarGuard AI — model training")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"SolarGuard AI — Training  [mode={args.mode}]")
    print("=" * 60)
    train(args)
