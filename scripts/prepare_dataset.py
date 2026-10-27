"""
SolarGuard AI — Dataset Preparation
======================================
Validates the raw dataset, removes malformed annotation lines, and
produces a clean copy in data/processed/ ready for training.

Usage
-----
    python scripts/prepare_dataset.py

Output
------
    data/processed/            Cleaned dataset tree
    data/processed/data.yaml   Fixed YAML with absolute paths
    reports/annotation_audit.csv   Malformed-line report
    data/processed/dataset_summary.json  Machine-readable summary
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "ImageSet"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

EXPECTED_CLASSES = [
    "MultiByPassed", "MultiDiode", "MultiHotSpot", "SingleByPassed",
    "SingleDiode", "SingleHotSpot", "StringOpenCircuit", "StringReversedPolarity",
]
NUM_CLASSES = len(EXPECTED_CLASSES)
SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_yolo_line(line: str) -> tuple[bool, str]:
    """
    Return (is_valid, reason).

    A valid YOLO line has exactly 5 whitespace-separated tokens:
        class_id (int, 0–N_classes-1)
        cx cy w h  (floats, all in (0, 1])
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Expected 5 fields, got {len(parts)}"
    try:
        cls_id = int(parts[0])
    except ValueError:
        return False, f"class_id is not an integer: '{parts[0]}'"
    if cls_id < 0 or cls_id >= NUM_CLASSES:
        return False, f"class_id {cls_id} out of range [0, {NUM_CLASSES - 1}]"
    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return False, f"Non-numeric coordinate in {parts[1:]}"
    cx, cy, w, h = coords
    if not (0.0 < cx <= 1.0 and 0.0 < cy <= 1.0):
        return False, f"cx/cy must be in (0,1]: got cx={cx}, cy={cy}"
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False, f"w/h must be in (0,1]: got w={w}, h={h}"
    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_dataset() -> dict:
    """Run the full preparation pipeline. Returns the summary dict."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    yaml_path = DATA_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get("nc", 0)
    class_names = cfg.get("names", [])
    print(f"Dataset YAML: nc={nc}, classes={class_names}")
    if nc != NUM_CLASSES or class_names != EXPECTED_CLASSES:
        print(
            f"WARNING: Expected {NUM_CLASSES} classes {EXPECTED_CLASSES}, "
            f"got nc={nc} names={class_names}"
        )

    # ---- scan ---------------------------------------------------------------
    audit_rows: list[dict] = []
    split_stats: dict[str, dict] = {}

    for split in SPLITS:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        labels = sorted(lbl_dir.glob("*.txt"))

        lbl_stems = {lbl.stem for lbl in labels}
        img_stems = {img.stem for img in images}

        orphan_imgs = img_stems - lbl_stems
        orphan_lbls = lbl_stems - img_stems
        empty_lbls = [lbl for lbl in labels if lbl.stat().st_size == 0]

        total_valid = 0
        total_bad = 0
        total_empty = len(empty_lbls)

        for lbl in labels:
            lines = lbl.read_text(encoding="utf-8", errors="replace").splitlines()
            for lineno, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                ok, reason = _validate_yolo_line(stripped)
                if ok:
                    total_valid += 1
                else:
                    total_bad += 1
                    audit_rows.append(
                        {
                            "split": split,
                            "label_file": lbl.name,
                            "line_number": lineno,
                            "raw_content": stripped[:120],
                            "reason": reason,
                        }
                    )

        split_stats[split] = {
            "images": len(images),
            "label_files": len(labels),
            "valid_annotation_lines": total_valid,
            "malformed_annotation_lines": total_bad,
            "empty_label_files": total_empty,
            "orphan_images": len(orphan_imgs),
            "orphan_labels": len(orphan_lbls),
        }
        print(
            f"  {split:5s}: {len(images)} images, {total_valid} valid annotations, "
            f"{total_bad} malformed, {total_empty} empty labels"
        )

    # ---- write audit CSV ----------------------------------------------------
    audit_csv_path = REPORTS_DIR / "annotation_audit.csv"
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "label_file", "line_number", "raw_content", "reason"]
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"\nAudit CSV written → {audit_csv_path} ({len(audit_rows)} malformed lines)")

    # ---- build processed copy -----------------------------------------------
    print(f"\nBuilding cleaned dataset → {PROCESSED_DIR}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    bad_files_set: set[str] = {row["label_file"] for row in audit_rows}

    for split in SPLITS:
        img_src = DATA_DIR / split / "images"
        lbl_src = DATA_DIR / split / "labels"
        img_dst = PROCESSED_DIR / split / "images"
        lbl_dst = PROCESSED_DIR / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        skipped_bad = 0

        for lbl_file in sorted(lbl_src.glob("*.txt")):
            stem = lbl_file.stem
            # Find corresponding image
            img_file = img_src / f"{stem}.jpg"
            if not img_file.exists():
                img_file = img_src / f"{stem}.png"
            if not img_file.exists():
                continue

            lines = lbl_file.read_text(encoding="utf-8", errors="replace").splitlines()
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                ok, _ = _validate_yolo_line(stripped)
                if ok:
                    clean_lines.append(stripped)

            # If all lines were bad and file was not originally empty, skip.
            orig_non_empty = any(l.strip() for l in lines)
            if orig_non_empty and not clean_lines:
                skipped_bad += 1
                continue

            # Copy image
            shutil.copy2(img_file, img_dst / img_file.name)
            # Write cleaned label
            dst_lbl = lbl_dst / lbl_file.name
            dst_lbl.write_text(
                "\n".join(clean_lines) + ("\n" if clean_lines else ""),
                encoding="utf-8",
            )
            copied_images += 1

        print(f"  {split:5s}: copied {copied_images} image/label pairs, skipped {skipped_bad} fully-invalid files")

    # ---- write fixed data.yaml -----------------------------------------------
    new_yaml = {
        "path": str(PROCESSED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": NUM_CLASSES,
        "names": EXPECTED_CLASSES,
    }
    new_yaml_path = PROCESSED_DIR / "data.yaml"
    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(new_yaml, f, default_flow_style=False, allow_unicode=True)
    print(f"\ndata.yaml written → {new_yaml_path}")

    # ---- write dataset summary -----------------------------------------------
    summary = {
        "source": str(DATA_DIR),
        "processed": str(PROCESSED_DIR),
        "splits": split_stats,
        "total_malformed_lines": len(audit_rows),
        "classes": EXPECTED_CLASSES,
        "num_classes": NUM_CLASSES,
        "audit_csv": str(audit_csv_path),
        "data_yaml": str(new_yaml_path),
    }
    summary_path = PROCESSED_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Dataset summary written → {summary_path}")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("SolarGuard AI — Dataset Preparation")
    print("=" * 60)
    summary = prepare_dataset()
    total_imgs = sum(s["images"] for s in summary["splits"].values())
    total_bad = summary["total_malformed_lines"]
    print(f"\nDone. {total_imgs} images validated. {total_bad} malformed annotation lines excluded.")
    print("Cleaned dataset ready for training.")
