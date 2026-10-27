"""
SolarGuard AI — Fault Detector
================================
Wraps Ultralytics YOLOv8 for solar-panel fault detection.

Model loading is lazy and cached.  If custom weights are not found,
a clear message is raised rather than falling back to COCO predictions.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

CLASS_NAMES: list[str] = [
    "MultiByPassed",
    "MultiDiode",
    "MultiHotSpot",
    "SingleByPassed",
    "SingleDiode",
    "SingleHotSpot",
    "StringOpenCircuit",
    "StringReversedPolarity",
]

_cached_model: Any = None
_cached_model_path: str = ""


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def find_best_weights() -> Path | None:
    """
    Search models/ for the best available custom weights.
    Returns the path of the newest 'best.pt' found, or None.
    """
    candidates = list(MODEL_DIR.rglob("best.pt"))
    if not candidates:
        # Also accept last.pt as a fallback
        candidates = list(MODEL_DIR.rglob("last.pt"))
    if not candidates:
        return None
    # Prefer the most recently modified file
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(weights_path: str | Path | None = None, conf: float = 0.25) -> Any:
    """
    Load and return a YOLOv8 model from custom weights.

    Raises
    ------
    FileNotFoundError
        If no custom weights exist in models/.
    ImportError
        If ultralytics is not installed.
    """
    global _cached_model, _cached_model_path

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed.  "
            "Run: pip install ultralytics"
        ) from exc

    if weights_path is None:
        weights_path = find_best_weights()

    if weights_path is None:
        raise FileNotFoundError(
            "No custom model weights found in models/.\n\n"
            "Train the model first:\n"
            "    python train_model.py\n\n"
            "This will run a quick prototype training (~2–3 epochs).\n"
            "See README.md for full training instructions."
        )

    weights_path = str(weights_path)

    if _cached_model is not None and _cached_model_path == weights_path:
        return _cached_model

    model = YOLO(weights_path)
    model.conf = conf
    _cached_model = model
    _cached_model_path = weights_path
    return model


def is_model_available() -> bool:
    """Return True if custom weights exist."""
    return find_best_weights() is not None


def get_model_info() -> dict[str, Any]:
    """Return metadata about the current model weights."""
    wp = find_best_weights()
    if wp is None:
        return {"available": False, "path": None, "modified": None}
    return {
        "available": True,
        "path": str(wp),
        "modified": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(wp.stat().st_mtime)
        ),
        "size_mb": round(wp.stat().st_size / 1_048_576, 2),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _pil_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image to RGB (handles grayscale, RGBA, palette, etc.)."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "P"):
        return image.convert("RGBA").convert("RGB")
    return image.convert("RGB")


def run_inference(
    image: Image.Image,
    conf_threshold: float = 0.25,
    weights_path: str | Path | None = None,
    imgsz: int = 640,
) -> dict[str, Any]:
    """
    Run the custom YOLOv8 detector on a PIL image.

    Returns
    -------
    dict with keys:
        detections  - list of detection dicts
        annotated_image - PIL Image with bounding boxes drawn
        inference_time_ms - float
        model_path - str
        error - str or None
    """
    try:
        model = load_model(weights_path, conf=conf_threshold)
    except (FileNotFoundError, ImportError) as exc:
        return {"detections": [], "annotated_image": image, "error": str(exc),
                "inference_time_ms": 0.0, "model_path": None}

    rgb_image = _pil_to_rgb(image)
    img_array = np.array(rgb_image)

    t0 = time.perf_counter()
    try:
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as exc:
        return {
            "detections": [],
            "annotated_image": image,
            "error": f"Inference failed: {exc}",
            "inference_time_ms": 0.0,
            "model_path": _cached_model_path,
        }
    inference_ms = (time.perf_counter() - t0) * 1000

    detections: list[dict] = []
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cid in zip(boxes_xyxy, confs, class_ids):
            cls_name = (
                CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            )
            detections.append(
                {
                    "class_name": cls_name,
                    "class_id": int(cid),
                    "confidence": float(conf),
                    "bbox": [float(v) for v in box],  # [x1, y1, x2, y2]
                }
            )

    # Build annotated image via ultralytics plot()
    try:
        annotated_array = result.plot()
        annotated_pil = Image.fromarray(annotated_array[..., ::-1])  # BGR→RGB
    except Exception:
        annotated_pil = rgb_image

    return {
        "detections": detections,
        "annotated_image": annotated_pil,
        "error": None,
        "inference_time_ms": round(inference_ms, 1),
        "model_path": _cached_model_path,
    }
