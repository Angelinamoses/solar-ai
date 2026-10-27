# SolarGuard AI

**Custom YOLOv8n solar panel fault detection with inspection history and risk prioritisation.**

> ⚠️ **Prototype / Research Tool** — All priorities, suggested actions, and review intervals are provisional demonstration rules.
> They must be verified by qualified solar engineers before any operational use.

---

## Project Purpose and Scope

SolarGuard AI detects eight categories of solar panel anomalies in drone thermal imagery using a custom-trained YOLOv8n model. It provides:

- Bounding-box fault detection across 8 fault classes
- Rule-based review priority assignment (Critical / High / Medium / Low)
- Persistent inspection history stored in a local SQLite database
- Detection comparison between separate uploads
- CSV export of detections and inspection records

**What it does not do:** temporal fault prediction, GPS mapping, calibrated temperature measurement, or autonomous drone control. See Section 9 for full dataset limitations.

---

## Fault Classes

| ID | Class | Default Priority |
|---|---|---|
| 0 | MultiByPassed | High |
| 1 | MultiDiode | High |
| 2 | MultiHotSpot | Critical |
| 3 | SingleByPassed | Medium |
| 4 | SingleDiode | Medium |
| 5 | SingleHotSpot | High |
| 6 | StringOpenCircuit | Critical |
| 7 | StringReversedPolarity | Critical |

---

## Prerequisites

- Windows 10/11
- Python 3.10 or later (tested with 3.13)
- Internet access for downloading pretrained YOLOv8n weights on first training run
- Dataset extracted to `data/ImageSet/` (see Section 5)

---

## Installation (Windows PowerShell)

### 1. Clone or download the project

```powershell
git clone <your-repo-url>
cd solar-ai
```

### 2. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell restricts execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Dataset Preparation

The dataset must be located at `data/ImageSet/` with the following structure:

```
data/
└── ImageSet/
    ├── data.yaml
    ├── train/images/  ← 6,924 JPEG files
    ├── train/labels/  ← 6,924 YOLO TXT files
    ├── valid/images/  ← 400 files
    ├── valid/labels/  ← 400 files
    ├── test/images/   ← 15 files
    └── test/labels/   ← 15 files
```

Run the preparation script to validate annotations and build a clean training copy:

```powershell
python scripts/prepare_dataset.py
```

This creates:
- `data/processed/` — cleaned dataset (156 malformed annotation lines excluded)
- `reports/annotation_audit.csv` — list of excluded lines
- `data/processed/dataset_summary.json` — machine-readable summary

**The original dataset is never modified.**

---

## Model Training

### Quick prototype (default — recommended first run)

Trains for 2 epochs on a class-aware subset of ~1,200 images. Takes 5–30 minutes depending on hardware.

```powershell
python train_model.py
```

### Full training

```powershell
python train_model.py --mode full --epochs 30
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `quick` | `quick` or `full` |
| `--epochs` | 2 (quick) / 30 (full) | Number of training epochs |
| `--batch` | 8 (quick) / 16 (full) | Batch size |
| `--imgsz` | 320 (quick) / 640 (full) | Image size |
| `--device` | auto | `cpu`, `0`, `0,1` |

GPU is used automatically if CUDA is available. CPU training is slower but functional.

After training, the best weights are saved to:

```
models/best.pt
```

---

## Launch the Dashboard

```powershell
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Running Tests

```powershell
pytest tests/ -v
```

Tests cover database operations, risk-rule evaluation, annotation validation, and report generation.
They do **not** require a trained model.

---

## Project Structure

```
solar-ai/
├── app.py                      Main Streamlit dashboard
├── train_model.py              Model training (quick & full)
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── risk_rules.json         Editable priority rules
├── src/
│   ├── __init__.py
│   ├── detector.py             YOLOv8 inference wrapper
│   ├── database.py             SQLite layer
│   ├── risk.py                 Risk prioritisation
│   └── reports.py              CSV report generation
├── scripts/
│   └── prepare_dataset.py      Dataset validation and cleaning
├── data/
│   ├── ImageSet/               Original dataset (not committed to Git)
│   ├── processed/              Cleaned copy (not committed)
│   ├── uploads/                Saved uploaded images (not committed)
│   └── outputs/                Saved annotated images (not committed)
├── models/                     Trained weights (not committed)
├── reports/                    Generated reports (not committed)
└── tests/
    └── test_solarguard.py
```

---

## Troubleshooting

### `No custom model weights found in models/`

Run training first:

```powershell
python train_model.py
```

### `ultralytics is not installed`

```powershell
pip install ultralytics
```

### PowerShell execution policy error

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Streamlit not found

```powershell
pip install streamlit
```

### CUDA / GPU errors

Add `--device cpu` to the training command:

```powershell
python train_model.py --device cpu
```

### `data.yaml not found` during training

Run dataset preparation first:

```powershell
python scripts/prepare_dataset.py
```

### Training is very slow on CPU

Use quick mode (default) with a small imgsz:

```powershell
python train_model.py --mode quick --imgsz 320 --epochs 2
```

---

## Dataset Limitations

This dataset has significant limitations that affect what the application can reliably demonstrate:

1. **Grayscale only** — thermal temperature values are converted to a grayscale rendering. Calibrated temperature readings are unavailable.
2. **No temporal data** — only two distinct flight dates (2023-08-09, 2024-04-11). No panel IDs link repeat inspections. Temporal fault prediction is not supported.
3. **Severe class imbalance (67:1)** — SingleHotSpot dominates (36.8%); StringReversedPolarity is rare (0.5%).
4. **Tiny test set** — 15 images, 5 of 8 classes. Not suitable for statistically meaningful evaluation.
5. **No GPS or panel IDs** — geographic mapping and panel tracking are not possible.
6. **Augmentation siblings** — each source image has ~3 augmented variants; split integrity should be verified.

---

## Responsible Use

- Do not use model outputs as the sole basis for maintenance decisions.
- All priorities and review intervals are provisional demonstration rules.
- "No fault detected" does not confirm that a panel is healthy.
- Model confidence is not a measure of fault severity.
- Verify all outputs with qualified solar engineers before operational deployment.
