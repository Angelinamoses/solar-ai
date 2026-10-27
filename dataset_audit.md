# SolarGuard AI — Dataset Audit Report

**Generated:** 2026-10-27  
**Dataset path:** `data/ImageSet/`  
**Source:** Roboflow project `anomalies-detection-blv72 v14` (workspace: `solarpanels-odahn`)  
**License:** Private

---

## 1. Summary

| Property | Value |
|---|---|
| Total image files | **7,339** |
| Total label files | **7,339** |
| Uncompressed size (all images + labels) | **≈ 366.2 MB** |
| Image format | JPEG |
| Image dimensions | **640 × 640 px** (pre-resized, stretched) |
| Image colour space | **Grayscale** (CRT phosphor conversion applied) |
| Annotation format | **YOLO TXT** (class cx cy w h, normalised) |
| Number of classes | **8** |
| Splits | train / valid / test |
| Dataset creation date | 2024-08-24 |

---

## 2. Folder Structure

```
data/
└── ImageSet/
    ├── README.txt
    ├── data.yaml
    ├── train/
    │   ├── images/   (6,924 × .jpg)
    │   └── labels/   (6,924 × .txt)
    ├── valid/
    │   ├── images/   (400 × .jpg)
    │   └── labels/   (400 × .txt)
    └── test/
        ├── images/   (15 × .jpg)
        └── labels/   (15 × .txt)
```

**File extensions present:** `.jpg` (7,339), `.txt` (7,339). No other formats.  
**No archives, CSV, XML, COCO JSON, or segmentation masks** were found.

---

## 3. Dataset Splits

| Split | Images | Labels | Annotations | Size (images) |
|---|---|---|---|---|
| train | 6,924 | 6,924 | 19,845 | 345.1 MB |
| valid | 400 | 400 | 1,119 | 19.2 MB |
| test | 15 | 15 | 31 | 0.6 MB |
| **Total** | **7,339** | **7,339** | **20,995** | **≈ 366.2 MB** |

> [!WARNING]
> The **test set is critically small** — only 15 images and 31 annotations across just 5 of 8 classes. It cannot provide a statistically meaningful benchmark. Classes 0 (MultiByPassed) and 7 (StringReversedPolarity) are completely absent from the test set.

---

## 4. Image Characteristics

- **Dimensions:** All images are 640 × 640 px (Roboflow applied a stretch resize).
- **Colour space:** Grayscale. The README confirms a "CRT phosphor" conversion was applied, meaning the original thermal/IR imagery was linearised to a single luminance channel. **The raw thermal temperature data is not preserved** — pixel values represent a visual rendering of temperature, not calibrated readings.
- **Source equipment:** A significant portion of filenames follow the DJI drone naming pattern (e.g., `DJI_20230809115745_0013_T_jpg`), confirming **aerial drone capture**.
- **Thermal indicator (`_T_`):** 2,772 train files carry a `_T_` suffix, a DJI convention for thermal (IR) imagery. The remaining DJI files without `_T_` may be RGB visual images that were also converted to grayscale.

---

## 5. Augmentation and Source Image Count

The README states each source image was used to generate **3 augmented versions** using:
- 50% probability horizontal / vertical flip
- Random 90° rotation (none / CW / CCW / 180°)
- Random shear ±15°
- Random brightness ±25%

| Property | Value |
|---|---|
| Unique source image stems (train) | **2,081** |
| Total train images (after augmentation) | 6,924 |
| Effective augmentation factor | ≈ 3.3× |

> [!IMPORTANT]
> Because each source image produces ~3 near-identical siblings, the **effective training set size is ~2,081 unique scenes**, not 6,924. Standard random train/valid split risks data leakage if augmented siblings span splits. Verify the Roboflow split was done **before** augmentation (valid/test contain 400 and 15 images respectively and appear to be held-out source images).

---

## 6. Classes and Label Distribution

Classes are defined in `data.yaml`:

```yaml
names: ['MultiByPassed', 'MultiDiode', 'MultiHotSpot', 'SingleByPassed',
        'SingleDiode', 'SingleHotSpot', 'StringOpenCircuit', 'StringReversedPolarity']
```

### Annotation counts per class

| ID | Class | Train | Valid | Test | Total | % of Train |
|---|---|---|---|---|---|---|
| 0 | MultiByPassed | 357 | 34 | 0 | 391 | 1.8% |
| 1 | MultiDiode | 969 | 77 | 1 | 1,047 | 4.9% |
| 2 | MultiHotSpot | 5,081 | 209 | 7 | 5,297 | 25.6% |
| 3 | SingleByPassed | 2,978 | 263 | 11 | 3,252 | 15.0% |
| 4 | SingleDiode | 1,844 | 128 | 10 | 1,982 | 9.3% |
| 5 | SingleHotSpot | 7,311 | 323 | 2 | 7,636 | 36.8% |
| 6 | StringOpenCircuit | 1,197 | 69 | 0 | 1,266 | 6.0% |
| 7 | StringReversedPolarity | 108 | 16 | 0 | 124 | 0.5% |
| | **Total** | **19,845** | **1,119** | **31** | **20,995** | |

### Class imbalance analysis

```
SingleHotSpot          ████████████████████████████████████  7,311  (36.8%)
MultiHotSpot           ████████████████████                  5,081  (25.6%)
SingleByPassed         ████████████                          2,978  (15.0%)
SingleDiode            ███████                               1,844   (9.3%)
StringOpenCircuit      ████                                  1,197   (6.0%)
MultiDiode             ███                                     969   (4.9%)
MultiByPassed          █                                       357   (1.8%)
StringReversedPolarity ▏                                       108   (0.5%)
```

> [!WARNING]
> **Severe class imbalance.** The ratio between the most frequent class (SingleHotSpot: 7,311) and the least frequent (StringReversedPolarity: 108) is **67.7 : 1**. Without mitigation (focal loss, class-weighted sampling, or oversampling), a naïve model will overwhelmingly predict hot-spot classes and ignore bypassed diode and string faults.

---

## 7. Annotation Quality

### Format validation
- All label files use YOLO TXT format: `class_id cx cy w h` (5 space-separated floats per line).
- **156 malformed annotation lines** were found in the train set (lines not containing exactly 5 fields). This is **0.79% of all 19,845 annotation lines**. These should be inspected and either corrected or removed before training.

### Missing / empty labels

| Split | Images missing a label file | Empty label files (background) |
|---|---|---|
| train | 0 | **7** |
| valid | 0 | 0 |
| test | 0 | 0 |

The 7 empty train label files are valid "background" (negative) examples — no annotation is the correct annotation — but they should be confirmed as intentional rather than annotation omissions.

### Duplicate files
- **0 duplicate filenames** detected across train / valid / test splits. ✓

### Sample annotations (raw)

```
# File: -_png_jpg.rf.77c8b49...txt
1 0.43035 0.71443 0.09522 0.41819   # MultiDiode
1 0.57928 0.70375 0.10131 0.41528   # MultiDiode
2 0.55915 0.25989 0.09547 0.42288   # MultiHotSpot

# File: 0266_jpg.rf.00e3e6cd...txt
4 0.61306 0.12023 0.08662 0.04156   # SingleDiode
5 0.86372 0.22367 0.08514 0.04621   # SingleHotSpot
2 0.86444 0.27215 0.08516 0.04777   # MultiHotSpot
```

Bounding boxes look plausible — cx/cy/w/h are all in [0, 1], and widths/heights are reasonable for individual panel cells within a 640×640 drone image.

---

## 8. Metadata: Temporal, Spatial, and Panel Identity

| Metadata type | Present? | Notes |
|---|---|---|
| Panel IDs | No | Not in filenames or any sidecar file |
| GPS coordinates | No | Not in filenames; EXIF stripped during preprocessing |
| Structured inspection dates | No | Only implicit in some DJI filenames |
| Temperature readings | No | Original thermal data converted to grayscale; no calibrated values |
| Historical inspection records | No | No multi-visit data per panel |
| Acquisition timestamps | Partial | DJI filenames embed datetime (e.g., `DJI_20230809115745` = 2023-08-09 11:57:45). Only **two distinct flight dates** are visible: **2023-08-09** and **2024-04-11**. |
| Drone model | Partial | DJI naming convention confirmed; specific model unknown |

> [!CAUTION]
> Only **two distinct capture dates** span the entire dataset (Aug 2023 and Apr 2024). Many filenames lack any date at all (e.g., `0266_jpg`, `-_png_jpg`). There are **no panel IDs linking the same panel across visits**, and no structured time-series records. **Temporal fault analysis and future fault prediction are not supported** by this dataset in any rigorous sense.

---

## 9. Computer-Vision Task Assessment

| Task | Supported? | Reason |
|---|---|---|
| **Object Detection** | YES — primary task | YOLO TXT bounding boxes, 8 classes, multiple objects per image |
| Image Classification (whole-image) | Possible with cropping | Would require cropping each annotated region; multi-label per image complicates this |
| Instance / Semantic Segmentation | No | No polygon masks or pixel-level labels |
| Temporal fault tracking | No | No panel IDs, no repeated inspections of the same panel |
| Future fault prediction | No | Single-visit snapshot data only; no longitudinal records |
| Anomaly / novelty detection | Limited | Possible if treating unlabelled patches as "normal", but no normal-panel baseline is provided |

**Conclusion:** The dataset is a straightforward **multi-class object detection** dataset annotated in YOLO TXT format. Its natural application is detecting fault types within a drone thermal image of a solar farm.

---

## 10. Key Limitations

1. **Grayscale only** — thermal temperature gradients survive as luminance, but calibrated temperature values are lost. No original RAW or 16-bit thermal data.
2. **No temporal capability** — two flight dates and no panel identity mean no fault progression analysis is possible.
3. **Extreme class imbalance (67:1)** — rare faults (StringReversedPolarity, MultiByPassed) will be underfit without explicit countermeasures.
4. **Tiny test set** — 15 images, 31 annotations, 5 of 8 classes. Not fit for final model evaluation; do not report final metrics from it.
5. **Augmentation leakage risk** — if Roboflow augmented before splitting, augmented siblings of validation/test images may be in training, causing optimistic metrics.
6. **156 malformed annotation lines (~0.79%)** — small but non-zero; must be cleaned before training.
7. **7 empty train label files** — may be intentional negatives or annotation omissions; confirm before training.
8. **Stretched resize** — original aspect ratios are distorted (640×640 stretch). Panel shapes may be geometrically inaccurate.
9. **No normal-panel class** — the model can only detect anomalies; it cannot explicitly confirm a panel as healthy.
10. **Single source** — Roboflow project v14; unknown how diverse the underlying farm sites are.

---

## 11. Recommended Baseline Model

Given the actual dataset characteristics — YOLO TXT bounding boxes, 640×640 grayscale images, 8 classes, ~2,000 unique source scenes — the recommended baseline is:

### YOLOv8-Nano (`yolov8n`) or YOLOv8-Small (`yolov8s`)

**Rationale:**
- The annotation format is native YOLO — zero conversion overhead.
- 640×640 is the default YOLOv8 inference resolution.
- Grayscale input is handled by setting `channels=1` or converting images to 3-channel at load time (standard practice).
- `yolov8n` (~3.2M parameters) is fast enough to prototype on CPU; `yolov8s` adds ~5 pp mAP with modest extra cost.
- Pre-trained COCO weights provide useful low-level feature initialisation even for thermal imagery.

**Required mitigations for class imbalance:**
- Use `cls` loss weight overrides for minority classes, or
- Apply focal loss (built-in to Ultralytics), or
- Oversample images containing `StringReversedPolarity` (class 7) and `MultiByPassed` (class 0).

**Alternative if detection is later simplified to per-panel classification:**
- Crop annotated regions then train an **EfficientNet-B0** image classifier — much simpler, fewer hyperparameters, faster to iterate. Only viable if panels are pre-localised.

---

## 12. Recommended Next Steps

1. **Fix malformed annotations** — inspect the 156 lines with non-5 fields; decide whether to correct or remove.
2. **Confirm empty label files** — verify the 7 empty train label files are intentional background images, not missed annotations.
3. **Verify augmentation split integrity** — confirm Roboflow split was performed *before* augmentation to prevent data leakage.
4. **Rebuild a larger test set** — hold out a stratified sample (50–100 images per class) from training before any model is run, to enable honest evaluation of all 8 classes.
5. **Establish class-imbalance strategy** — choose focal loss or oversampling for minority classes before first training run.
6. **Prototype with YOLOv8n** — single-GPU/CPU friendly, native YOLO TXT support, fast iteration cycle.
7. **Do not attempt temporal analysis** — the dataset does not contain the metadata required. If this feature is needed, a new dataset with panel IDs and multi-visit records must be sourced or constructed.
8. **Consider re-exporting in 3-channel grayscale** — some model backbones expect RGB input; replicate the single channel to all 3 to use pre-trained weights without architecture changes.

---

*Audit performed by automated inspection of filenames, label files, and metadata. No images were loaded into memory. No modifications were made to the original dataset.*
