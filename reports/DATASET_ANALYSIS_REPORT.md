# Comprehensive Medical Dataset Audit & Analysis Report
## Glaucoma Detection: Segmentation, Classification & Explainable AI
**Prepared by**: Lead Deep Learning Engineer & Medical CV Specialist  
**Date**: 2026-11-12  
**Dataset Directory**: `e:\MLDL MODEL TRAINING PROJECT\dataset`  

---

### 1. Executive Summary

A comprehensive multi-dataset audit was executed across all five datasets specified in the project requirements. Every individual image file was verified for I/O integrity, format specification, color channel representation, spatial dimensions, class labels, and segmentation mask alignments.

**Key Findings**:
- **Total Validated Retinal Images**: 3,141 fundus images.
- **Corrupted / Unreadable Images**: **0** (100% integrity across all files).
- **Segmentation Ground Truth Verified**: Available for **REFUGE** (1,200 masks) and **DRISHTI-GS** (101 soft maps from 4 experts).
- **Classification Ground Truth Verified**: Available across all 5 datasets: ACRIMA (705), ORIGA (650), RIM-ONE DL (485), REFUGE (400 train labelled), and DRISHTI-GS (101).
- **ACRIMA & ORIGA Segmentation Verification**: ACRIMA is strictly a classification dataset (0 masks). ORIGA contains classification labels in `ORIGA_train.csv` and `ORIGA_test.csv` (0 pixel masks in this release). As instructed, **no artificial masks will be synthesized for ACRIMA or ORIGA**.

---

### 2. Dataset-by-Dataset Audit Table

| Dataset | Total Images | Image Format | Typical Dimensions ($W \times H$) | Glaucoma | Normal | Unlabeled | Segmentation Ground Truth | Disc/Cup Format | Partitions | Missing/Corrupt |
|---|---|---|---|---|---|---|---|---|---|---|
| **REFUGE (REFUGE2)** | 1,200 | JPEG / BMP | $2124 \times 2056$, $1940 \times 1940$ | 40 (train) | 360 (train) | 800 (val/test) | 1,200 Masks | 8-bit Grayscale BMP/PNG: 0=Cup, 128=Disc, 255=BG | Train: 400<br>Val: 400<br>Test: 400 | 0 |
| **DRISHTI-GS** | 101 | PNG | $2045 \times 1752$, $2049 \times 1762$ | 70 | 31 | 0 | 101 SoftMaps | 4-expert soft maps: [0, 64, 128, 191, 255], boundary txt | Train: 50<br>Test: 51 | 0 |
| **ORIGA** | 650 | JPEG | $394 \times 394$, $421 \times 421$ | 168 | 482 | 0 | None (Classification) | N/A | Train: 454<br>Test: 196 | 0 |
| **ACRIMA** | 705 | JPEG | $1054 \times 1054$, $901 \times 901$ | 396 | 309 | 0 | None (Classification) | N/A | Train: 565<br>Test: 140 | 0 |
| **RIM-ONE DL** | 485 | PNG | $652 \times 652$, $551 \times 551$ | 172 | 313 | 0 | None (Classification) | N/A | Hospital & Random splits | 0 |

---

### 3. Deep Analysis by Dataset

#### 3.1 REFUGE (Retinal Fundus Glaucoma Challenge 2018/2020)
- **Clinical Description**: High-resolution digital retinal fundus photographs captured using Carl Zeiss Meditec Visucam 500 ($2124 \times 2056$) and Canon CR-2 ($1940 \times 1940$).
- **Classification Labels**:
  - Training partition: Filename prefix `g` ($n=40$) denotes confirmed primary open-angle glaucoma (POAG); prefix `n` ($n=360$) denotes healthy controls. Glaucoma prevalence in training set is 10.0%, reflecting realistic population screening prevalence.
  - Validation/Test partitions: Unlabeled for challenge ranking (`V0001`-`V0400`, `T0001`-`T0400`).
- **Segmentation Masks**:
  - Provided in `train/mask/` (400 BMPs), `val/mask/` (400 PNGs), `test/mask/` (400 BMPs).
  - Pixel values:
    - `0`: Optic Cup
    - `128`: Optic Disc rim
    - `255`: Retinal Background
  - Mask validation: Converted to canonical target (`0: Background, 1: Disc Rim, 2: Cup`). Verified nested containment ($\text{Cup} \subset \text{Disc}$).

#### 3.2 DRISHTI-GS
- **Clinical Description**: Captured at Aravind Eye Hospital, Madurai, India, centered on the optic nerve head (ONH) with a 30-degree field-of-view, $2045 \times 1752$ resolution.
- **Classification Labels**:
  - Training partition: 32 Glaucoma, 18 Normal (total 50).
  - Testing partition: 38 Glaucoma, 13 Normal (total 51).
  - Ground truth labels cataloged in `Notching___Image__level_decisions.xlsx`.
- **Segmentation Masks**:
  - Each sample folder contains soft maps from 4 ophthalmologists (`_cupsegSoftmap.png`, `_ODsegSoftmap.png`) with pixel intensities representing expert consensus ($0$: 0/4, $64$: 1/4, $128$: 2/4, $191$: 3/4, $255$: 4/4).
  - Standard majority consensus threshold ($\ge 128$) adopted.
  - Verified nested cup-inside-disc containment for all 101 subjects.

#### 3.3 ORIGA (-light)
- **Clinical Description**: Singapore Malay Eye Study (SiMES) dataset.
- **Classification Labels**:
  - Manifested via `ORIGA_train.csv` (454 samples, 117 Glaucoma / 337 Normal) and `ORIGA_test.csv` (196 samples, 51 Glaucoma / 145 Normal). Total: 168 Glaucoma, 482 Normal (25.8% prevalence).
- **Segmentation Annotations**:
  - Not included in this downloaded release. Utilized purely for classification and cross-dataset evaluation.

#### 3.4 ACRIMA
- **Clinical Description**: Captured at the Oftalvist clinic in Spain using an Imagenet fundus camera, centered on the optic disc.
- **Classification Labels**:
  - `train/Glaucoma` (326), `train/Non Glaucoma` (239).
  - `test/Glaucoma` (70), `test/Non Glaucoma` (70).
  - Total: 705 images (396 Glaucoma, 309 Normal).
- **Segmentation**: Treated strictly as classification-only; no artificial masks generated.

#### 3.5 RIM-ONE DL (Release)
- **Clinical Description**: Retinal Images for Optic Nerve Evaluation from Hospital Universitario de Canarias and other Spanish clinics.
- **Classification Labels**:
  - Partitioned by hospital: Train (116 Glaucoma, 195 Normal = 311), Test (56 Glaucoma, 118 Normal = 174). Total: 485 images.
  - Partitioned randomly: Train (120 Glaucoma, 219 Normal = 339), Test (52 Glaucoma, 94 Normal = 146).

---

### 4. Visual Verification Montages

Thirty visual audit montages were generated and cataloged at:
`results/segmentation/visual_audit_samples/`

- 15 Samples for REFUGE (`refuge_sample_01` to `refuge_sample_15`)
- 15 Samples for DRISHTI-GS (`drishti_sample_01` to `drishti_sample_15`)

Each montage illustrates:
1. Original high-resolution color fundus photograph (zoomed on ONH).
2. Ground-truth canonical mask (0: Black background, 1: Cyan Disc Rim, 2: Coral Cup).
3. Semi-transparent color mask overlay with precise contour boundaries (Yellow: Disc, Red: Cup).
