# TECHNICAL EXECUTION PLAN
## Glaucoma Detection Using Deep Learning: Image Segmentation + Classification + Explainable AI

---

### 1. Project Architecture & Modular Design

The system is architected as an end-to-end, reproducible, medical research pipeline consisting of three core clinical AI tasks and an integrated multi-modal inference application:

```
[Retinal Fundus Image]
          │
          ▼
┌───────────────────────────────┐
│ 1. Data Harmonization Pipeline │
│   - Aspect ratio preservation │
│   - Consistent resizing       │
│   - Standard normalization    │
└──────────────┬────────────────┘
               │
       ┌───────┴───────────────────────┐
       ▼                               ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ 2. Semantic Segmentation    │  │ 3. Glaucoma Classification   │
│   - Pretrained ResNet34     │  │   - EfficientNet-B0 (Primary)│
│     U-Net / Attention U-Net │  │   - ResNet50 & DenseNet121   │
│   - Multi-class Mask:       │  │   - Binary Probability Logit │
│     0: Background           │  │   - Class Imbalance Handling │
│     1: Optic Disc Rim       │  └──────────────┬──────────────┘
│     2: Optic Cup            │                 │
└──────────────┬──────────────┘                 │
               │                                │
               ▼                                ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ 4. Anatomical Biomarkers    │  │ 5. Explainable AI (XAI)     │
│   - Vertical CDR (vCDR)     │  │   - Grad-CAM Heatmaps       │
│   - Area Cup/Disc Ratio     │  │   - Grad-CAM++ Maps         │
│   - Clinical ISNT alignment │  │   - Plausibility Validation │
└──────────────┬──────────────┘  └──────────────┬──────────────┘
               │                                │
               └───────────────┬────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Integrated Clinical Prototype & Streamlit Dashboard       │
│   - Simultaneous mask overlay, risk score & Grad-CAM visual  │
│   - Strict Medical Screening Research Disclaimer             │
└──────────────────────────────────────────────────────────────┘
```

The codebase is organized into isolated, reusable modules:
- `src/data/`: Data loading, manifest construction, mask format converters, deterministic splitting, and data integrity tests.
- `src/segmentation/`: Custom and library U-Net and Attention U-Net architectures, hybrid Dice + Cross-Entropy loss functions, trainers, and boundary evaluators.
- `src/classification/`: Transfer learning models (EfficientNet-B0, ResNet50, DenseNet121), two-stage training routines, and class-weighted loss strategies.
- `src/explainability/`: Grad-CAM and Grad-CAM++ gradient hook extractors, heatmap normalization, and visual overlay synthesizers.
- `src/integration/`: Unified inference pipeline calculating geometric cup-to-disc ratios (vCDR) alongside image-level diagnostic predictions.
- `src/evaluation/`: Metric suites (per-class Dice, IoU, sensitivity, specificity, ROC-AUC, PR-AUC, confusion matrices) and cross-dataset testing.
- `app/`: Streamlit web application providing interactive upload, segmentation, classification, and explanation inspection.

---

### 2. Dataset Access and Storage Requirements

The datasets are stored in `e:\MLDL MODEL TRAINING PROJECT\dataset`:
1. **DRISHTI-GS**: 101 fundus images with 4-expert soft maps and boundary coordinates for Optic Disc and Cup; image-level diagnostic labels.
2. **REFUGE (REFUGE2)**: 1,200 images total (Train: 400 with 40 glaucoma, 360 normal; Val: 400; Test: 400). Ground truth masks encoded with pixel values 0 (Cup), 128 (Disc), and 255 (Background).
3. **ORIGA**: 650 images (Train: 454, Test: 196) with verified glaucoma/normal ground truth classification labels.
4. **RIM-ONE DL**: 485 images organized into hospital and random partitions with verified glaucoma/normal labels.
5. **ACRIMA**: 705 images (396 glaucoma, 309 normal) with dedicated train/test partitions. ACRIMA is treated strictly as a classification dataset.

**Harmonized Canonical Segmentation Format**:
- `0`: Background
- `1`: Optic Disc Rim (Disc minus Cup)
- `2`: Optic Cup (nested innermost structure)

---

### 3. Available Computational Resources & Optimization Strategy

- **Host OS**: Windows 11 (64-bit).
- **CPU**: 12 Logical Cores (Intel 12th/13th Gen).
- **RAM**: 7.73 GB Total (~1.0 GB immediately available).
- **GPU**: Intel(R) UHD Graphics (Integrated; discrete NVIDIA CUDA unavailable).
- **Storage**: ~39 GB free on Drive E:.

**Computational Optimization Tactics**:
1. **Memory-Conscious Image Resolution**:
   - Segmentation: 384×384 (or 256×256 during initial rapid baselines) to prevent RAM exhaustion and preserve anatomical features.
   - Classification: 224×224 standard ImageNet resolution.
2. **Batch Sizing & PyTorch Multi-threading**:
   - Batch size: 4 to 8 for segmentation, 16 for classification.
   - DataLoader `num_workers`: 0 or 2 (Windows multi-processing memory overhead avoided), with `torch.set_num_threads(8)`.
   - Streaming generator data loaders (no in-memory caching of raw 2000×2000 TIFFs/BMPs).
3. **Training Iterations**:
   - Transfer learning using pretrained ImageNet backbones (ResNet34 encoder for U-Net, EfficientNet-B0 / ResNet50 for classification).
   - Early stopping on validation metrics (patience = 7-10 epochs) with CosineAnnealingLR or ReduceLROnPlateau.

---

### 4. Model Implementation Strategy

#### A. Segmentation Models
- **Baseline Architecture**: U-Net with ResNet34 encoder pretrained on ImageNet.
- **Advanced Architecture**: Attention U-Net incorporating additive attention gates to focus on optic nerve head features and suppress peripheral fundus noise.
- **Loss Function**: Hybrid loss $\mathcal{L}_{total} = \alpha \mathcal{L}_{CE} + (1-\alpha) \mathcal{L}_{Dice}$, where $\mathcal{L}_{Dice}$ explicitly computes foreground Dice for Disc Rim (Class 1) and Cup (Class 2), excluding background dominance.

#### B. Classification Models
- **Baseline Architecture**: EfficientNet-B0 (lightweight, parameter-efficient, optimal for medical transfer learning).
- **Comparative Architectures**: ResNet50 and DenseNet121.
- **Training Routine**: Two-stage transfer learning:
  - Stage 1: Freeze backbone, train classification head (1e-3 LR, 5 epochs).
  - Stage 2: Unfreeze top layers, fine-tune end-to-end (1e-4 LR with weight decay 1e-4, 15-25 epochs).
- **Imbalance Mitigation**: Positive class weighting via `pos_weight` in `BCEWithLogitsLoss` calculated directly from dataset training proportions.

#### C. Explainability (XAI)
- **Technique**: Grad-CAM and Grad-CAM++ implemented on the final convolutional layer of each backbone (e.g. `layer4` in ResNet50, `features[-1]` in EfficientNet).
- **Visual Validation**: Heatmap alignment inspection against segmented Optic Disc and Cup regions.

---

### 5. Training and Evaluation Strategy

#### A. Data Leakage Prevention
- Strict patient-level separation where IDs exist; zero cross-contamination of test or validation sets.
- Augmentation (mild rotation, affine, contrast, brightness, flips) applied strictly to training batches only.
- Validation sets used solely for threshold selection, learning rate scheduling, and early stopping.

#### B. Evaluation Protocol
- **Segmentation**: Per-class Dice Coefficient and Intersection over Union (IoU) evaluated separately for Optic Disc and Optic Cup. Mean ± Std reported.
- **Classification**: Sensitivity (Recall for Glaucoma), Specificity, Precision, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
- **Operating Threshold**: Picked on validation ROC curve (Youden's J statistic or equalized sensitivity/specificity), never on the test set.
- **Cross-Dataset Generalization**:
  - REFUGE -> ORIGA
  - ACRIMA -> RIM-ONE DL / REFUGE
  - Document domain shift characteristics (contrast, field-of-view, camera differences).

---

### 6. Expected Deliverables

1. **Modular Source Code**: Full directory structure with clean Python modules (`src/`, `configs/`, `tests/`).
2. **Preprocessed Manifests & Data**: Standardized CSV manifests and verified converted masks.
3. **Trained Checkpoints**: Best U-Net, Attention U-Net, EfficientNet-B0, and ResNet50 model weights.
4. **Evaluation Artifacts**: Metrics tables, ROC curves, PR curves, confusion matrices, and segmentation overlays.
5. **Interactive UI**: Working Streamlit application with live inference, mask display, vCDR, Grad-CAM overlay, and medical disclaimer.
6. **Documentation & Research Report**: Final technical report, README, and reproducible scripts.

---

### 7. Potential Technical Risks & Mitigation

| Identified Risk | Severity | Mitigation Strategy |
|---|---|---|
| **High Memory Usage on Windows** | High | Load images lazily, use 256/384px resolution, keep batch sizes modest (4-8), run garbage collection between epochs. |
| **CPU Training Latency** | Medium | Utilize transfer learning with frozen backbones, lightweight architectures (EfficientNet-B0, ResNet34), and early stopping. |
| **Annotation Inconsistency Across Datasets** | High | Write isolated mask converters with automated unit tests verifying nested containment (Cup $\subset$ Disc) and pixel values {0, 1, 2}. |
| **Severe Class Imbalance (REFUGE: 10% glaucoma)** | Medium | Use `pos_weight` in BCE loss, compute PR-AUC, and tune operating decision threshold on validation set. |
| **Grad-CAM Focus on Peripheral Artifacts** | Low | Verify heatmap bounding boxes against segmented optic disc mask; document failure modes transparently. |
