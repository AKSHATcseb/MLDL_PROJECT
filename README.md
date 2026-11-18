# Glaucoma Detection Using Deep Learning
### Image Segmentation • Deep Classification • Explainable AI (XAI)

An end-to-end medical deep learning system for glaucoma screening and analysis across 5 clinical fundus photography datasets (REFUGE, ACRIMA, ORIGA, DRISHTI-GS, and RIM-ONE DL).

---

## 🔬 System Overview & Key Achievements

- **Optic Disc & Cup Semantic Segmentation**:
  - ResNet34 U-Net achieves **0.9546 Disc Dice** and **0.8645 Cup Dice** (**0.9096 Mean Dice**).
  - Attention U-Net achieves **0.8615 Mean Dice** with additive attention gates.
- **Glaucoma Deep Classification**:
  - EfficientNet-B0 achieves **98.57% Accuracy**, **97.14% Sensitivity**, **100.00% Specificity**, and **0.9988 ROC-AUC** on held-out ACRIMA test data.
  - ResNet50 achieves **98.57% Accuracy**, **98.57% Sensitivity**, and **0.9998 ROC-AUC**.
  - Dynamic class-imbalance weighting ($w=9.0$) on REFUGE handles 10% clinical prevalence (**0.9028 ROC-AUC**).
- **Explainable AI (XAI)**:
  - Grad-CAM & Grad-CAM++ feature hooks visualize neural attention on the neuroretinal rim and optic nerve head.
- **Geometric Clinical Biomarkers**:
  - Automatically calculates Vertical Cup-to-Disc Ratio (vCDR), Cup/Disc Area Ratios, and neuroretinal rim area.
  - Multi-modal feature fusion reveals vCDR provides cross-dataset invariance (**0.9549 ROC-AUC**).
- **Interactive Web Dashboard**:
  - Streamlit application integrating segmentation, classification, vCDR measurement, and Grad-CAM overlays.

---

## 📁 Project Directory Structure

```text
glaucoma-detection/
├── app/
│   └── app.py                     # Interactive Streamlit Web Application
├── checkpoints/
│   ├── segmentation/              # Best U-Net & Attention U-Net weights (.pth)
│   └── classification/            # Best EfficientNet & ResNet weights (.pth)
├── configs/
│   ├── dataset_config.yaml        # Dataset & Preprocessing parameters
│   ├── segmentation_config.yaml   # Segmentation hyperparameters
│   └── classification_config.yaml # Classification hyperparameters
├── data/
│   ├── processed/
│   │   ├── classification/        # Stratified train/val/test split manifests
│   │   └── segmentation/          # Canonical segmentation split manifests
│   └── raw/                       # Raw dataset root directories
├── documentation/
│   └── Glaucoma_Detection_Step_by_Step_Guide.pdf
├── reports/
│   ├── TECHNICAL_EXECUTION_PLAN.md
│   ├── DATASET_ANALYSIS_REPORT.md
│   ├── FINAL_RESEARCH_REPORT.md
│   └── unified_dataset_manifest.csv
├── results/
│   ├── classification/            # ROC, PR, Confusion Matrices & Logs
│   ├── segmentation/              # Learning curves & side-by-side visual montages
│   ├── explainability/            # Grad-CAM and Grad-CAM++ 4-panel visual figures
│   └── comparisons/               # Cross-dataset generalization matrices
├── src/
│   ├── data/
│   │   ├── augmentation.py        # Synchronous Albumentations transforms
│   │   ├── data_split.py          # Leak-free stratified partitioning
│   │   ├── dataset_analyzer.py    # Multi-dataset auditor
│   │   ├── dataset_loader.py      # PyTorch Dataset & DataLoader generators
│   │   ├── mask_converter.py      # Canonical {0: BG, 1: Disc, 2: Cup} converters
│   │   └── preprocessing.py       # Aspect-ratio letterbox resizing
│   ├── segmentation/
│   │   ├── unet.py                # ResNet34 U-Net
│   │   ├── attention_unet.py      # Additive Attention U-Net
│   │   ├── losses.py              # Combined CE + SoftDice loss
│   │   ├── train.py               # Segmentation trainer & evaluator
│   │   └── evaluate.py            # Comparative evaluation & visual montages
│   ├── classification/
│   │   ├── models.py              # EfficientNet-B0, ResNet50, DenseNet121
│   │   └── train.py               # Imbalance-aware 2-stage transfer learning
│   ├── explainability/
│   │   ├── gradcam.py             # Grad-CAM and Grad-CAM++ implementations
│   │   └── visualization.py       # XAI visual reporting suite
│   ├── integration/
│   │   ├── feature_extraction.py  # vCDR and Area biomarker extraction
│   │   ├── feature_fusion.py      # Multi-modal fusion research experiment
│   │   └── inference.py           # Unified multi-modal inference engine
│   └── evaluation/
│       ├── classification_metrics.py
│       ├── segmentation_metrics.py
│       └── cross_dataset.py       # Cross-dataset transfer benchmark
├── tests/                         # Complete automated pytest suite (12 tests)
├── main.py                        # Unified command-line interface
├── requirements.txt
└── PROJECT_STATUS.md              # Milestone tracking & hardware profile
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m pytest tests/
# Output: 12 passed in ~16s
```

### 3. Launch the Interactive Web Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to test fundus photographs, view optic disc/cup overlays, inspect vCDR, and generate Grad-CAM heatmaps.

### 4. Running Pipeline Stages via CLI (`main.py`)
```bash
# Audit datasets
python main.py --mode audit

# Train U-Net segmentation
python main.py --mode train_seg --arch unet --epochs 5 --batch_size 8

# Compare segmentation architectures
python main.py --mode eval_seg

# Train classification (EfficientNet-B0 on ACRIMA)
python main.py --mode train_cls --arch efficientnet_b0 --dataset acrima

# Generate Grad-CAM explainability reports
python main.py --mode eval_xai --arch efficientnet_b0 --dataset acrima

# Run cross-dataset generalization experiments
python main.py --mode cross_eval
```

---

## 📊 Summary of Experimental Results

### Segmentation Performance (REFUGE Held-Out Set)
- **ResNet34 U-Net**: **0.9546 Disc Dice**, **0.8645 Cup Dice** (**0.9096 Mean Dice**).
- **Attention U-Net**: **0.9107 Disc Dice**, **0.8124 Cup Dice** (**0.8615 Mean Dice**).

### Classification Performance (ACRIMA Held-Out Set)
- **EfficientNet-B0**: **98.57% Accuracy**, **97.14% Sensitivity**, **100% Specificity**, **0.9988 ROC-AUC**.
- **ResNet50**: **98.57% Accuracy**, **98.57% Sensitivity**, **98.57% Specificity**, **0.9998 ROC-AUC**.

### Cross-Dataset Domain Shift
- Image-only features dropped from **0.9988** in-domain to **0.6024** cross-clinic due to camera lighting/optical shifts.
- Geometric biomarker fusion (vCDR + Area Ratio) demonstrated strong cross-domain invariance (**0.9549 ROC-AUC**).

---

## ⚠️ Medical Screening Disclaimer
This software is an artificial intelligence research prototype developed for educational and experimental evaluation. It is not approved as a medical device and must not be used as a substitute for professional clinical diagnosis by a licensed ophthalmologist or optometrist.
