"""
Data Splitting & Partition Management.
Generates reproducible, leak-free train/validation/test CSV manifests for segmentation and classification.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'reports', 'unified_dataset_manifest.csv')
PROCESSED_SEG_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation')
PROCESSED_CLS_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification')

os.makedirs(PROCESSED_SEG_DIR, exist_ok=True)
os.makedirs(PROCESSED_CLS_DIR, exist_ok=True)

def build_segmentation_splits(manifest_df: pd.DataFrame, seed: int = 42):
    print("--- Building Segmentation Partitions ---")
    seg_manifest = manifest_df[manifest_df['has_mask'] == True].copy()
    
    # 1. REFUGE
    refuge_all = seg_manifest[seg_manifest['source_dataset'] == 'REFUGE'].copy()
    refuge_train = refuge_all[refuge_all['split'] == 'train'].copy()
    refuge_val = refuge_all[refuge_all['split'] == 'val'].copy()
    refuge_test = refuge_all[refuge_all['split'] == 'test'].copy()
    
    # For robust model selection during segmentation training, we create an 80/20 train/val split from the labelled 400 train set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, internal_val_idx = next(sss.split(refuge_train, refuge_train['label']))
    
    refuge_train_sub = refuge_train.iloc[train_idx].copy()
    refuge_val_sub = refuge_train.iloc[internal_val_idx].copy()
    
    refuge_train_sub.to_csv(os.path.join(PROCESSED_SEG_DIR, 'refuge_train_320.csv'), index=False)
    refuge_val_sub.to_csv(os.path.join(PROCESSED_SEG_DIR, 'refuge_val_80.csv'), index=False)
    refuge_train.to_csv(os.path.join(PROCESSED_SEG_DIR, 'refuge_official_train_400.csv'), index=False)
    refuge_val.to_csv(os.path.join(PROCESSED_SEG_DIR, 'refuge_official_val_400.csv'), index=False)
    refuge_test.to_csv(os.path.join(PROCESSED_SEG_DIR, 'refuge_official_test_400.csv'), index=False)
    
    print(f"REFUGE Segmentation: Train={len(refuge_train_sub)} (Glaucoma: {(refuge_train_sub['label']==1).sum()}), Val={len(refuge_val_sub)} (Glaucoma: {(refuge_val_sub['label']==1).sum()})")
    
    # 2. DRISHTI-GS
    drishti_all = seg_manifest[seg_manifest['source_dataset'] == 'DRISHTI-GS'].copy()
    drishti_train = drishti_all[drishti_all['split'] == 'train'].copy()
    drishti_test = drishti_all[drishti_all['split'] == 'test'].copy()
    
    drishti_train.to_csv(os.path.join(PROCESSED_SEG_DIR, 'drishti_train.csv'), index=False)
    drishti_test.to_csv(os.path.join(PROCESSED_SEG_DIR, 'drishti_test.csv'), index=False)
    print(f"DRISHTI-GS Segmentation: Train={len(drishti_train)}, Test={len(drishti_test)}")

def build_classification_splits(manifest_df: pd.DataFrame, seed: int = 42):
    print("\n--- Building Classification Partitions ---")
    cls_manifest = manifest_df[manifest_df['label'].notna()].copy()
    
    # 1. ACRIMA
    acrima = cls_manifest[cls_manifest['source_dataset'] == 'ACRIMA'].copy()
    acrima_train = acrima[acrima['split'] == 'train'].copy()
    acrima_test = acrima[acrima['split'] == 'test'].copy()
    
    # Validation split from training (20%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(acrima_train, acrima_train['label']))
    acrima_tr_split = acrima_train.iloc[tr_idx].copy()
    acrima_val_split = acrima_train.iloc[val_idx].copy()
    
    acrima_tr_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'acrima_train.csv'), index=False)
    acrima_val_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'acrima_val.csv'), index=False)
    acrima_test.to_csv(os.path.join(PROCESSED_CLS_DIR, 'acrima_test.csv'), index=False)
    print(f"ACRIMA: Train={len(acrima_tr_split)}, Val={len(acrima_val_split)}, Test={len(acrima_test)}")
    
    # 2. ORIGA
    origa = cls_manifest[cls_manifest['source_dataset'] == 'ORIGA'].copy()
    origa_train = origa[origa['split'] == 'train'].copy()
    origa_test = origa[origa['split'] == 'test'].copy()
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(origa_train, origa_train['label']))
    origa_tr_split = origa_train.iloc[tr_idx].copy()
    origa_val_split = origa_train.iloc[val_idx].copy()
    
    origa_tr_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'origa_train.csv'), index=False)
    origa_val_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'origa_val.csv'), index=False)
    origa_test.to_csv(os.path.join(PROCESSED_CLS_DIR, 'origa_test.csv'), index=False)
    print(f"ORIGA: Train={len(origa_tr_split)}, Val={len(origa_val_split)}, Test={len(origa_test)}")
    
    # 3. RIM-ONE DL
    rimone = cls_manifest[cls_manifest['source_dataset'] == 'RIM-ONE-DL'].copy()
    rim_train = rimone[rimone['split'] == 'train'].copy()
    rim_test = rimone[rimone['split'] == 'test'].copy()
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(rim_train, rim_train['label']))
    rim_tr_split = rim_train.iloc[tr_idx].copy()
    rim_val_split = rim_train.iloc[val_idx].copy()
    
    rim_tr_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'rimone_train.csv'), index=False)
    rim_val_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'rimone_val.csv'), index=False)
    rim_test.to_csv(os.path.join(PROCESSED_CLS_DIR, 'rimone_test.csv'), index=False)
    print(f"RIM-ONE DL: Train={len(rim_tr_split)}, Val={len(rim_val_split)}, Test={len(rim_test)}")
    
    # 4. REFUGE
    refuge_train = cls_manifest[(cls_manifest['source_dataset'] == 'REFUGE') & (cls_manifest['split'] == 'train')].copy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(refuge_train, refuge_train['label']))
    refuge_tr_split = refuge_train.iloc[tr_idx].copy()
    refuge_val_split = refuge_train.iloc[val_idx].copy()
    
    refuge_tr_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'refuge_train.csv'), index=False)
    refuge_val_split.to_csv(os.path.join(PROCESSED_CLS_DIR, 'refuge_val.csv'), index=False)
    print(f"REFUGE Classification: Train={len(refuge_tr_split)}, Val={len(refuge_val_split)}")

def main():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Run dataset_analyzer.py first to create {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    build_segmentation_splits(manifest)
    build_classification_splits(manifest)
    print("\nAll data partitions created and saved successfully.")

if __name__ == '__main__':
    main()
