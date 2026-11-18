"""
Dataset Analyzer and Auditor for Glaucoma Detection.
Performs data integrity verification, image dimension profiling, class distribution checks,
annotation verification, and builds dataset manifests.
"""

import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = r'e:\MLDL MODEL TRAINING PROJECT\dataset'
OUTPUT_DIR = r'e:\MLDL MODEL TRAINING PROJECT\reports'
RESULTS_DIR = r'e:\MLDL MODEL TRAINING PROJECT\results'

def audit_acrima():
    records = []
    splits = {'train': os.path.join(DATASET_ROOT, 'train'),
              'test': os.path.join(DATASET_ROOT, 'test')}
    
    for split_name, split_path in splits.items():
        if not os.path.exists(split_path):
            continue
        for cls_name, label in [('Glaucoma', 1), ('Non Glaucoma', 0)]:
            cls_dir = os.path.join(split_path, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, f)
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            mode = img.mode
                        records.append({
                            'image_id': f,
                            'source_dataset': 'ACRIMA',
                            'image_path': img_path,
                            'label': label,
                            'has_mask': False,
                            'mask_path': None,
                            'patient_id': f.split('_')[0] if '_' in f else None,
                            'split': split_name,
                            'width': w,
                            'height': h,
                            'channels': 3 if mode == 'RGB' else 1,
                            'corrupted': False,
                            'notes': 'Classification only'
                        })
                    except Exception as e:
                        records.append({
                            'image_id': f,
                            'source_dataset': 'ACRIMA',
                            'image_path': img_path,
                            'label': label,
                            'has_mask': False,
                            'mask_path': None,
                            'patient_id': None,
                            'split': split_name,
                            'width': None,
                            'height': None,
                            'channels': None,
                            'corrupted': True,
                            'notes': str(e)
                        })
    return pd.DataFrame(records)

def audit_origa():
    records = []
    base = os.path.join(DATASET_ROOT, 'ORIGA')
    for split in ['train', 'test']:
        csv_file = os.path.join(DATASET_ROOT, f'ORIGA_{split}.csv')
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            img_rel = row['ImageName'].replace('/', os.sep)
            img_path = os.path.join(DATASET_ROOT, img_rel)
            label = int(row['glaucoma'])
            f_name = os.path.basename(img_path)
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        mode = img.mode
                    records.append({
                        'image_id': f_name,
                        'source_dataset': 'ORIGA',
                        'image_path': img_path,
                        'label': label,
                        'has_mask': False,
                        'mask_path': None,
                        'patient_id': os.path.splitext(f_name)[0],
                        'split': split,
                        'width': w,
                        'height': h,
                        'channels': 3 if mode == 'RGB' else 1,
                        'corrupted': False,
                        'notes': 'Classification labels from ORIGA train/test CSV'
                    })
                except Exception as e:
                    records.append({
                        'image_id': f_name,
                        'source_dataset': 'ORIGA',
                        'image_path': img_path,
                        'label': label,
                        'has_mask': False,
                        'mask_path': None,
                        'patient_id': None,
                        'split': split,
                        'width': None,
                        'height': None,
                        'channels': None,
                        'corrupted': True,
                        'notes': str(e)
                    })
    return pd.DataFrame(records)

def audit_refuge():
    records = []
    base = os.path.join(DATASET_ROOT, 'REFUGE2')
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = os.path.join(base, split, 'images')
        mask_dir = os.path.join(base, split, 'mask')
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, f)
                base_name = os.path.splitext(f)[0]
                
                # REFUGE ground truth mask extension
                mask_path = None
                for ext in ['.bmp', '.png', '.jpg']:
                    candidate = os.path.join(mask_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                
                # Label: in train, g* = 1 (glaucoma), n* = 0 (normal)
                label = None
                if split == 'train':
                    label = 1 if f.startswith('g') else 0
                
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        mode = img.mode
                    records.append({
                        'image_id': f,
                        'source_dataset': 'REFUGE',
                        'image_path': img_path,
                        'label': label,
                        'has_mask': mask_path is not None,
                        'mask_path': mask_path,
                        'patient_id': base_name,
                        'split': split,
                        'width': w,
                        'height': h,
                        'channels': 3 if mode == 'RGB' else 1,
                        'corrupted': False,
                        'notes': f'REFUGE2 official challenge {split} split'
                    })
                except Exception as e:
                    records.append({
                        'image_id': f,
                        'source_dataset': 'REFUGE',
                        'image_path': img_path,
                        'label': label,
                        'has_mask': False,
                        'mask_path': None,
                        'patient_id': base_name,
                        'split': split,
                        'width': None,
                        'height': None,
                        'channels': None,
                        'corrupted': True,
                        'notes': str(e)
                    })
    return pd.DataFrame(records)

def audit_drishti_gs():
    records = []
    
    # 1. Training set
    train_img_dir = os.path.join(DATASET_ROOT, 'Training-20211018T055246Z-001', 'Training', 'Images')
    train_gt_dir = os.path.join(DATASET_ROOT, 'Training-20211018T055246Z-001', 'Training', 'GT')
    
    for cls_folder, label in [('GLAUCOMA', 1), ('NORMAL', 0)]:
        d = os.path.join(train_img_dir, cls_folder)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith('.png'):
                    img_path = os.path.join(d, f)
                    sample_id = os.path.splitext(f)[0]
                    # Check softmap
                    cup_soft = os.path.join(train_gt_dir, sample_id, 'SoftMap', f'{sample_id}_cupsegSoftmap.png')
                    od_soft = os.path.join(train_gt_dir, sample_id, 'SoftMap', f'{sample_id}_ODsegSoftmap.png')
                    has_masks = os.path.exists(cup_soft) and os.path.exists(od_soft)
                    
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            mode = img.mode
                        records.append({
                            'image_id': f,
                            'source_dataset': 'DRISHTI-GS',
                            'image_path': img_path,
                            'label': label,
                            'has_mask': has_masks,
                            'mask_path': os.path.join(train_gt_dir, sample_id, 'SoftMap'),
                            'patient_id': sample_id,
                            'split': 'train',
                            'width': w,
                            'height': h,
                            'channels': 3 if mode == 'RGB' else 1,
                            'corrupted': False,
                            'notes': 'Training split, 4-expert soft maps available'
                        })
                    except Exception as e:
                        pass
                        
    # 2. Test set
    test_img_dir = os.path.join(DATASET_ROOT, 'Test-20211018T060000Z-001', 'Test', 'Images')
    test_gt_dir = os.path.join(DATASET_ROOT, 'Test-20211018T060000Z-001', 'Test', 'Test_GT')
    for cls_folder, label in [('glaucoma', 1), ('normal', 0)]:
        d = os.path.join(test_img_dir, cls_folder)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith('.png'):
                    img_path = os.path.join(d, f)
                    sample_id = os.path.splitext(f)[0]
                    cup_soft = os.path.join(test_gt_dir, sample_id, 'SoftMap', f'{sample_id}_cupsegSoftmap.png')
                    od_soft = os.path.join(test_gt_dir, sample_id, 'SoftMap', f'{sample_id}_ODsegSoftmap.png')
                    has_masks = os.path.exists(cup_soft) and os.path.exists(od_soft)
                    
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            mode = img.mode
                        records.append({
                            'image_id': f,
                            'source_dataset': 'DRISHTI-GS',
                            'image_path': img_path,
                            'label': label,
                            'has_mask': has_masks,
                            'mask_path': os.path.join(test_gt_dir, sample_id, 'SoftMap'),
                            'patient_id': sample_id,
                            'split': 'test',
                            'width': w,
                            'height': h,
                            'channels': 3 if mode == 'RGB' else 1,
                            'corrupted': False,
                            'notes': 'Test split with ground truth soft maps'
                        })
                    except Exception as e:
                        pass
                        
    return pd.DataFrame(records)

def audit_rim_one():
    records = []
    base = os.path.join(DATASET_ROOT, 'RIM-ONE_DL_images', 'partitioned_by_hospital')
    for split, split_name in [('training_set', 'train'), ('test_set', 'test')]:
        split_dir = os.path.join(base, split)
        if not os.path.exists(split_dir):
            continue
        for cls_name, label in [('glaucoma', 1), ('normal', 0)]:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.png', '.jpg')):
                    img_path = os.path.join(cls_dir, f)
                    sample_id = os.path.splitext(f)[0]
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            mode = img.mode
                        records.append({
                            'image_id': f,
                            'source_dataset': 'RIM-ONE-DL',
                            'image_path': img_path,
                            'label': label,
                            'has_mask': False,
                            'mask_path': None,
                            'patient_id': sample_id,
                            'split': split_name,
                            'width': w,
                            'height': h,
                            'channels': 3 if mode == 'RGB' else 1,
                            'corrupted': False,
                            'notes': 'RIM-ONE DL hospital partition (classification)'
                        })
                    except Exception as e:
                        pass
    return pd.DataFrame(records)

def main():
    print('Starting systematic multi-dataset audit...')
    df_acrima = audit_acrima()
    print(f'ACRIMA audited: {len(df_acrima)} images')
    
    df_origa = audit_origa()
    print(f'ORIGA audited: {len(df_origa)} images')
    
    df_refuge = audit_refuge()
    print(f'REFUGE audited: {len(df_refuge)} images')
    
    df_drishti = audit_drishti_gs()
    print(f'DRISHTI-GS audited: {len(df_drishti)} images')
    
    df_rimone = audit_rim_one()
    print(f'RIM-ONE DL audited: {len(df_rimone)} images')
    
    all_manifest = pd.concat([df_refuge, df_drishti, df_origa, df_acrima, df_rimone], ignore_index=True)
    manifest_path = os.path.join(OUTPUT_DIR, 'unified_dataset_manifest.csv')
    all_manifest.to_csv(manifest_path, index=False)
    print(f'\nUnified manifest saved to: {manifest_path}')
    print(f'Total records in manifest: {len(all_manifest)}')
    
    # Summary Table
    summary = []
    for ds_name, grp in all_manifest.groupby('source_dataset'):
        total = len(grp)
        corrupted = grp['corrupted'].sum()
        with_mask = grp['has_mask'].sum()
        glaucoma = (grp['label'] == 1).sum()
        normal = (grp['label'] == 0).sum()
        unlabeled = grp['label'].isna().sum()
        dims = grp[['width', 'height']].drop_duplicates().to_dict('records')
        summary.append({
            'Dataset': ds_name,
            'Total Images': total,
            'Glaucoma': glaucoma,
            'Normal': normal,
            'Unlabeled (Test/Val)': unlabeled,
            'Usable Masks': with_mask,
            'Corrupted': corrupted,
            'Sample Dimensions (W x H)': str(dims[:2])
        })
    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, 'dataset_summary_statistics.csv')
    df_summary.to_csv(summary_path, index=False)
    print('\n=== DATASET AUDIT SUMMARY TABLE ===')
    print(df_summary.to_string(index=False))

if __name__ == '__main__':
    main()
