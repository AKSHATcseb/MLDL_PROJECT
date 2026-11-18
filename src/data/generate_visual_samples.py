"""
Generates visual verification montages for segmentation datasets:
1. Original fundus image.
2. Canonical 3-class segmentation mask.
3. Image with semi-transparent mask overlay and boundary contours.
Saves 15 samples each for REFUGE and DRISHTI-GS.
"""

import os
import sys
import random
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
from src.data.mask_converter import convert_refuge_mask, convert_drishti_softmaps, create_mask_overlay

DATASET_ROOT = r'e:\MLDL MODEL TRAINING PROJECT\dataset'
OUTPUT_DIR = r'e:\MLDL MODEL TRAINING PROJECT\results\segmentation\visual_audit_samples'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_refuge_samples(num_samples: int = 15, seed: int = 42):
    random.seed(seed)
    train_img_dir = os.path.join(DATASET_ROOT, 'REFUGE2', 'train', 'images')
    train_mask_dir = os.path.join(DATASET_ROOT, 'REFUGE2', 'train', 'mask')
    
    all_imgs = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]
    selected = random.sample(all_imgs, min(num_samples, len(all_imgs)))
    
    print(f'Generating {len(selected)} visual audit samples for REFUGE...')
    for idx, f in enumerate(selected):
        img_p = os.path.join(train_img_dir, f)
        base_name = os.path.splitext(f)[0]
        mask_p = os.path.join(train_mask_dir, base_name + '.bmp')
        if not os.path.exists(mask_p):
            mask_p = os.path.join(train_mask_dir, base_name + '.png')
            
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raw_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        converted_mask = convert_refuge_mask(raw_mask)
        overlay = create_mask_overlay(img_rgb, converted_mask)
        
        # Color map for mask display: 0=black, 1=cyan, 2=red
        h, w = converted_mask.shape
        disp_mask = np.zeros((h, w, 3), dtype=np.uint8)
        disp_mask[converted_mask == 1] = [0, 200, 255] # Cyan (OD Rim)
        disp_mask[converted_mask == 2] = [255, 60, 60]  # Red (Optic Cup)
        
        # Crop zoom around Optic Disc region for detailed visualization if possible
        disc_y, disc_x = np.where(converted_mask >= 1)
        if len(disc_y) > 0:
            cy, cx = int(np.mean(disc_y)), int(np.mean(disc_x))
            crop_rad = max(int(np.max(disc_y) - np.min(disc_y)), int(np.max(disc_x) - np.min(disc_x))) * 2
            crop_rad = max(crop_rad, 200)
            ymin, ymax = max(0, cy - crop_rad), min(h, cy + crop_rad)
            xmin, xmax = max(0, cx - crop_rad), min(w, cx + crop_rad)
            zoom_img = img_rgb[ymin:ymax, xmin:xmax]
            zoom_mask = disp_mask[ymin:ymax, xmin:xmax]
            zoom_overlay = overlay[ymin:ymax, xmin:xmax]
        else:
            zoom_img, zoom_mask, zoom_overlay = img_rgb, disp_mask, overlay
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(zoom_img)
        axes[0].set_title(f'Original Fundus (Zoomed OD)\n{f}')
        axes[0].axis('off')
        
        axes[1].imshow(zoom_mask)
        axes[1].set_title('Ground Truth Mask\n[Cyan: Disc Rim, Red: Cup]')
        axes[1].axis('off')
        
        axes[2].imshow(zoom_overlay)
        axes[2].set_title('Mask Overlay with Contours')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'refuge_sample_{idx+1:02d}_{base_name}.png')
        plt.savefig(save_path, dpi=120)
        plt.close(fig)

def generate_drishti_samples(num_samples: int = 15, seed: int = 42):
    random.seed(seed)
    train_gt_dir = os.path.join(DATASET_ROOT, 'Training-20211018T055246Z-001', 'Training', 'GT')
    img_folders = [
        os.path.join(DATASET_ROOT, 'Training-20211018T055246Z-001', 'Training', 'Images', 'GLAUCOMA'),
        os.path.join(DATASET_ROOT, 'Training-20211018T055246Z-001', 'Training', 'Images', 'NORMAL')
    ]
    all_imgs = []
    for d in img_folders:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.png'):
                    all_imgs.append(os.path.join(d, f))
                    
    selected = random.sample(all_imgs, min(num_samples, len(all_imgs)))
    print(f'Generating {len(selected)} visual audit samples for DRISHTI-GS...')
    for idx, img_p in enumerate(selected):
        f = os.path.basename(img_p)
        sample_id = os.path.splitext(f)[0]
        cup_p = os.path.join(train_gt_dir, sample_id, 'SoftMap', f'{sample_id}_cupsegSoftmap.png')
        od_p = os.path.join(train_gt_dir, sample_id, 'SoftMap', f'{sample_id}_ODsegSoftmap.png')
        
        if not (os.path.exists(cup_p) and os.path.exists(od_p)):
            continue
            
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cup_soft = cv2.imread(cup_p, cv2.IMREAD_GRAYSCALE)
        od_soft = cv2.imread(od_p, cv2.IMREAD_GRAYSCALE)
        
        converted_mask = convert_drishti_softmaps(cup_soft, od_soft, threshold=128)
        overlay = create_mask_overlay(img_rgb, converted_mask)
        
        h, w = converted_mask.shape
        disp_mask = np.zeros((h, w, 3), dtype=np.uint8)
        disp_mask[converted_mask == 1] = [0, 200, 255] # Cyan (OD Rim)
        disp_mask[converted_mask == 2] = [255, 60, 60]  # Red (Optic Cup)
        
        disc_y, disc_x = np.where(converted_mask >= 1)
        if len(disc_y) > 0:
            cy, cx = int(np.mean(disc_y)), int(np.mean(disc_x))
            crop_rad = max(int(np.max(disc_y) - np.min(disc_y)), int(np.max(disc_x) - np.min(disc_x))) * 2
            crop_rad = max(crop_rad, 200)
            ymin, ymax = max(0, cy - crop_rad), min(h, cy + crop_rad)
            xmin, xmax = max(0, cx - crop_rad), min(w, cx + crop_rad)
            zoom_img = img_rgb[ymin:ymax, xmin:xmax]
            zoom_mask = disp_mask[ymin:ymax, xmin:xmax]
            zoom_overlay = overlay[ymin:ymax, xmin:xmax]
        else:
            zoom_img, zoom_mask, zoom_overlay = img_rgb, disp_mask, overlay
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(zoom_img)
        axes[0].set_title(f'Original Fundus (Zoomed OD)\n{f}')
        axes[0].axis('off')
        
        axes[1].imshow(zoom_mask)
        axes[1].set_title('Ground Truth Mask\n[Cyan: Disc Rim, Red: Cup]')
        axes[1].axis('off')
        
        axes[2].imshow(zoom_overlay)
        axes[2].set_title('Mask Overlay with Contours')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'drishti_sample_{idx+1:02d}_{sample_id}.png')
        plt.savefig(save_path, dpi=120)
        plt.close(fig)

if __name__ == '__main__':
    generate_refuge_samples(num_samples=15)
    generate_drishti_samples(num_samples=15)
    print(f'Done! All visual audit samples generated in {OUTPUT_DIR}')
