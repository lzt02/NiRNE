import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import os
import glob
from tqdm import tqdm
import re
from imageio import imread
import cv2
from utils import compute_mae_np, edge_detection_mask
from torch.cuda.amp import autocast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--output_dir", "-o", type=str, default="Normals", help="Path to save the rendered normals.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # normal_predictor = None
    normal_predictor = torch.hub.load("lzt02/NiRNE", "NiRNE", local_cache_dir='weights')

    image_paths = glob.glob(os.path.join(args.input_dir, "images", "*.png"))
    total_mae = 0
    total_smae = 0
    num_images = len(image_paths)

    for image_path in tqdm(image_paths, desc="Processing images"):
        
        filename = os.path.basename(image_path)
        # Construct the output path
        output_normal_path = os.path.join(args.output_dir, filename)  # Use args.output_dir and filename
        input_image = Image.open(image_path)

        # Generate mask path
        mask_path = image_path.replace("images", "mask")  # Replace folder name
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            print(f"Mask not found for: {image_path}")
            mask = None
        
        with autocast(dtype=torch.float16), torch.no_grad():
            pred_normal = normal_predictor(input_image)
        os.makedirs(os.path.dirname(output_normal_path), exist_ok=True)
        pred_normal.save(output_normal_path)
        pred_normal = np.array(pred_normal, dtype=np.float32) / 255.0
        pred_normal = 2 * pred_normal - 1  # [-1,1]
        
        # Calculate MAE
        gt_path = image_path.replace("images", "normals")
        gt_normal = Image.open(gt_path)

        gt_normal = gt_normal.convert('RGB')
        
        gt_normal = np.array(gt_normal, dtype=np.float32) / 255.0   # [0,1]
        gt_normal = 2 * gt_normal - 1   # [-1,1]
        input_image = np.array(input_image, dtype=np.float32) / 255.0
        input_image = 2 * input_image - 1
        
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = 2 * mask - 1
        if len(mask.shape) == 3:  # Convert to grayscale if mask is RGB
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype(np.float32)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        gt_normal_image = (gt_normal + 1) / 2 * 255
        sharpness_mask = edge_detection_mask(gt_normal_image)

        print(f"{pred_normal.shape}")
        print(f"{gt_normal.shape}")
        print(f"{mask.shape}")
        mae, _, _, error_map = compute_mae_np(pred_normal, gt_normal, mask)
        S_mae, _, _, S_error_map = compute_mae_np(pred_normal, gt_normal, mask * (sharpness_mask > 0).astype(np.uint8))
        total_mae += mae
        total_smae += S_mae

        print(f"Results of {filename}...")
        print(f"NE: {mae:.4f}, SNE: {S_mae:.4f}")

    # Output average NE and SNE
    avg_mae = total_mae / num_images
    avg_smae = total_smae / num_images
    print(f"Average NE: {avg_mae:.4f}, Average SNE: {avg_smae:.4f}")
