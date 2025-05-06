import numpy as np
from PIL import Image
# from data import ImagePreprocessor
import cv2

def compute_mae_np(x1, x2, mask=None):
    x1 = np.divide(x1, np.linalg.norm(x1, axis=2, keepdims=True) + 1.0e-12)
    x2 = np.divide(x2, np.linalg.norm(x2, axis=2, keepdims=True) + 1.0e-12)
    thresholds = [3, 5, 7.5, 11.5, 22.5, 30]
    if mask is not None:
        dot = np.sum(x1 * x2 * mask[:, :, np.newaxis], axis=-1, keepdims=True)
        dot = np.maximum(np.minimum(dot, np.array([1.0 - 1.0e-12])), np.array([-1.0 + 1.0e-12]))
        emap = np.abs(180 * np.arccos(dot) / np.pi) * mask[:, :, np.newaxis]
        mae = np.sum(emap) / np.sum(mask)
        median_ae = np.median(emap[mask > 0])
        accuracies = {f"acc_{t}": np.sum(emap[mask > 0] <= t) / np.sum(mask) for t in thresholds}
        return mae, median_ae, accuracies, emap
    else:
        dot = np.sum(x1 * x2, axis=-1, keepdims=True)
        dot = np.maximum(np.minimum(dot, np.array([1.0 - 1.0e-12])), np.array([-1.0 + 1.0e-12]))
        error = np.abs(180 * np.arccos(dot) / np.pi)
        return error

def edge_detection_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.uint8(gray)
    edges = cv2.Canny(gray, 50, 100)  # 0, 255 np.array
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    # mask = (edges > 0).astype(np.uint8)  # 0, 1
    return edges_dilated