import os
from typing import List
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim


def readHDR(img_path: str) -> np.ndarray:
    """Reads a high dynamic range (HDR) image (.hdr) as float32."""
    assert os.path.exists(img_path), f"Image path {img_path} does not exist"
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    assert img.dtype == np.float32, "HDR image should be loaded as float32"
    return img

def writeHDR(img: np.ndarray, img_path: str) -> None:
    """Writes a high dynamic range (HDR) image (.hdr) as float32."""
    if os.path.exists(img_path):
        print(f"Warning: {img_path} already exists. Overwriting.")
    else:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    assert img.dtype == np.float32, "Image must be of type float32"
    assert img.ndim == 3, "Image must be a 3D array (H, W, C)"
    assert img.shape[2] == 3, "Image must have 3 channels (RGB)"
    cv2.imwrite(img_path, img.astype(np.float32))

def minmax_norm(img: np.ndarray) -> np.ndarray:
    """Normalizes an image to the range [0, 1]."""
    min_val = np.min(img)
    max_val = np.max(img)
    assert min_val != max_val, "Image has no variation (min == max)"
    img = img.astype(np.float32)
    img = (img - min_val) / (max_val - min_val)
    return np.clip(img, 0, 1)

def log_minmax(img: np.ndarray) -> np.ndarray:
    """Applies log transformation and then min-max normalization to an image."""
    assert img.dtype == np.float32, "Image must be of type float32"
    img = np.log1p(img)
    return minmax_norm(img)

def normalize_mean_std(img: np.ndarray, means: List[float], stds: List[float]) -> np.ndarray:
    """Normalizes an image using channel-wise mean and standard deviation."""
    assert img.dtype == np.float32, "Image must be of type float32"
    img = (img - means) / stds
    return img

def align_rotation(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 0.8) -> tuple:
    img1_np = image1.permute(1, 2, 0).cpu().numpy()
    img2_np = image2.permute(1, 2, 0).cpu().numpy()
    
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    
    min_size = min(img1_gray.shape[:2])
    win_size = min(7, min_size)

    data_range = img1_gray.max() - img1_gray.min()
    if data_range == 0:
        data_range = 1.0

    def compute_ssim(a, b):
        return ssim(a, b, win_size=win_size, data_range=data_range)

    if compute_ssim(img1_gray, img2_gray) > threshold:
        return image1, image2

    rotations = [
        img2_np,
        cv2.rotate(img2_np, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img2_np, cv2.ROTATE_180),
        cv2.rotate(img2_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]
    
    gray_rotations = [cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) for r in rotations]
    scores = [compute_ssim(img1_gray, r) for r in gray_rotations]
    
    best_idx = int(np.argmax(scores))
    aligned_img2 = rotations[best_idx]
    
    aligned_img2_tensor = torch.from_numpy(aligned_img2).permute(2, 0, 1).float()
    return image1, aligned_img2_tensor
