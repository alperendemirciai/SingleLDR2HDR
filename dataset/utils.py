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


def align_rotation(image1: torch.Tensor, image2: torch.Tensor, threshold: float = 0.8, patch_size: int = 100) -> tuple:
    def center_crop(img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        ch, cw = h // 2, w // 2
        half = size // 2
        return img[ch - half: ch + half, cw - half: cw + half]

    img1_np = image1.permute(1, 2, 0).cpu().numpy()
    img2_np = image2.permute(1, 2, 0).cpu().numpy()

    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

    # Crop center patches
    img1_patch = center_crop(img1_gray, patch_size)
    img2_patch = center_crop(img2_gray, patch_size)

    #win_size = min(7, patch_size)
    data_range = img1_patch.max() - img1_patch.min() or 1.0

    def compute_ssim(a, b):
        return ssim(a, b, win_size=7, data_range=data_range)
    #print(compute_ssim(img1_patch, img2_patch))
    # 1. Direct match
    if compute_ssim(img1_patch, img2_patch) > threshold:
        return image1, image2

    # 2. Flip-only check
    flip_variants = [
        cv2.flip(img2_np, 0),
        cv2.flip(img2_np, 1),
        cv2.flip(img2_np, -1)
    ]
    flip_patches = [center_crop(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), patch_size) for f in flip_variants]
    scores = []
    for fp in flip_patches:
        scores.append(compute_ssim(img1_patch, fp))

    if max(scores) > threshold:
        best_idx = int(np.argmax(scores))
        aligned = flip_variants[best_idx]
        return image1, torch.from_numpy(aligned).permute(2, 0, 1).float()

    # 3. Full search: rotation + flips
    rotations = [
        img2_np,
        cv2.rotate(img2_np, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img2_np, cv2.ROTATE_180),
        cv2.rotate(img2_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    transformations = []
    for rot in rotations:
        transformations.extend([
            rot,
            cv2.flip(rot, 0),
            cv2.flip(rot, 1),
            cv2.flip(rot, -1)
        ])

    gray_patches = [center_crop(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), patch_size) for t in transformations]
    scores = []
    for gp in gray_patches:
        scores.append(compute_ssim(img1_patch, gp))

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score < threshold:
        return image1, image2

    aligned_img2 = transformations[best_idx]
    aligned_img2_tensor = torch.from_numpy(aligned_img2).permute(2, 0, 1).float()

    return image1, aligned_img2_tensor
