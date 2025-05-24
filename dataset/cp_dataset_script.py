import os
import shutil
import numpy as np
import cv2
from utils import readHDR

def is_ldr_too_dark(ldr_path, brightness_threshold=40):
    image = cv2.imread(ldr_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not read LDR image: {ldr_path}")
        return True
    avg_brightness = np.mean(image)
    return avg_brightness < brightness_threshold

def is_hdr_too_dark(hdr_img, brightness_threshold=0.01):
    """
    Estimate brightness of HDR image using grayscale conversion:
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    """
    if hdr_img is None or hdr_img.ndim != 3 or hdr_img.shape[2] != 3:
        print("Warning: Invalid HDR image format.")
        return True
    luminance = (
        0.2126 * hdr_img[:, :, 0] +
        0.7152 * hdr_img[:, :, 1] +
        0.0722 * hdr_img[:, :, 2]
    )
    avg_brightness = np.mean(luminance)
    return avg_brightness < brightness_threshold

def filter_and_copy_dataset(
    src_root,
    dst_root,
    max_hdr_threshold=4095,
    ldr_brightness_threshold=40,
    hdr_brightness_threshold=0.01
):
    ldr_src = os.path.join(src_root, "LDR_in")
    hdr_src = os.path.join(src_root, "HDR_gt")

    ldr_dst = os.path.join(dst_root, "LDR_in")
    hdr_dst = os.path.join(dst_root, "HDR_gt")

    os.makedirs(ldr_dst, exist_ok=True)
    os.makedirs(hdr_dst, exist_ok=True)

    filenames = sorted([f for f in os.listdir(ldr_src) if f.endswith(".jpg")])

    total = len(filenames)
    kept = 0

    for filename in filenames:
        hdr_filename = filename.replace(".jpg", ".hdr")

        hdr_path = os.path.join(hdr_src, hdr_filename)
        ldr_path = os.path.join(ldr_src, filename)

        if not os.path.exists(hdr_path):
            print(f"Missing HDR: {hdr_filename}, skipping.")
            continue

        hdr_img = readHDR(hdr_path)

        max_hdr_val = np.max(hdr_img)
        if max_hdr_val >= max_hdr_threshold:
            print(f"Filtered out (HDR too bright): {filename} (max HDR = {max_hdr_val:.2f})")
            continue

        if is_hdr_too_dark(hdr_img, hdr_brightness_threshold):
            print(f"Filtered out (HDR too dark): {filename}")
            continue

        if is_ldr_too_dark(ldr_path, ldr_brightness_threshold):
            print(f"Filtered out (LDR too dark): {filename}")
            continue

        shutil.copy2(hdr_path, os.path.join(hdr_dst, hdr_filename))
        shutil.copy2(ldr_path, os.path.join(ldr_dst, filename))
        kept += 1

    print(f"Done. Kept {kept}/{total} image pairs.")

if __name__ == "__main__":
    src_dataset_path = "../data/HDR-Real"
    dst_dataset_path = "../data/HDR-Real-Filtered"

    filter_and_copy_dataset(src_dataset_path, dst_dataset_path)
