import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional, Tuple
import random
import numpy as np
from utils import *
import cv2
import sys

from dataset.utils import readHDR, log_minmax, align_rotation

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EvalDataset(Dataset):
    """
    PyTorch Dataset for HDR-to-LDR Image Translation.

    Supports train/val/test split with reproducibility, 
    HDR preprocessing (log/minmax normalization),
    and torchvision transforms.
    """

    def __init__(
        self,
        root_dir: str,
        random_seed: int = 42,
        transforms: Optional[Callable] = None,
        normalize_imgnet1k: bool = True
    ) -> None:
        """
        Args:
            root_dir (str): Root directory of HDR_Real dataset.
            split (str): One of ['train', 'val', 'test'].
            split_ratios (Tuple[float, float, float]): Ratios for train, val, and test splits.
            random_seed (int): Seed for reproducible random split.
            transform_LDR (Callable, optional): Transformations for LDR images.
            transform_HDR (Callable, optional): Transformations for HDR images.
            hdr_preprocessing (str, optional): HDR preprocessing method: ['minmax', 'log_minmax', None].
        """
        super().__init__()
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.random_seed = random_seed
        self.normalize_imgnet1k = normalize_imgnet1k


        # List and sort files
        self.filenames = sorted([
            f for f in os.listdir(self.root_dir) 
            if not f.endswith(".DS_Store")
        ])

        # Split dataset
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Shuffle filenames for reproducibility
        random.shuffle(self.filenames)        

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (LDR_image, HDR_image)
        """

        filename = self.filenames[idx]
        
        ldr_path = os.path.join(self.root_dir, filename, "input.jpg")
        hdr_path = os.path.join(self.root_dir, filename, "gt.hdr")

        # Read images
        ldr_img = cv2.imread(ldr_path, cv2.IMREAD_COLOR)  # uint8
        ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        hdr_img = readHDR(hdr_path)  # float32

        #print("Min HDR:", np.min(hdr_img))
        #print("Max HDR:", np.max(hdr_img))

        # HDR Preprocessing
        hdr_img_log = log_minmax(hdr_img)
        hdr_log_tensor = transforms.ToTensor()(hdr_img_log)
        
        # LDR Preprocessing
        ldr_img = ldr_img.astype(np.float32)
        ldr_img = ldr_img / 255.0
        ldr_img = np.clip(ldr_img, 0, 1)

        ldr_img_tensor = transforms.ToTensor()(ldr_img)  
        
        # Normalize LDR image if required
        if self.normalize_imgnet1k:
            ldr_img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(ldr_img_tensor)
        
        #print("LDR image shape: ",ldr_img_tensor.shape)
        #print("HDR image shape: ",hdr_log_tensor.shape)
        # Check if images are aligned
        

        ldr_img_tensor, hdr_log_tensor = align_rotation(ldr_img_tensor, hdr_log_tensor)

        # create a stack stensor with size [2, 3, H, W]
        stack = torch.stack([ldr_img_tensor, hdr_log_tensor], dim=0)
        #print("Stack shape:", stack.shape)

        # make stack to [6, H, W]
        stack = stack.view(2*3, stack.shape[2], stack.shape[3])
        #print("Stack shape after view:", stack.shape)
        
        # Apply transformations if provided
        if self.transforms:
            stack = self.transforms(stack)
            ldr_img_tensor = stack[:3]
            hdr_log_tensor = stack[3:]
        
        return_dict = {
            "ldr_nonlinear_01": ldr_img_tensor,
            "hdr_log_01": hdr_log_tensor,
            "file_name": filename,
        }
        return return_dict
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    
    torch_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),])


    dataset = HDRRealDataset(
        root_dir="../data/HDR-Real/",
        split="train",
        transforms=torch_transforms,
        split_ratios=(0.7, 0.2, 0.1),
        normalize_imgnet1k=False,
        random_seed=20
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        ldr_img = batch["ldr_nonlinear_01"][0].numpy().transpose(1, 2, 0)
        hdr_img = batch["hdr_log_01"][0].numpy().transpose(1, 2, 0)
        file_name = batch["file_name"][0]
        print(f"File: {file_name}")

        plt.subplot(1, 2, 1)
        plt.imshow(ldr_img)
        plt.title("LDR Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(hdr_img)
        plt.title("HDR Image")
        plt.axis("off")

        plt.show()
        break