import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class HDRDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Args:
            root_dir (str): Path to root folder (contains numbered folders with input.png/output.png).
            split (str): One of "train", "val", or "test".
            transform (callable): Transform to apply on stacked input/output image.
            train_ratio (float): Proportion of samples for training.
            val_ratio (float): Proportion of samples for validation.
            seed (int): Random seed for reproducibility.
        """
        assert split in ["train", "val", "test"], f"Invalid split '{split}'"

        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Set seeds
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get all sample folders
        all_dirs = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Shuffle for reproducibility
        np.random.shuffle(all_dirs)

        # Split dataset
        n = len(all_dirs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if split == "train":
            self.sample_dirs = all_dirs[:train_end]
        elif split == "val":
            self.sample_dirs = all_dirs[train_end:val_end]
        else:
            self.sample_dirs = all_dirs[val_end:]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        input_path = os.path.join(sample_dir, "input.jpg")
        output_path = os.path.join(sample_dir, "ours.png")

        input_img = Image.open(input_path).convert("RGB")
        output_img = Image.open(output_path).convert("RGB")

        # Convert to tensors
        input_tensor = transforms.ToTensor()(input_img)
        output_tensor = transforms.ToTensor()(output_img)

        # Stack channel-wise: [6, H, W]
        stacked = torch.cat([input_tensor, output_tensor], dim=0)

        if self.transform:
            stacked = self.transform(stacked)

        # Unstack
        input_tensor = stacked[:3, :, :]
        output_tensor = stacked[3:, :, :]

        return input_tensor, output_tensor
