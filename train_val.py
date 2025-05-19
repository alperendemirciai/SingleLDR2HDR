import models.model_a
import models.model_b
import models.unet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import numpy as np

from utils.argparser import get_train_val_args
from utils.utils import *

import models
from models.custom_loss import HDRLossWithLPIPS
from dataset.custom_dataset import HDRRealDataset

from torchmetrics.functional import structural_similarity_index_measure

def train_validate():
    args = get_train_val_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    os.makedirs(args.save_dir, exist_ok=True)

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f"Using device: {device}")

    if args.model == 'unet':
        generator = models.unet.UNet(in_channels=3, out_channels= 3, base_filters= 16, upsampling_method= args.upsampling_method).to(device)
    elif args.model == 'model_a':
        generator = models.model_a.Autoencoder(upsample_mode=args.upsampling_method).to(device)
    elif args.model == 'model_b':
        generator = models.model_b.MobileNetV3_UNet(3).to(device)


    optimizer_G = optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    custom_loss = HDRLossWithLPIPS(lpips_weight=2)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    train_dataset = HDRRealDataset(
        root_dir=args.data,
        split="train",
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        random_seed=args.random_state,
        transforms=transform
    )

    val_dataset = HDRRealDataset(
        root_dir=args.data,
        split="val",
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        random_seed=args.random_state,
        transforms=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    history = {
        'train_pixel_loss': [],
        'val_pixel_loss': [],
        'val_psnr': [],
        'val_ssim': [],
    }

    best_val_psnr = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        generator.train()
        train_pixel_loss = 0

        loop = tqdm(train_loader, leave=True, desc=f"Training {epoch+1}/{args.epochs}")

        for batch in loop:
            input_img = batch["ldr_nonlinear_01"].to(device)*2 - 1   # input
            # Normalize target_img to [-1, 1]
            target_img = batch["hdr_log_01"].to(device) * 2 - 1


            optimizer_G.zero_grad()
            gen_output = generator(input_img)

            combined_loss = custom_loss(gen_output, target_img)
            combined_loss.backward()
            optimizer_G.step()

            train_pixel_loss += combined_loss.item() * input_img.size(0)

            loop.set_postfix(pixel_loss=combined_loss.item())

        train_pixel_loss /= len(train_dataset)
        history['train_pixel_loss'].append(train_pixel_loss)

        # Validation
        generator.eval()
        val_pixel_loss = 0
        val_psnr_total = 0
        val_ssim_total = 0

        loop = tqdm(val_loader, leave=True, desc=f"Validation {epoch+1}/{args.epochs}")
        with torch.no_grad():
            for batch in loop:
                input_img = batch["ldr_nonlinear_01"].to(device)*2 - 1   # input
                # Normalize target_img to [-1, 1]
                target_img = batch["hdr_log_01"].to(device) * 2 - 1

                gen_output = generator(input_img)
                combined_loss = custom_loss(gen_output, target_img)

                psnr_val = calculate_psnr(gen_output, target_img)
                ssim_val = structural_similarity_index_measure(gen_output, target_img, data_range=2.0)

                val_pixel_loss += combined_loss.item() * input_img.size(0)
                val_psnr_total += psnr_val.item() * input_img.size(0)
                val_ssim_total += ssim_val.item() * input_img.size(0)

                loop.set_postfix(pixel_loss=combined_loss.item(), PSNR=psnr_val.item(), SSIM=ssim_val.item())

        val_pixel_loss /= len(val_dataset)
        val_psnr_avg = val_psnr_total / len(val_dataset)
        val_ssim_avg = val_ssim_total / len(val_dataset)

        history['val_pixel_loss'].append(val_pixel_loss)
        history['val_psnr'].append(val_psnr_avg)
        history['val_ssim'].append(val_ssim_avg)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Pixel Loss: {train_pixel_loss:.4f}")
        print(f"Validation Pixel Loss: {val_pixel_loss:.4f} | PSNR: {val_psnr_avg:.4f} | SSIM: {val_ssim_avg:.4f}")

        if val_psnr_avg > best_val_psnr:
            best_val_psnr = val_psnr_avg
            best_epoch = epoch + 1
            os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)
            save_checkpoint(generator, optimizer_G, epoch + 1,
                            os.path.join(args.save_dir, "checkpoints", "gen_best.pth"))
            print(f"Saved best model at epoch {epoch+1} with PSNR: {best_val_psnr:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            os.makedirs(os.path.join(args.save_dir, "results"), exist_ok=True)
            save_some_examples(generator, val_loader, epoch + 1,
                               folder=os.path.join(args.save_dir, "results"),
                               device=device, denorm=False)

    print(f"Training completed. Best model at epoch {best_epoch} with PSNR: {best_val_psnr:.4f}")

    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)
    plot_metrics(
        history['train_pixel_loss'],
        history['val_pixel_loss'],
        'Pixel Loss',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    plot_metric(
        history['val_psnr'],
        'Validation PSNR',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )
    plot_metric(
        history['val_ssim'],
        'Validation SSIM',
        os.path.join(args.save_dir, "plots"),
        args.epochs
    )



if __name__ == '__main__':
    train_validate()
