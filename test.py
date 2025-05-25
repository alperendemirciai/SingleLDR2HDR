import models.model_a
import models.model_b
import models.unet
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np

from utils.argparser import get_test_args
from utils.utils import calculate_psnr, save_some_examples_plots, load_checkpoint, save_some_examples
from models.custom_loss import HDRLossWithLPIPS
from dataset.custom_dataset import HDRRealDataset
from torchmetrics.functional import structural_similarity_index_measure

@torch.no_grad()
def test():
    args = get_test_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f"Using device: {device}")

    # Model selection
    if args.model == 'unet':
        generator = models.unet.UNet(in_channels=3, out_channels=3, base_filters=16, upsampling_method=args.upsampling_method).to(device)
    elif args.model == 'model_a':
        generator = models.model_a.Autoencoder(upsample_mode=args.upsampling_method).to(device)
    elif args.model == 'model_b':
        generator = models.model_b.MobileNetV3_UNet(3).to(device)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(args.save_dir, "checkpoints", "gen_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    load_checkpoint(checkpoint_path, generator,device= device)

    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
    ])

    test_dataset = HDRRealDataset(
        root_dir=args.data,
        split="test",
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        random_seed=args.random_state,
        transforms=transform,
        normalize_imgnet1k=True
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    generator.eval()

    test_pixel_loss = 0
    test_psnr_total = 0
    test_ssim_total = 0

    custom_loss = HDRLossWithLPIPS(lpips_weight=2, l1_weight=3)

    loop = tqdm(test_loader, desc="Testing")

    for batch in loop:
        input_img = batch["ldr_nonlinear_01"].to(device) * 2.0 - 1.0
        target_img = batch["hdr_log_01"].to(device) * 2.0 - 1.0

        gen_output = generator(input_img)
        loss = custom_loss(gen_output, target_img)

        psnr_val = calculate_psnr(gen_output, target_img)
        ssim_val = structural_similarity_index_measure(gen_output, target_img, data_range=2.0)

        test_pixel_loss += loss.item() * input_img.size(0)
        test_psnr_total += psnr_val.item() * input_img.size(0)
        test_ssim_total += ssim_val.item() * input_img.size(0)

        loop.set_postfix(loss=loss.item(), PSNR=psnr_val.item(), SSIM=ssim_val.item())

    test_pixel_loss /= len(test_dataset)
    test_psnr_avg = test_psnr_total / len(test_dataset)
    test_ssim_avg = test_ssim_total / len(test_dataset)

    print(f"\nTest Results:")
    print(f"Pixel Loss: {test_pixel_loss:.4f}")
    print(f"PSNR: {test_psnr_avg:.4f}")
    print(f"SSIM: {test_ssim_avg:.4f}")

    # Save a few output samples
    
    save_some_examples(generator, test_loader, epoch="test", folder=os.path.join(args.save_dir, f"test_results_{args.random_state}"), 
                       device=device, denorm=False, num_samples=args.save_n, imgnet_denorm=( test_dataset.normalize_imgnet1k))
    save_some_examples_plots(generator, test_loader, epoch="test", folder=os.path.join(args.save_dir, f"test_results_{args.random_state}"), 
                             device=device, denorm=False, num_samples=args.save_n, imgnet_denorm=( test_dataset.normalize_imgnet1k))

if __name__ == '__main__':
    test()
