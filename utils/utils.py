import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import cv2
import torchvision.transforms.functional as TF



def calculate_ssim(img1, img2, data_range=2.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        data_range: Range of the image data (default: 1.0)
        
    Returns:
        ssim_val: SSIM value
    """
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    return ssim(img1_np, img2_np, data_range=data_range, multichannel=False)



def save_checkpoint(model, optimizer, epoch, filename):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        filename: File path for saving checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filepath, model, optimizer=None, lr=None, device=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load parameters into
        optimizer: PyTorch optimizer to load state into (optional)
        lr: Learning rate to set (optional)
        device: Device to load the model on (optional)
        
    Returns:
        epoch: The epoch number of the loaded checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at {filepath}")
        return 0
        
    if device is None:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    return checkpoint['epoch']


def save_image(img_array, filepath):
    """
    Save a NumPy image array to a file
    
    Args:
        img_array: NumPy array of shape [H, W, C] with values in [0, 1]
        filepath: Path to save the image
    """
    # Convert to uint8
    img_array = (img_array + 1.0)/2.0
    img_array = (img_array * 255).astype(np.uint8)
    
    # Handle different channel numbers
    if img_array.shape[2] == 1:
        img = Image.fromarray(img_array[:, :, 0], 'L')
    elif img_array.shape[2] == 3:
        img = Image.fromarray(img_array, 'RGB')
    elif img_array.shape[2] == 4:
        img = Image.fromarray(img_array, 'RGBA')
    else:
        # For other channel numbers, save first three channels as RGB
        img = Image.fromarray(img_array[:, :, :3], 'RGB')
    
    img.save(filepath)

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum value of the image (default: 1.0)
        
    Returns:
        psnr_val: PSNR value
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def save_some_examples_plots(gen, val_loader, epoch, folder, device, num_samples=4, denorm=True, imgnet_denorm=True):
    """
    Save a grid of sample outputs from the generator
    
    Args:
        gen: Generator model
        val_loader: DataLoader for validation set
        epoch: Current epoch number
        folder: Folder to save images
        device: Device to run inference on
        num_samples: Number of samples to visualize
        denorm: Whether to denormalize images from [-1, 1] to [0, 1]
    """
    os.makedirs(folder, exist_ok=True)
    
    batch = next(iter(val_loader))

    if isinstance(batch, dict):
        x_b = batch["ldr_nonlinear_01"] * 2 - 1
        y_b = batch["hdr_log_01"] * 2 - 1
    else:
        raise ValueError("Expected dict from val_loader, got something else.")
        
    x = x_b[:num_samples].clone().to(device)
    y = y_b[:num_samples].clone().to(device)

    #print("X_maxmin: ", x.max(), x.min())

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        gen.train()

        # Rescale from [-1, 1] to [0, 1] if needed
        x = (x + 1.0) / 2.0
        if imgnet_denorm:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            x = x * imagenet_std + imagenet_mean

        y = (y + 1.0) / 2.0
        y_fake = (y_fake + 1.0) / 2.0

        y = torch.flip(y, dims=[1])        # Flips along channel axis
        y_fake = torch.flip(y_fake, dims=[1])

        # Visualization grids
        x_grid = make_grid(x, nrow=num_samples, normalize=True, value_range=(-1, 1) if denorm else None)
        y_grid = make_grid(y, nrow=num_samples, normalize=True, value_range=(-1, 1) if denorm else None)
        y_fake_grid = make_grid(y_fake, nrow=num_samples, normalize=True, value_range=(-1, 1) if denorm else None)

        # Exp versions (HDR)
        y_exp = torch.exp(y) + 1.0
        y_fake_exp = torch.exp(y_fake) + 1.0

        y_exp_grid = make_grid(y_exp, nrow=num_samples, normalize=True, value_range=(0, 1) if denorm else None)
        y_fake_exp_grid = make_grid(y_fake_exp, nrow=num_samples, normalize=True, value_range=(0, 1) if denorm else None)

        # Convert to numpy for plotting
        x_grid = x_grid.cpu().numpy().transpose(1, 2, 0)
        y_grid = y_grid.cpu().numpy().transpose(1, 2, 0)
        y_fake_grid = y_fake_grid.cpu().numpy().transpose(1, 2, 0)
        y_exp_grid = y_exp_grid.cpu().numpy().transpose(1, 2, 0)
        y_fake_exp_grid = y_fake_exp_grid.cpu().numpy().transpose(1, 2, 0)

        # Create figure with subplots
        fig, axs = plt.subplots(5, 1, figsize=(15, 15))
        
        axs[0].imshow(x_grid)
        axs[0].set_title(f"Input RGB - Epoch {epoch}")
        axs[0].axis('off')
        
        axs[1].imshow(y_grid)
        axs[1].set_title(f"Target log HDR - Epoch {epoch}")
        axs[1].axis('off')
        
        axs[2].imshow(y_fake_grid)
        axs[2].set_title(f"Generated log HDR - Epoch {epoch}")
        axs[2].axis('off')

        axs[3].imshow(y_exp_grid)
        axs[3].set_title(f"Target HDR - Epoch {epoch}")
        axs[3].axis('off')

        axs[4].imshow(y_fake_exp_grid)
        axs[4].set_title(f"Generated HDR - Epoch {epoch}")
        axs[4].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"epoch_{epoch}.png"))
        plt.close()


def writeHDR(img: np.ndarray, img_path: str) -> None:
    """Writes a high dynamic range (HDR) image (.hdr) as float32."""
    if os.path.exists(img_path):
        print(f"Warning: {img_path} already exists. Overwriting.")
    else:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
    assert img.dtype == np.float32, "Image must be of type float32"
    assert img.ndim == 3, "Image must be a 3D array (H, W, C)"
    assert img.shape[2] == 3, "Image must have 3 channels (RGB)"
    
    success = cv2.imwrite(img_path, img)
    if not success:
        raise IOError(f"Failed to write HDR image to {img_path}")

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def save_some_examples(gen, val_loader, epoch, folder, device, num_samples=4, denorm=True, imgnet_denorm=True):
    os.makedirs(folder, exist_ok=True)
    batch = next(iter(val_loader))

    if isinstance(batch, dict):
        x_b = batch["ldr_nonlinear_01"] * 2.0 - 1.0
        y_b = batch["hdr_log_01"] * 2.0 - 1.0
    else:
        raise ValueError("Expected dict from val_loader, got something else.")

    x = x_b[:num_samples].clone().to(device)
    y = y_b[:num_samples].clone().to(device)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        x = (x + 1.0) / 2.0
    gen.train()

    if imgnet_denorm:
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        x = x * imagenet_std + imagenet_mean

    y = torch.flip(y, dims=[1])        # RGB to BGR
    y_fake = torch.flip(y_fake, dims=[1])

    for i in range(num_samples):
        input_img = x[i].cpu()
        target_img = y[i].cpu()
        pred_img = y_fake[i].cpu()

        target_img = (target_img + 1.0) / 2.0
        pred_img = (pred_img + 1.0) / 2.0

        triplet_folder = os.path.join(folder, f"img{i+1}_epoch{epoch}")
        os.makedirs(triplet_folder, exist_ok=True)

        # Save input/target/pred PNG
        TF.to_pil_image(input_img).save(os.path.join(triplet_folder, "input.png"))
        TF.to_pil_image(target_img).save(os.path.join(triplet_folder, "target.png"))
        TF.to_pil_image(pred_img).save(os.path.join(triplet_folder, "prediction.png"))

        # Recover HDR from log space
        target_hdr = torch.exp(target_img) + 1.0
        pred_hdr = torch.exp(pred_img) + 1.0

        # Convert to NumPy
        target_hdr_np = target_hdr.numpy().transpose(1, 2, 0).astype(np.float32)
        pred_hdr_np = pred_hdr.numpy().transpose(1, 2, 0).astype(np.float32)

        # Save HDR .hdr files
        writeHDR(target_hdr_np, os.path.join(triplet_folder, "target.hdr"))
        writeHDR(pred_hdr_np, os.path.join(triplet_folder, "prediction.hdr"))

        # Reinhard tonemapping
        def apply_reinhard_tonemap(hdr_np):
            tonemap = cv2.createTonemapReinhard(gamma=1.0, intensity=0.0, light_adapt=1.0, color_adapt=0.0)
            ldr = tonemap.process(hdr_np)
            ldr = np.clip(ldr, 0, 1)
            return (ldr * 255).astype(np.uint8)

        reinhard_target = apply_reinhard_tonemap(target_hdr_np)
        reinhard_pred = apply_reinhard_tonemap(pred_hdr_np)

        tonemapl01_target = tone_mapping_l1_l0_approx(target_hdr_np)
        tonemapl01_pred = tone_mapping_l1_l0_approx(pred_hdr_np)

        # Save Reinhard tonemapped images
        Image.fromarray(reinhard_target).save(os.path.join(triplet_folder, "target_reinhard.png"))
        Image.fromarray(reinhard_pred).save(os.path.join(triplet_folder, "prediction_reinhard.png"))
        
        # Save Reinhard tonemapped images
        Image.fromarray(tonemapl01_target).save(os.path.join(triplet_folder, "target_l0l1.png"))
        Image.fromarray(tonemapl01_pred).save(os.path.join(triplet_folder, "prediction_l0l1.png"))

        # Compute PSNR and SSIM
        psnr_val = psnr(reinhard_target/255.0, reinhard_pred/255.0, data_range=1)
        ssim_val = ssim(reinhard_target/255.0, reinhard_pred/255.0, channel_axis=2, data_range=1)

        # Plot and save comparison
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(np.array(TF.to_pil_image(input_img)))
        axs[0].set_title("Input")
        axs[1].imshow(reinhard_pred)
        axs[1].set_title("Predicted Reinhard")
        axs[2].imshow(reinhard_target)
        axs[2].set_title("GT Reinhard")

        for ax in axs:
            ax.axis('off')

        fig.suptitle(f"Epoch {epoch} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}", fontsize=14)
        #plt.tight_layout()
        plt.savefig(os.path.join(triplet_folder, "comparison.png"))
        plt.close()

def plot_metrics(train_metric, val_metric, metric_name, folder, epoch):
    """
    Plot training and validation metrics
    
    Args:
        train_metric: Training metric values
        val_metric: Validation metric values
        metric_name: Name of the metric (e.g., 'PSNR', 'SSIM')
        folder: Folder to save the plot
        epoch: Current epoch number
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_metric, label='Train', color='blue')
    plt.plot(val_metric, label='Validation', color='orange')
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    
    # Save the plot
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{metric_name}_epoch_{epoch}.png"))
    plt.close()

def plot_metric(metric, metric_name, folder, epoch):
    """
    Plot a single metric over epochs
    
    Args:
        metric: Metric values
        metric_name: Name of the metric (e.g., 'PSNR', 'SSIM')
        folder: Folder to save the plot
        epoch: Current epoch number
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metric, label=metric_name, color='blue')
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    
    # Save the plot
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{metric_name}_epoch_{epoch}.png"))
    plt.close()

from scipy.ndimage import gaussian_filter
    
def tone_mapping_l1_l0_approx(hdr_image, sigma=3, compress_factor=0.7):
    """
    Approximates Liang et al.'s L1-L0 tone mapping using Gaussian-based layer decomposition.
    Args:
        hdr_image: np.array, shape (H, W, C), float32
    """
    hdr_image = hdr_image / np.max(hdr_image + 1e-8)  # Normalize
    log_hdr = np.log1p(hdr_image)
    base = gaussian_filter(log_hdr, sigma=sigma)
    detail = log_hdr - base
    base_compressed = base * compress_factor
    log_tone_mapped = base_compressed + detail
    tone_mapped = np.expm1(log_tone_mapped)
    tone_mapped = np.clip(tone_mapped / np.max(tone_mapped + 1e-8), 0, 1)
    return (tone_mapped * 255).astype(np.uint8)