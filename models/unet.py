import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        res = self.conv(x)
        pooled = self.pool(res)
        return res, pooled

    
class UNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

        
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling_method: str = "bilinear"):
        super().__init__()

        if upsampling_method == "transpose":
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        elif upsampling_method == "nearest":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        elif upsampling_method == "pixelshuffle":
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        else:
            raise ValueError(f"Unsupported upsampling method: {upsampling_method}")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        if x2 is None:
            return self.conv(x1)

        if x1.shape[2] != x2.shape[2] or x1.shape[3] != x2.shape[3]:
            x2 = F.interpolate(x2, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)

        x1 = torch.cat([x1, x2], dim=1)

        return self.conv(x1)



class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, base_filters=8, upsampling_method:str="pixelshuffle"):
        super().__init__()
        # Encoder (contracting path)

        self.enc1 = UNetDown(in_channels, base_filters)
        self.enc2 = UNetDown(base_filters, base_filters * 2)
        self.enc3 = UNetDown(base_filters * 2, base_filters * 4)
        self.enc4 = UNetDown(base_filters * 4, base_filters * 8)

        # Bottleneck

        self.bottleneck = UNetBottleneck(base_filters * 8, base_filters * 16)

        # Decoder (expanding path)

        self.dec1 = UNetUp(base_filters * 16, base_filters * 8, upsampling_method=upsampling_method)
        self.dec2 = UNetUp(base_filters * 8 , base_filters * 4, upsampling_method=upsampling_method)
        self.dec3 = UNetUp(base_filters * 4 , base_filters * 2, upsampling_method=upsampling_method)
        self.dec4 = UNetUp(base_filters * 2 , base_filters, upsampling_method=upsampling_method)

        # Final 1×1 conv
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)


    def forward(self, x, verbose = False):

        original_h, original_w = x.shape[2], x.shape[3]
        # Encoder
        enc1, x = self.enc1(x)
        if verbose:
            print(f"Shape of x after enc1: {x.shape}")
        enc2, x = self.enc2(x)
        if verbose:
            print(f"Shape of x after enc2: {x.shape}")
        enc3, x = self.enc3(x)
        if verbose:
            print(f"Shape of x after enc3: {x.shape}")
        
        enc4, x = self.enc4(x)
        if verbose:
            print(f"Shape of x after enc4: {x.shape}")

        # Bottleneck

        x = self.bottleneck(x)
        if verbose:
            print(f"Shape of x after bottleneck: {x.shape}")

        # Decoder + skip connections

        x = self.dec1(x, enc4)
        if verbose:
            print(f"Shape of x after dec1: {x.shape}")
        x = self.dec2(x, enc3)
        if verbose:
            print(f"Shape of x after dec2: {x.shape}")
        x = self.dec3(x, enc2)
        if verbose:
            print(f"Shape of x after dec3: {x.shape}")
        x = self.dec4(x, enc1)
        if verbose:
            print(f"Shape of x after dec4: {x.shape}")

        # Final 1×1 conv
        x = self.out_conv(x)
        if x.shape[2] != original_h or x.shape[3] != original_w:
            x = F.interpolate(x, size=(original_h, original_w), mode='bilinear', align_corners=True)
        
        return x


if __name__ == "__main__":
    # Example usage
    model = UNet(in_channels=3, out_channels=4, base_filters=8, upsampling_method="pixelshuffle")
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image
    output = model(x, verbose=True)
    print(f"Output shape: {output.shape}")  # Should be (1, 4, 256, 256)
    # Example usage with different upsampling methods
    for method in ["transpose", "bilinear", "nearest", "pixelshuffle"]:
        print(f"Testing upsampling method: {method}")
        model = UNet(in_channels=3, out_channels=4, base_filters=8, upsampling_method=method)
        x = torch.randn(1, 3, 600, 960)  # Batch size of 1, 3 channels, 256x256 image
        output = model(x)
        print(f"Output shape with {method}: {output.shape}")  # Should be (1, 4, 256, 256)
    # Example usage with different input sizes
    for size in [(128, 128), (512, 512), (1024, 1024)]:
        print(f"Testing input size: {size}")
        model = UNet(in_channels=3, out_channels=4, base_filters=8, upsampling_method="pixelshuffle")
        x = torch.randn(1, 3, *size)  # Batch size of 1, 3 channels, variable size
        output = model(x)
        print(f"Output shape with input size {size}: {output.shape}")  # Should be (1, 4, *size)
    # Example usage with different input channels
    for in_channels in [1, 3, 5]:
        print(f"Testing input channels: {in_channels}")
        model = UNet(in_channels=in_channels, out_channels=4, base_filters=8, upsampling_method="pixelshuffle")
        x = torch.randn(1, in_channels, 256, 256)  # Batch size of 1, variable channels, 256x256 image
        output = model(x)
        print(f"Output shape with {in_channels} input channels: {output.shape}")  # Should be (1, 4, 256, 256)