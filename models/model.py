import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, mode="pixelshuffle", scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        if mode == "pixelshuffle":
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
            )
        elif mode == "convtranspose":
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=scale_factor)
        elif mode == "bilinear":
            self.upsample = nn.Sequential()
        else:
            raise ValueError(f"Unknown upsampling mode: {mode}")

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        if self.mode == "bilinear":
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class MobileNetV3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            feature_maps.append(x)
        return x, feature_maps

class Decoder(nn.Module):
    def __init__(self, upsample_mode="pixelshuffle"):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(960, 1472, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1472, 1472, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpsampleBlock(1472, 112, 480, mode=upsample_mode)
        self.up2 = UpsampleBlock(480, 40, 160, mode=upsample_mode)
        self.up3 = UpsampleBlock(160, 24, 64, mode=upsample_mode)
        self.up4 = UpsampleBlock(64, 16, 32, mode=upsample_mode)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, features):
        skip_112 = features[12]
        skip_40 = features[6]
        skip_24 = features[3]
        skip_16 = features[1]

        x = self.bottleneck(x)
        x = self.up1(x, skip_112)
        x = self.up2(x, skip_40)
        x = self.up3(x, skip_24)
        x = self.up4(x, skip_16)
        return self.final(x)

class Autoencoder(nn.Module):
    def __init__(self, upsample_mode="pixelshuffle"):
        super().__init__()
        self.encoder = MobileNetV3Encoder()
        self.decoder = Decoder(upsample_mode)

    def forward(self, x):
        encoded, features = self.encoder(x)
        return self.decoder(encoded, features)

if __name__ == "__main__":
    model = Autoencoder(upsample_mode="bilinear").to("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
