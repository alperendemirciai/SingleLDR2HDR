# by Levent Karacan

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
import torchsummary

class ConvBNAct(nn.Module):
    """Helper: Conv -> BatchNorm -> Activation"""
    # REGCOV
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = activation(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    """Upsample, concat skip, double conv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Up: ConvTranspose2d for learnable upsampling
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concat, channels = out_ch + skip_ch -> fuse back to out_ch
        self.conv1 = ConvBNAct(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # both x and skip spatial must match
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class MobileNetV3_UNet(nn.Module):
    def __init__(self, num_classes):
        # ============================================
        # R                                          R
        # E               MR. LEVENT                 E
        # G                                          G
        # C                KARACAN                   C
        # O                                          O
        # V                  :)                      V
        # ============================================
        super().__init__()
        base_model = mobilenet_v3_large(pretrained=True)
        self.encoder = base_model.features
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.skip_idxs = [0, 2, 4, 7]

        # D0: 7->14, (idx7 channels=80) -> output 160
        self.up0 = UpBlock(960, 80, 160)
        # D1: 14->28, (idx4 channels=40) -> output 80
        self.up1 = UpBlock(160, 40, 80)
        # D2: 28->56, (idx2 channels=24) -> output 40
        self.up2 = UpBlock(80, 24, 40)
        # D3: 56->112, (idx0 channels=16) -> output 24
        self.up3 = UpBlock(40, 16, 24)
        # D4: 112->224, no skip -> output 16
        self.up4 = UpBlock(24, 0, 16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

        # TANH
        self.act_out = nn.Tanh()

    def forward(self, x):
        skips = []
        # collect skips
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.skip_idxs:
                skips.append(x)
        # x is now bottleneck
        assert len(skips) == 4, f"Expected 4 skip features, got {len(skips)}"
        f0, f1, f2, f3 = skips  # f0:112×112, f1:56×56, f2:28×28, f3:14×14

        # Decoder
        d0 = self.up0(x, f3)    # ->14×14×160
        d1 = self.up1(d0, f2)   # ->28×28×80
        d2 = self.up2(d1, f1)   # ->56×56×40
        d3 = self.up3(d2, f0)   # ->112×112×24
        d4 = self.up4(d3, None) # ->224×224×16

        out = self.final_conv(d4)  # ->224×224×num_classes
        out = self.act_out(out)    # ->224×224×num_classes
        return out

if __name__ == "__main__":
    model = MobileNetV3_UNet(num_classes=3)
    x = torch.randn(1,3,224,224)
    y = model(x)
    print(y.shape)  
    print("Model summary:")
    torchsummary.summary(model, (3, 224, 224), device="cpu")
