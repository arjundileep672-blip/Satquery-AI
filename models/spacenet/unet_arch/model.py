"""
SatQuery AI - Bundled U-Net Architecture for SpaceNet Rio Building Detector

This is a standard encoder-decoder U-Net compatible with the SpaceNet Rio
pretrained checkpoint (harshinde/spacenet-models on HuggingFace).

Input:  3-channel RGB image, normalized to [0, 1]
Output: 1-channel sigmoid probability map (building = high probability)

Source architecture: standard U-Net (Ronneberger et al., 2015)
Compatible checkpoint: spacenet_rio.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with BN and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """MaxPool + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Bilinear upsample + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial dims if they differ by 1 pixel
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary building segmentation.

    Args:
        in_channels:  Number of input channels (3 for RGB).
        out_channels: Number of output channels (1 for binary mask).
        base_features: Base channel count (default 64).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 64):
        super().__init__()
        f = base_features
        self.inc  = _DoubleConv(in_channels, f)
        self.down1 = _Down(f,     f * 2)
        self.down2 = _Down(f * 2, f * 4)
        self.down3 = _Down(f * 4, f * 8)
        self.down4 = _Down(f * 8, f * 16)
        self.up1  = _Up(f * 16 + f * 8, f * 8)
        self.up2  = _Up(f * 8  + f * 4, f * 4)
        self.up3  = _Up(f * 4  + f * 2, f * 2)
        self.up4  = _Up(f * 2  + f,     f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)
