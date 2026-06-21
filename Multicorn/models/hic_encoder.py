"""2D convolutional encoder for scHi-C contact maps.

Produces the spatial feature map ``F_HiC`` that the multimodal fusion layer
conditions with the regulatory context ``z_multi`` before restoration.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block for stable deep feature extraction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class HiCEncoder(nn.Module):
    """Encode a low-resolution scHi-C map into a fixed-size feature grid.

    Args:
        in_channels: Number of input channels (1 for a single contact map).
        base_channels: Width of the first convolutional stage.
        latent_grid: Spatial resolution of the pooled feature grid.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 latent_grid: int = 8) -> None:
        super().__init__()
        self.feature_channels = base_channels * 2
        self.latent_grid = latent_grid
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
            nn.AdaptiveAvgPool2d((latent_grid, latent_grid)),
        )

    def forward(self, hic: torch.Tensor) -> torch.Tensor:
        return self.encoder(hic)


if __name__ == "__main__":
    encoder = HiCEncoder()
    features = encoder(torch.randn(2, 1, 40, 40))
    print(f"F_HiC shape: {tuple(features.shape)}")
