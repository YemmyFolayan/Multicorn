"""Dynamic degradation kernel for Multicorn blind super-resolution.

The kernel models the unknown forward degradation ``L = T(H) + E`` that turns a
high-resolution scHi-C map into the observed sparse low-resolution map. Rather
than assuming a fixed downsampling ratio, the kernel is learned end to end and
co-evolves with the restored map inside the alternating optimization loop.
"""

from typing import Tuple

import random

import torch
import torch.nn as nn


class DegradationKernel(nn.Module):
    """Learnable degradation operator with stochastic downsampling.

    Args:
        kernel_size: Spatial size of the learned convolution kernel.
        scale_factor: Downsampling factor applied to the high-resolution map.
    """

    def __init__(self, kernel_size: int = 3, scale_factor: int = 2) -> None:
        super().__init__()
        self.kernel = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.scale_factor = scale_factor

    def random_downsample(self, hr_data: torch.Tensor) -> torch.Tensor:
        """Sample a degraded view of the HR map at random row/column indices."""
        _, _, height, width = hr_data.size()
        target_height = max(1, int(height / self.scale_factor))
        target_width = max(1, int(width / self.scale_factor))

        indices_h = sorted(random.sample(range(height), target_height))
        indices_w = sorted(random.sample(range(width), target_width))

        return hr_data[:, :, indices_h, :][:, :, :, indices_w]

    def forward(self, hr_data: torch.Tensor) -> torch.Tensor:
        """Apply stochastic downsampling followed by the learned kernel."""
        downsampled = self.random_downsample(hr_data)
        return self.kernel(downsampled)

    def apply_kernel(self, data: torch.Tensor) -> torch.Tensor:
        """Apply only the learned convolution, used for LR-consistency checks."""
        return self.kernel(data)


def degradation_self_check() -> Tuple[torch.Size, torch.Size]:
    """Return the input/output shapes for a smoke test of the kernel."""
    hr_input = torch.randn(1, 1, 128, 128)
    kernel = DegradationKernel(kernel_size=3, scale_factor=2)
    lr_output = kernel(hr_input)
    return hr_input.shape, lr_output.shape


if __name__ == "__main__":
    in_shape, out_shape = degradation_self_check()
    print(f"HR input shape: {tuple(in_shape)}")
    print(f"LR output shape: {tuple(out_shape)}")
