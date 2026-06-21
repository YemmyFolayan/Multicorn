"""Deep alternating restoration loop for Multicorn.

Following the Deep Alternating Network (DAN) paradigm, an Estimator and a
Restorer alternate so that errors in one are corrected by the other. The
Estimator refines the degradation kernel features; the biologically constrained
Restorer reconstructs the high-resolution map conditioned on the regulatory
context ``z_multi``, which is reinjected at every iteration. The loop is
unrolled for ``N = 5`` iterations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hic_encoder import HiCEncoder
from models.fusion import MultimodalFusion


class Estimator(nn.Module):
    """Refine the degradation-kernel features from low-resolution features.

    Args:
        input_dim: Flattened dimension of the current low-resolution map.
        kernel_dim: Dimension of the refined kernel representation.
    """

    def __init__(self, input_dim: int, kernel_dim: int = 512) -> None:
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, kernel_dim),
            nn.ReLU(inplace=True),
            nn.Linear(kernel_dim, kernel_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, lr_features: torch.Tensor) -> torch.Tensor:
        return self.estimator(lr_features)


class Restorer(nn.Module):
    """Reconstruct the high-resolution map under the regulatory prior.

    The Restorer encodes the current estimate, fuses the regulatory context via
    :class:`MultimodalFusion`, and decodes back to a contact map of the same
    spatial size as the input.

    Args:
        base_channels: Width of the Hi-C encoder.
        latent_grid: Spatial resolution of the encoded feature grid.
        latent_dim: Latent dimension of each modality encoder.
    """

    def __init__(self, base_channels: int = 64, latent_grid: int = 8,
                 latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = HiCEncoder(base_channels=base_channels, latent_grid=latent_grid)
        self.fusion = MultimodalFusion(
            latent_dim=latent_dim,
            feature_channels=self.encoder.feature_channels,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder.feature_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, lr_data: torch.Tensor,
                z_multi: Optional[torch.Tensor] = None) -> torch.Tensor:
        spatial_size = lr_data.shape[-2:]
        features = self.encoder(lr_data)
        if z_multi is not None:
            features = self.fusion(features, z_multi)
        decoded = self.decoder(features)
        return F.interpolate(decoded, size=spatial_size, mode="bilinear", align_corners=False)


class RestorationLoop(nn.Module):
    """Unrolled alternating optimization between Estimator and Restorer.

    Args:
        iterations: Number of unrolled iterations (``N`` in the paper).
        base_channels: Width passed to the Restorer encoder/decoder.
        latent_grid: Spatial resolution of the encoded feature grid.
        latent_dim: Latent dimension of each modality encoder.
        kernel_dim: Dimension of the refined kernel representation.
        blend: Convex weight used to blend the previous estimate with the new one.
    """

    def __init__(self, iterations: int = 5, base_channels: int = 64,
                 latent_grid: int = 8, latent_dim: int = 64,
                 kernel_dim: int = 512, blend: float = 0.5) -> None:
        super().__init__()
        self.iterations = iterations
        self.blend = blend
        self.restorer = Restorer(
            base_channels=base_channels,
            latent_grid=latent_grid,
            latent_dim=latent_dim,
        )
        self._estimator: Optional[Estimator] = None
        self._kernel_dim = kernel_dim

    def _ensure_estimator(self, flat_dim: int, device: torch.device) -> None:
        if self._estimator is None:
            self._estimator = Estimator(flat_dim, self._kernel_dim).to(device)
            self.add_module("estimator", self._estimator)

    def forward(self, lr_data: torch.Tensor,
                z_multi: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = lr_data
        flat_dim = current.view(current.size(0), -1).size(1)
        self._ensure_estimator(flat_dim, current.device)

        for _ in range(self.iterations):
            lr_features = current.view(current.size(0), -1)
            _ = self._estimator(lr_features)
            restored = self.restorer(current, z_multi)
            current = self.blend * current + (1.0 - self.blend) * restored

        return current


if __name__ == "__main__":
    loop = RestorationLoop(iterations=5)
    z = torch.randn(2, 192)
    out = loop(torch.randn(2, 1, 40, 40), z)
    print(f"Restored shape: {tuple(out.shape)}")
