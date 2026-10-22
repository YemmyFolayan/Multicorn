"""Early-fusion baseline that gives the unimodal loop the same three tracks.

The tracks are stacked as extra input channels. There is no multimodal fusion
layer and the biological consistency weight stays at zero, so this baseline
separates extra inputs from Multicorn's fusion and loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hic_encoder import HiCEncoder
from models.restoration_loop import Estimator


class EarlyFusionBaseline(nn.Module):
    """Unimodal restoration loop with regulatory tracks as extra channels.

    Args:
        iterations: Number of unrolled restoration iterations.
        base_channels: Width of the contact-map encoder.
        latent_grid: Spatial size of the pooled feature grid.
        blend: Weight on the previous estimate inside the unrolled loop.
    """

    def __init__(self, iterations: int = 5, base_channels: int = 64,
                 latent_grid: int = 8, blend: float = 0.5) -> None:
        super().__init__()
        self.iterations = iterations
        self.blend = blend
        self.encoder = HiCEncoder(
            in_channels=4, base_channels=base_channels, latent_grid=latent_grid,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder.feature_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )
        self._estimator: Optional[Estimator] = None

    def _channel(self, track: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if track is None:
            return torch.zeros_like(like)
        if track.dim() == 1:
            track = track.unsqueeze(0).expand(like.size(0), -1)
        row = track.reshape(track.size(0), 1, track.size(1), 1)
        return row.expand(-1, 1, -1, like.size(-1))

    def _stack(self, hic: torch.Tensor, atac: Optional[torch.Tensor],
               chip: Optional[torch.Tensor], rna: Optional[torch.Tensor]) -> torch.Tensor:
        # Tracks enter as extra channels with the biological loss off.
        return torch.cat(
            [hic, self._channel(atac, hic), self._channel(chip, hic), self._channel(rna, hic)],
            dim=1,
        )

    def forward(self, lr_hic: torch.Tensor, atac: Optional[torch.Tensor] = None,
                chip: Optional[torch.Tensor] = None,
                rna: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = lr_hic
        flat_dim = current.reshape(current.size(0), -1).size(1)
        if self._estimator is None:
            self._estimator = Estimator(flat_dim).to(current.device)
            self.add_module("estimator", self._estimator)
        for _ in range(self.iterations):
            _ = self._estimator(current.reshape(current.size(0), -1))
            features = self.encoder(self._stack(current, atac, chip, rna))
            restored = self.decoder(features)
            restored = F.interpolate(
                restored, size=current.shape[-2:], mode="bilinear", align_corners=False,
            )
            current = self.blend * current + (1.0 - self.blend) * restored
        return current


if __name__ == "__main__":
    model = EarlyFusionBaseline(iterations=2)
    enhanced = model(torch.rand(2, 1, 16, 16), torch.rand(2, 16), torch.rand(2, 16), None)
    print(f"Early-fusion map shape: {tuple(enhanced.shape)}")
