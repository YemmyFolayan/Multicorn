"""Full Multicorn model.

Multicorn (Multimodal Unicorn) extends the ScUnicorn blind super-resolution
backbone with three additions over the unimodal baseline: independent encoders
for the regulatory tracks, a fusion injection that conditions restoration on the
regulatory context ``z_multi``, and a biologically constrained objective. The
same Deep Alternating Network backbone is reused; setting the modality inputs to
``None`` recovers the unimodal model.
"""

from typing import Optional

import torch
import torch.nn as nn

from models.modality_encoders import RegulatoryEncoders
from models.restoration_loop import RestorationLoop


class Multicorn(nn.Module):
    """Multimodal blind super-resolution model for scHi-C contact maps.

    Args:
        atac_dim: Input feature dimension of the ATAC-seq track.
        chip_dim: Input feature dimension of the H3K27ac ChIP-seq track.
        rna_dim: Input feature dimension of the RNA-seq track.
        iterations: Number of unrolled restoration iterations (``N``).
        base_channels: Width of the restoration encoder/decoder.
        latent_dim: Latent dimension shared by the modality encoders.
    """

    def __init__(self, atac_dim: int, chip_dim: int, rna_dim: int,
                 iterations: int = 5, base_channels: int = 64,
                 latent_dim: int = 64) -> None:
        super().__init__()
        self.encoders = RegulatoryEncoders(
            atac_dim=atac_dim,
            chip_dim=chip_dim,
            rna_dim=rna_dim,
            latent_dim=latent_dim,
        )
        self.loop = RestorationLoop(
            iterations=iterations,
            base_channels=base_channels,
            latent_dim=latent_dim,
        )

    def encode_context(self, atac: torch.Tensor, chip: torch.Tensor,
                       rna: torch.Tensor) -> torch.Tensor:
        """Encode the three regulatory tracks into ``z_multi``."""
        z_atac, z_chip, z_rna = self.encoders(atac, chip, rna)
        return self.loop.restorer.fusion.build_context(z_atac, z_chip, z_rna)

    def forward(self, lr_hic: torch.Tensor,
                atac: Optional[torch.Tensor] = None,
                chip: Optional[torch.Tensor] = None,
                rna: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_multi = None
        if atac is not None and chip is not None and rna is not None:
            z_multi = self.encode_context(atac, chip, rna)
        return self.loop(lr_hic, z_multi)


if __name__ == "__main__":
    model = Multicorn(atac_dim=8, chip_dim=6, rna_dim=4, iterations=5)
    hic = torch.randn(2, 1, 40, 40)
    enhanced = model(hic, torch.randn(2, 8), torch.randn(2, 6), torch.randn(2, 4))
    print(f"Enhanced HR map shape: {tuple(enhanced.shape)}")

    unimodal = model(hic)
    print(f"Unimodal fallback shape: {tuple(unimodal.shape)}")
