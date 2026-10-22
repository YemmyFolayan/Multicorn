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

    def encode_context(self, atac: Optional[torch.Tensor], chip: Optional[torch.Tensor],
                       rna: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode whichever regulatory tracks are present into ``z_multi``."""
        # A missing track adds a zero latent, not encoder bias.
        reference = next(track for track in (atac, chip, rna) if track is not None)
        zeros = torch.zeros(
            reference.shape[0], self.encoders.latent_dim,
            device=reference.device, dtype=reference.dtype,
        )
        z_atac = self.encoders.atac(atac) if atac is not None else zeros
        z_chip = self.encoders.chip(chip) if chip is not None else zeros
        z_rna = self.encoders.rna(rna) if rna is not None else zeros
        return self.loop.restorer.fusion.build_context(z_atac, z_chip, z_rna)

    def forward(self, lr_hic: torch.Tensor,
                atac: Optional[torch.Tensor] = None,
                chip: Optional[torch.Tensor] = None,
                rna: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_multi = None
        if any(track is not None for track in (atac, chip, rna)):
            z_multi = self.encode_context(atac, chip, rna)
        return self.loop(lr_hic, z_multi)


if __name__ == "__main__":
    model = Multicorn(atac_dim=8, chip_dim=6, rna_dim=4, iterations=5)
    hic = torch.randn(2, 1, 40, 40)
    enhanced = model(hic, torch.randn(2, 8), torch.randn(2, 6), torch.randn(2, 4))
    print(f"Enhanced HR map shape: {tuple(enhanced.shape)}")

    unimodal = model(hic)
    print(f"Unimodal fallback shape: {tuple(unimodal.shape)}")
