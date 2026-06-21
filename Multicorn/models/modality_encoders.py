"""Modality-specific encoders for the three regulatory tracks.

Each 1D omics signal (ATAC-seq accessibility ``a``, H3K27ac ChIP-seq enrichment
``c``, RNA-seq expression ``r``) is passed through an independent two-layer MLP
with hidden widths ``(128, 64)`` and ReLU activations, producing a 64-dimensional
latent. Modality-specific encoders are a deliberate inductive bias: contact
frequencies, read counts and FPKM values have very different dynamic ranges and
noise structures.
"""

from typing import Tuple

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Two-layer MLP encoder mapping a per-bin omics vector to a latent.

    Args:
        input_dim: Number of per-bin input features for the modality.
        hidden_dim: Width of the intermediate hidden layer.
        latent_dim: Dimension of the produced latent representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RegulatoryEncoders(nn.Module):
    """Bundle of the ATAC-seq, ChIP-seq and RNA-seq encoders.

    Args:
        atac_dim: Input feature dimension of the ATAC-seq track.
        chip_dim: Input feature dimension of the H3K27ac ChIP-seq track.
        rna_dim: Input feature dimension of the RNA-seq track.
        latent_dim: Latent dimension shared by all three encoders.
    """

    def __init__(self, atac_dim: int, chip_dim: int, rna_dim: int,
                 hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        self.atac = ModalityEncoder(atac_dim, hidden_dim, latent_dim)
        self.chip = ModalityEncoder(chip_dim, hidden_dim, latent_dim)
        self.rna = ModalityEncoder(rna_dim, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, atac: torch.Tensor, chip: torch.Tensor,
                rna: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.atac(atac), self.chip(chip), self.rna(rna)


if __name__ == "__main__":
    encoders = RegulatoryEncoders(atac_dim=8, chip_dim=6, rna_dim=4)
    z_a, z_c, z_r = encoders(torch.randn(2, 8), torch.randn(2, 6), torch.randn(2, 4))
    print(f"z_a: {tuple(z_a.shape)}, z_c: {tuple(z_c.shape)}, z_r: {tuple(z_r.shape)}")
