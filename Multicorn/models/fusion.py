"""Multimodal fusion layer.

The three regulatory latents are concatenated into the regulatory context
``z_multi = [z_a || z_c || z_r]`` (R^192). Concatenation is preferred over
averaging so the Restorer can learn asymmetric weights and preserve modality
identity. A learned linear projection ``phi`` broadcasts ``z_multi`` onto the
Hi-C feature map before every restoration step:

    F_cond = F_HiC + phi(z_multi)
"""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Project the regulatory context and add it to the Hi-C features.

    Args:
        latent_dim: Latent dimension of each modality encoder.
        feature_channels: Channel count of the Hi-C feature map ``F_HiC``.
    """

    def __init__(self, latent_dim: int = 64, feature_channels: int = 128) -> None:
        super().__init__()
        self.context_dim = latent_dim * 3
        self.feature_channels = feature_channels
        self.projection = nn.Linear(self.context_dim, feature_channels)

    def build_context(self, z_atac: torch.Tensor, z_chip: torch.Tensor,
                      z_rna: torch.Tensor) -> torch.Tensor:
        """Concatenate the modality latents into the regulatory context."""
        return torch.cat([z_atac, z_chip, z_rna], dim=1)

    def forward(self, hic_features: torch.Tensor,
                z_multi: torch.Tensor) -> torch.Tensor:
        """Broadcast ``phi(z_multi)`` over the spatial grid and add it."""
        projected = self.projection(z_multi)
        projected = projected.unsqueeze(-1).unsqueeze(-1)
        return hic_features + projected


if __name__ == "__main__":
    fusion = MultimodalFusion(latent_dim=64, feature_channels=128)
    context = fusion.build_context(torch.randn(2, 64), torch.randn(2, 64), torch.randn(2, 64))
    fused = fusion(torch.randn(2, 128, 8, 8), context)
    print(f"z_multi: {tuple(context.shape)}, F_cond: {tuple(fused.shape)}")
