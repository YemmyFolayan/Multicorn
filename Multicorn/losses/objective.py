"""Multicorn biologically constrained objective.

Augments the base blind super-resolution objective with the regulatory prior:

    J_Multi(H, T, O) = ||L - T(H)||^2 + beta * P(H) + gamma * B(H, O)

The data term measures degradation consistency, ``P(H)`` is a sparsity/smoothness
prior on the reconstructed map, and ``B(H, O)`` is the biological consistency
term. Training uses the ``L1`` form of the data and reconstruction terms.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from losses.biological_constraint import biological_constraint


def total_variation(hic: torch.Tensor) -> torch.Tensor:
    """Anisotropic total-variation smoothness prior ``P(H)``."""
    maps = hic if hic.dim() == 4 else hic.unsqueeze(1)
    dh = (maps[:, :, 1:, :] - maps[:, :, :-1, :]).abs().mean()
    dw = (maps[:, :, :, 1:] - maps[:, :, :, :-1]).abs().mean()
    return dh + dw


class MulticornObjective(nn.Module):
    """Composite loss combining reconstruction, smoothness and biology.

    Args:
        beta: Weight of the sparsity/smoothness prior ``P(H)``.
        gamma: Weight of the biological consistency term ``B(H, O)``.
        d0: Decay constant of the genomic-distance weight (in bins).
        percentile: Lower quantile defining the conflict threshold ``tau``.
    """

    def __init__(self, beta: float = 0.01, gamma: float = 0.1,
                 d0: float = 10.0, percentile: float = 0.25) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.d0 = d0
        self.percentile = percentile
        self.reconstruction = nn.L1Loss()

    def forward(self, restored: torch.Tensor, target: torch.Tensor,
                atac: Optional[torch.Tensor] = None,
                chip: Optional[torch.Tensor] = None,
                rna: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        data_term = self.reconstruction(restored, target)
        smoothness = total_variation(restored)

        biology = restored.new_zeros(())
        if atac is not None and chip is not None and rna is not None:
            biology = biological_constraint(
                restored, atac, chip, rna, d0=self.d0, percentile=self.percentile
            )

        total = data_term + self.beta * smoothness + self.gamma * biology
        return {
            "total": total,
            "data": data_term,
            "smoothness": smoothness,
            "biology": biology,
        }


if __name__ == "__main__":
    objective = MulticornObjective()
    restored = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    losses = objective(restored, target, torch.randn(16), torch.randn(16), torch.randn(16))
    for key, value in losses.items():
        print(f"{key}: {value.item():.4f}")
