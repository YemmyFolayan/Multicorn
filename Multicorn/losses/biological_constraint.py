"""Biologically constrained regularizer ``B(H, O)``.

The constraint penalizes contact intensities placed between bins that are
simultaneously inaccessible, unmarked and silent. For a contact map ``H`` and
normalized omics tracks ``O = (a, c, r)``:

    s(i, O) = mean(z(a_i), z(c_i), z(r_i))
    conflict(i, j) = [ s(i) < tau ]  and  [ s(j) < tau ]
    B(H, O) = sum_{i,j} w_{ij} * H_{ij} * conflict(i, j)
    w_{ij} = 1 / (1 + |i - j| / d0)

``tau`` is the lower 25th percentile of ``s`` across the chromosome and the
genomic-distance weight ``w_{ij}`` prevents long-range, low-intensity contacts
from dominating the term. ``B`` is a soft prior: the data term can still drive
``H_{ij} > 0`` in heterochromatin when the low-resolution data demands it, but at
a ``gamma * w_{ij} * H_{ij}`` cost.
"""

from typing import Tuple

import torch


def _zscore(track: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-normalize a 1D per-bin track."""
    mean = track.mean()
    std = track.std()
    return (track - mean) / (std + eps)


def bin_regulatory_score(atac: torch.Tensor, chip: torch.Tensor,
                         rna: torch.Tensor) -> torch.Tensor:
    """Mean of the z-normalized regulatory tracks at each genomic bin."""
    return (_zscore(atac) + _zscore(chip) + _zscore(rna)) / 3.0


def conflict_mask(scores: torch.Tensor, percentile: float = 0.25) -> torch.Tensor:
    """Boolean ``(n, n)`` mask marking pairs of low-regulation bins.

    Args:
        scores: Per-bin regulatory scores ``s`` of shape ``(n,)``.
        percentile: Lower quantile used as the threshold ``tau``.
    """
    tau = torch.quantile(scores, percentile)
    low = scores < tau
    return low.unsqueeze(0) & low.unsqueeze(1)


def distance_weights(n: int, d0: float, device: torch.device) -> torch.Tensor:
    """Genomic-distance weight matrix ``w_{ij} = 1 / (1 + |i - j| / d0)``."""
    idx = torch.arange(n, device=device, dtype=torch.float32)
    separation = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return 1.0 / (1.0 + separation / d0)


def biological_constraint(hic: torch.Tensor, atac: torch.Tensor,
                          chip: torch.Tensor, rna: torch.Tensor,
                          d0: float = 10.0, percentile: float = 0.25) -> torch.Tensor:
    """Evaluate ``B(H, O)`` for a batch of contact maps.

    Args:
        hic: Contact maps of shape ``(B, 1, n, n)`` or ``(B, n, n)``.
        atac: ATAC-seq per-bin track of shape ``(n,)`` or ``(B, n)``.
        chip: H3K27ac ChIP-seq per-bin track, same shape convention as ``atac``.
        rna: RNA-seq per-bin track, same shape convention as ``atac``.
        d0: Decay constant of the genomic-distance weight (in bins).
        percentile: Lower quantile defining the conflict threshold ``tau``.

    Returns:
        Scalar penalty averaged over the batch.
    """
    maps = hic.squeeze(1) if hic.dim() == 4 else hic
    batch_size, n, _ = maps.shape

    atac_b = atac.expand(batch_size, n) if atac.dim() == 1 else atac
    chip_b = chip.expand(batch_size, n) if chip.dim() == 1 else chip
    rna_b = rna.expand(batch_size, n) if rna.dim() == 1 else rna

    weights = distance_weights(n, d0, maps.device)

    penalties = []
    for b in range(batch_size):
        scores = bin_regulatory_score(atac_b[b], chip_b[b], rna_b[b])
        mask = conflict_mask(scores, percentile).float()
        penalties.append((weights * maps[b].abs() * mask).sum())

    return torch.stack(penalties).mean()


def biological_constraint_self_check() -> Tuple[float, float]:
    """Compare penalties for intensity placed in silent vs active regions.

    The regulatory landscape is fixed (an ascending track, so the lowest bins are
    inaccessible/unmarked/silent). A map that places intensity in the silent
    corner should be penalized far more than one that places it in the active
    corner.
    """
    n = 16
    track = torch.linspace(0.0, 1.0, n)

    silent_corner = torch.zeros(1, 1, n, n)
    silent_corner[0, 0, :4, :4] = 1.0

    active_corner = torch.zeros(1, 1, n, n)
    active_corner[0, 0, -4:, -4:] = 1.0

    penalized = biological_constraint(silent_corner, track, track, track).item()
    allowed = biological_constraint(active_corner, track, track, track).item()
    return penalized, allowed


if __name__ == "__main__":
    penalized, allowed = biological_constraint_self_check()
    print(f"B(H,O) for intensity in silent region: {penalized:.4f}")
    print(f"B(H,O) for intensity in active region: {allowed:.4f}")
