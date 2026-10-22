"""Shared evaluation script for every method in the camera-ready paper.

Scores restored contact tiles against the high-resolution target with the same
Pearson, Spearman, MSE, SSIM, PSNR, GenomeDISCO, and DeDoc2 definitions.
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from losses.biological_constraint import bin_regulatory_score
from utils.metrics import (
    evaluate_all,
    inactive_inactive_fraction,
    pearson_by_compartment,
    pearson_by_distance,
)


def load_maps(path: str, key: Optional[str] = None) -> np.ndarray:
    """Load a tile stack as ``(N, 1, H, W)``."""
    if path.endswith(".npy"):
        array = np.load(path)
    else:
        archive = np.load(path)
        if key is not None:
            array = archive[key]
        else:
            preferred = ("prediction", "restored", "multicorn", "target", "data")
            found = next((name for name in preferred if name in archive.files), archive.files[0])
            array = archive[found]
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, None, ...]
    elif array.ndim == 3:
        array = array[:, None, ...]
    return array


def _print_metrics(target: torch.Tensor, restored: torch.Tensor, scores: Optional[torch.Tensor]) -> None:
    # One script scores every method with the shared paper metrics.
    for name, value in evaluate_all(target, restored).items():
        print(f"{name}: {value:.6f}")
    distance = pearson_by_distance(target, restored)
    print(
        "distance_pearson short/mid/long: "
        f"{distance['short']:.6f} {distance['mid']:.6f} {distance['long']:.6f}"
    )
    if scores is None:
        return
    compartments = pearson_by_compartment(target, restored, scores)
    print(
        "compartment_pearson AA/BB/AB: "
        f"{compartments['AA']:.6f} {compartments['BB']:.6f} {compartments['AB']:.6f}"
    )
    signal = scores.detach().cpu().numpy()
    for label, tensor in (("prediction", restored), ("target", target)):
        rates = [
            inactive_inactive_fraction(tensor[i].numpy(), signal[i] if signal.ndim == 2 else signal)
            for i in range(tensor.size(0))
        ]
        print(f"inactive_inactive_top1_{label}: {float(np.mean(rates)):.6f}")


def self_check() -> None:
    """Score a near-copy tile so the shared protocol can be exercised."""
    generator = torch.Generator().manual_seed(0)
    target = torch.rand(2, 1, 40, 40, generator=generator)
    restored = target + 0.01 * torch.randn(2, 1, 40, 40, generator=generator)
    scores = torch.linspace(0.0, 1.0, 40)
    _print_metrics(target, restored, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score restored Hi-C tiles with the shared protocol.")
    parser.add_argument("--prediction", help="Restored tiles, .npy or .npz")
    parser.add_argument("--target", help="High-resolution tiles, .npy or .npz")
    parser.add_argument("--prediction_key", default=None)
    parser.add_argument("--target_key", default=None)
    parser.add_argument("--atac", default=None, help="Per-tile ATAC track, .npy of shape (N, bins) or (bins,)")
    parser.add_argument("--chip", default=None)
    parser.add_argument("--rna", default=None)
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.prediction or not args.target:
        raise ValueError("Provide --prediction and --target, or run --self_check.")

    restored = torch.from_numpy(load_maps(args.prediction, args.prediction_key))
    target = torch.from_numpy(load_maps(args.target, args.target_key))
    scores = None
    if args.atac and args.chip and args.rna:
        atac = torch.from_numpy(np.load(args.atac).astype(np.float32))
        chip = torch.from_numpy(np.load(args.chip).astype(np.float32))
        rna = torch.from_numpy(np.load(args.rna).astype(np.float32))
        if atac.dim() == 1:
            scores = bin_regulatory_score(atac, chip, rna)
        else:
            scores = torch.stack([
                bin_regulatory_score(atac[i], chip[i], rna[i]) for i in range(atac.size(0))
            ])
    _print_metrics(target, restored, scores)


if __name__ == "__main__":
    main()
