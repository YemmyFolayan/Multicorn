"""Cell-to-cell variability checks and input-dependence controls.

Compares pairwise similarity of raw, enhanced, and high-resolution maps, then
reruns inference with the scHi-C input zeroed, swapped, or bin-shuffled and
with the shared tracks shuffled.
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch
from scipy.stats import pearsonr

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from models.multicorn import Multicorn
from utils.metrics import _safe_correlation, _upper_triangle


def _batch(maps: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(maps, dtype=np.float32))
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    return tensor


def _repeat_track(track: np.ndarray, batch: int) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(track, dtype=np.float32)).reshape(-1)
    return values.unsqueeze(0).expand(batch, -1).contiguous()


def _pair_pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    return _safe_correlation(
        pearsonr,
        _upper_triangle(left.detach().cpu().numpy()),
        _upper_triangle(right.detach().cpu().numpy()),
    )


def mean_pairwise_pearson(maps: torch.Tensor) -> float:
    """Mean Pearson correlation between every pair of cells."""
    scores = [
        _pair_pearson(maps[i], maps[j])
        for i in range(maps.size(0))
        for j in range(i + 1, maps.size(0))
    ]
    return float(np.mean(scores)) if scores else 0.0


def _forward(model: Multicorn, hic: torch.Tensor, atac: torch.Tensor,
             chip: torch.Tensor, rna: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(hic, atac, chip, rna)


def run_controls(model: Multicorn, raw: torch.Tensor, hr: torch.Tensor,
                 atac: torch.Tensor, chip: torch.Tensor, rna: torch.Tensor) -> Dict[str, float]:
    """Run the variability comparison and the four input controls."""
    # Compare cell pairs after zeroing, swapping, or shuffling the Hi-C.
    model.eval()
    enhanced = _forward(model, raw, atac, chip, rna)
    zeroed = _forward(model, torch.zeros_like(raw), atac, chip, rna)
    swapped = raw.clone()
    swapped[0], swapped[1] = raw[1].clone(), raw[0].clone()
    swapped_out = _forward(model, swapped, atac, chip, rna)
    permutation = torch.randperm(raw.size(-1))
    shuffled = raw[:, :, permutation][:, :, :, permutation]
    shuffled_out = _forward(model, shuffled, atac, chip, rna)
    track_perm = torch.randperm(atac.size(-1))
    shuffled_tracks = _forward(
        model, raw, atac[:, track_perm], chip[:, track_perm], rna[:, track_perm],
    )
    bulk = hr.mean(dim=0, keepdim=True).expand_as(hr)
    paired = [
        _pair_pearson(enhanced[i], shuffled_tracks[i]) for i in range(enhanced.size(0))
    ]
    bulk_scores = [_pair_pearson(enhanced[i], bulk[i]) for i in range(enhanced.size(0))]
    return {
        "raw_pairwise_pearson": mean_pairwise_pearson(raw),
        "enhanced_pairwise_pearson": mean_pairwise_pearson(enhanced),
        "hr_pairwise_pearson": mean_pairwise_pearson(hr),
        "zero_input_vs_enhanced": _pair_pearson(zeroed[0], enhanced[0]),
        "swap_follows_donor": _pair_pearson(swapped_out[0], enhanced[1]),
        "swap_follows_slot": _pair_pearson(swapped_out[0], enhanced[0]),
        "bin_shuffle_vs_enhanced": float(np.mean([
            _pair_pearson(shuffled_out[i], enhanced[i]) for i in range(enhanced.size(0))
        ])),
        "track_shuffle_vs_enhanced": float(np.mean(paired)),
        "enhanced_vs_pseudobulk_hr": float(np.mean(bulk_scores)),
    }


def self_check() -> None:
    """Exercise the controls on synthetic cells with an untrained model."""
    generator = np.random.default_rng(0)
    count, size = 4, 16
    base = generator.random((size, size))
    base = ((base + base.T) / 2.0).astype(np.float32)
    raw, hr = [], []
    for index in range(count):
        cell = base * (0.4 + 0.15 * index) + 0.05 * generator.random((size, size))
        cell = ((cell + cell.T) / 2.0).astype(np.float32)
        hr.append(cell)
        keep = generator.random((size, size)) > 0.7
        raw.append((cell * keep).astype(np.float32))
    raw_t, hr_t = _batch(np.stack(raw)), _batch(np.stack(hr))
    track = generator.random(size).astype(np.float32)
    model = Multicorn(atac_dim=size, chip_dim=size, rna_dim=size, iterations=2)
    report = run_controls(
        model, raw_t, hr_t,
        _repeat_track(track, count), _repeat_track(track, count), _repeat_track(track, count),
    )
    for name, value in report.items():
        print(f"{name}: {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cell-to-cell variability and input controls.")
    parser.add_argument("--cells", help="npz with raw and hr arrays shaped (N, H, W)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--atac", required=False)
    parser.add_argument("--chip", required=False)
    parser.add_argument("--rna", required=False)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.cells or not args.atac or not args.chip or not args.rna:
        raise ValueError("Provide --cells, --atac, --chip, and --rna, or run --self_check.")
    archive = np.load(args.cells)
    raw, hr = _batch(archive["raw"]), _batch(archive["hr"])
    count, size = raw.size(0), raw.size(-1)
    model = Multicorn(atac_dim=size, chip_dim=size, rna_dim=size, iterations=args.iterations)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    report = run_controls(
        model, raw, hr,
        _repeat_track(np.load(args.atac), count),
        _repeat_track(np.load(args.chip), count),
        _repeat_track(np.load(args.rna), count),
    )
    for name, value in report.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
