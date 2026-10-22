"""Measure Multicorn parameter count and per-cell forward time.

Reports the omics-encoder cost separately from the restoration backbone, then
times one cell at tile sizes 40, 100, and 200. Larger cell counts are a linear
extrapolation of the measured per-cell time, not a separate benchmark.
"""

import os
import sys
import time

import torch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from models.early_fusion import EarlyFusionBaseline
from models.multicorn import Multicorn


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _time_forward(model: torch.nn.Module, tile_size: int, device: torch.device) -> float:
    hic = torch.rand(1, 1, tile_size, tile_size, device=device)
    track = torch.rand(1, tile_size, device=device)
    with torch.no_grad():
        model(hic, track, track, track)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(hic, track, track, track)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    # Report parameter counts and per cell time on this device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_size = 40
    model = Multicorn(
        atac_dim=tile_size, chip_dim=tile_size, rna_dim=tile_size, iterations=5,
    ).to(device)
    model.eval()
    _time_forward(model, tile_size, device)
    encoder_params = _count(model.encoders)
    projection_params = _count(model.loop.restorer.fusion)
    total_params = _count(model)
    one_encoder = _count(model.encoders.atac)
    print(f"device: {device}")
    print(f"one_encoder_params: {one_encoder}")
    print(f"three_encoder_params: {encoder_params}")
    print(f"projection_params: {projection_params}")
    print(f"multicorn_params_after_one_forward: {total_params}")
    print(f"backbone_params: {total_params - encoder_params - projection_params}")

    baseline = EarlyFusionBaseline(iterations=5).to(device)
    baseline.eval()
    _time_forward(baseline, tile_size, device)
    print(f"early_fusion_params: {_count(baseline)}")

    for size in (40, 100, 200):
        timed = Multicorn(atac_dim=size, chip_dim=size, rna_dim=size, iterations=5).to(device)
        timed.eval()
        seconds = _time_forward(timed, size, device)
        print(f"tile_{size}_seconds_per_cell: {seconds:.6f}")
        for cells in (100, 1000, 10000):
            print(f"tile_{size}_cells_{cells}_linear_extrapolation_seconds: {seconds * cells:.4f}")
    if device.type == "cuda":
        print(f"peak_gpu_bytes: {torch.cuda.max_memory_allocated(device)}")
    else:
        print("peak_gpu_bytes: unavailable on cpu")


if __name__ == "__main__":
    main()
