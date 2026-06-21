"""Run inference with a trained Multicorn model over a tile dataset.

Reconstructs each chromosome from the enhanced tiles and saves a per-chromosome
compressed matrix that can be passed to 3DUnicorn for 3D reconstruction.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from models.multicorn import Multicorn
from utils.data_loader import build_loader
from utils.preprocessing import load_regulatory_tracks


def reconstruct_matrices(tiles: np.ndarray, indices: np.ndarray,
                         tile_size: int = 40) -> dict:
    """Stitch enhanced tiles back into per-chromosome dense matrices."""
    matrices = {}
    for chrom in np.unique(indices[:, 0]):
        chrom_mask = indices[:, 0] == chrom
        chrom_inds = indices[chrom_mask]
        chrom_tiles = tiles[chrom_mask]
        max_x = int((chrom_inds[:, -2] + tile_size).max())
        max_y = int((chrom_inds[:, -1] + tile_size).max())
        matrix = np.zeros((max_x, max_y), dtype=np.float32)
        for i in range(len(chrom_inds)):
            x, y = int(chrom_inds[i, -2]), int(chrom_inds[i, -1])
            matrix[x:x + tile_size, y:y + tile_size] = chrom_tiles[i].squeeze()
        matrices[str(int(chrom))] = matrix
    return matrices


def run_inference(args) -> None:
    """Load a checkpoint, enhance every tile, and save reconstructed matrices."""
    device = torch.device(f"cuda:{args.cuda}" if (torch.cuda.is_available() and args.cuda >= 0) else "cpu")
    print(f"Using device: {device}")

    atac = chip = rna = None
    if args.atac_data and args.chip_data and args.rna_data:
        atac, chip, rna = load_regulatory_tracks(args.atac_data, args.chip_data, args.rna_data, args.num_bins)

    loader, dataset = build_loader(
        args.input, batch_size=args.batch_size, shuffle=False,
        atac=atac, chip=chip, rna=rna, tile_size=args.tile_size,
    )

    model = Multicorn(
        atac_dim=args.tile_size, chip_dim=args.tile_size, rna_dim=args.tile_size,
        iterations=args.iterations,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if dataset.multimodal:
                lr, _, atac_b, chip_b, rna_b = batch
                restored = model(lr.to(device), atac_b.to(device), chip_b.to(device), rna_b.to(device))
            else:
                lr = batch[0]
                restored = model(lr.to(device))
            outputs.append(restored.cpu().numpy())

    enhanced = np.concatenate(outputs, axis=0)
    os.makedirs(args.output, exist_ok=True)

    if dataset.inds is not None:
        matrices = reconstruct_matrices(enhanced, np.asarray(dataset.inds), args.tile_size)
        for chrom, matrix in matrices.items():
            out_path = os.path.join(args.output, f"chr{chrom}_multicorn.npz")
            np.savez_compressed(out_path, multicorn=matrix)
            print(f"Saved {out_path}")
    else:
        out_path = os.path.join(args.output, "multicorn_tiles.npz")
        np.savez_compressed(out_path, multicorn=enhanced)
        print(f"Saved {out_path}")

    print("Inference complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Multicorn inference.")
    parser.add_argument("--input", required=True, help="Path to input tile .npz")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", required=True, help="Directory for enhanced matrices")
    parser.add_argument("--atac_data", default=None)
    parser.add_argument("--chip_data", default=None)
    parser.add_argument("--rna_data", default=None)
    parser.add_argument("--num_bins", type=int, default=200)
    parser.add_argument("--tile_size", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
