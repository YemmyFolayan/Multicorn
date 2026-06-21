"""Train the Multicorn multimodal blind super-resolution model.

The model is trained end to end with the L1 form of the biologically constrained
objective. Gradients flow through the entire unrolled restoration loop so the
regulatory encoders are shaped by the contact-map objective rather than learned
in isolation.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from losses.objective import MulticornObjective
from models.multicorn import Multicorn
from utils.data_loader import build_loader
from utils.preprocessing import load_regulatory_tracks


def _load_tracks(args, num_bins):
    if not (args.atac_data and args.chip_data and args.rna_data):
        return None, None, None
    return load_regulatory_tracks(args.atac_data, args.chip_data, args.rna_data, num_bins)


def train(args) -> None:
    """Run the full training loop and save the trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    atac, chip, rna = _load_tracks(args, args.num_bins)
    train_loader, _ = build_loader(
        args.train_data, batch_size=args.batch_size, shuffle=True,
        atac=atac, chip=chip, rna=rna, tile_size=args.tile_size,
    )

    multimodal = atac is not None
    model = Multicorn(
        atac_dim=args.tile_size,
        chip_dim=args.tile_size,
        rna_dim=args.tile_size,
        iterations=args.iterations,
    ).to(device)

    objective = MulticornObjective(beta=args.beta, gamma=args.gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), eps=1e-8)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            if multimodal:
                lr, hr, atac_b, chip_b, rna_b = batch
                atac_b, chip_b, rna_b = atac_b.to(device), chip_b.to(device), rna_b.to(device)
            else:
                lr, hr = batch
                atac_b = chip_b = rna_b = None

            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            restored = model(lr, atac_b, chip_b, rna_b)
            losses = objective(restored, hr, atac_b, chip_b, rna_b)
            losses["total"].backward()
            optimizer.step()
            running += losses["total"].item()

        print(f"Epoch {epoch + 1}: train loss = {running / max(1, len(train_loader)):.4f}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Training complete. Saved model to {args.save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Multicorn model.")
    parser.add_argument("--train_data", required=True, help="Path to training .npz")
    parser.add_argument("--atac_data", default=None, help="Path to ATAC-seq track")
    parser.add_argument("--chip_data", default=None, help="Path to H3K27ac ChIP-seq track")
    parser.add_argument("--rna_data", default=None, help="Path to RNA-seq track")
    parser.add_argument("--num_bins", type=int, default=200, help="Chromosome bin count")
    parser.add_argument("--tile_size", type=int, default=40, help="Tile side length in bins")
    parser.add_argument("--iterations", type=int, default=5, help="Unrolled loop iterations N")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--save_path", default="../../checkpoint/multicorn_model.pytorch")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
