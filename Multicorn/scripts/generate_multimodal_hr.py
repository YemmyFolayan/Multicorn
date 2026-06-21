"""Generate an enhanced high-resolution scHi-C map with Multicorn.

Loads a dense low-resolution contact map together with matched ATAC-seq,
H3K27ac ChIP-seq and RNA-seq tracks, runs the multimodal blind super-resolution
model, and writes both an enhanced contact matrix (for 3DUnicorn) and a heatmap
image.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from models.multicorn import Multicorn
from utils.preprocessing import (
    load_hic_matrix,
    normalize_hic,
    load_regulatory_tracks,
)


def build_model(num_bins: int, iterations: int, device: torch.device) -> Multicorn:
    """Instantiate the Multicorn model for full-matrix inference."""
    model = Multicorn(
        atac_dim=num_bins, chip_dim=num_bins, rna_dim=num_bins, iterations=iterations
    ).to(device)
    model.eval()
    return model


def load_checkpoint(model: Multicorn, checkpoint_path: str, device: torch.device) -> None:
    """Load model weights when a compatible checkpoint is available."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
            return
        except (RuntimeError, EOFError, OSError) as error:
            print(f"[WARN] Could not load checkpoint ({error}); using initialized weights.")
    else:
        print("[WARN] Checkpoint not found; using initialized weights.")


def save_matrix(matrix: np.ndarray, out_path: str) -> None:
    """Write the enhanced contact matrix as a tab-separated text file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savetxt(out_path, matrix, fmt="%.6f", delimiter="\t")
    print(f"[INFO] Saved enhanced Hi-C matrix to {out_path}")


def save_heatmap(matrix: np.ndarray, out_path: str) -> None:
    """Write a normalized grayscale heatmap of the enhanced map."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scaled = np.log1p(np.nan_to_num(matrix))
    span = scaled.max() - scaled.min()
    scaled = (scaled - scaled.min()) / (span + 1e-8) * 255.0
    Image.fromarray(scaled.astype(np.uint8)).save(out_path)
    print(f"[INFO] Saved enhanced heatmap to {out_path}")


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    raw = load_hic_matrix(args.data_path)
    num_bins = raw.shape[0]
    normalized = normalize_hic(raw)

    atac, chip, rna = load_regulatory_tracks(
        args.atac_data_path, args.chip_data_path, args.rna_data_path, num_bins
    )

    model = build_model(num_bins, args.iterations, device)
    load_checkpoint(model, args.model_path, device)

    hic_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)
    atac_tensor = torch.from_numpy(atac).float().unsqueeze(0).to(device)
    chip_tensor = torch.from_numpy(chip).float().unsqueeze(0).to(device)
    rna_tensor = torch.from_numpy(rna).float().unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = model(hic_tensor, atac_tensor, chip_tensor, rna_tensor)

    enhanced_log = enhanced.squeeze().cpu().numpy()
    enhanced_matrix = np.expm1(np.clip(enhanced_log, a_min=0.0, a_max=None))
    enhanced_matrix = (enhanced_matrix + enhanced_matrix.T) / 2.0

    save_matrix(enhanced_matrix, args.output_hic_path)
    save_heatmap(enhanced_matrix, args.output_image_path)
    print("[INFO] Enhancement complete. Pass the matrix to 3DUnicorn for 3D reconstruction.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HR Hi-C map with Multicorn.")
    parser.add_argument("--model_path", default="", help="Path to a trained checkpoint")
    parser.add_argument("--data_path", required=True, help="Path to dense Hi-C input (.txt/.npy/.npz)")
    parser.add_argument("--atac_data_path", required=True, help="Path to ATAC-seq track")
    parser.add_argument("--chip_data_path", required=True, help="Path to H3K27ac ChIP-seq track")
    parser.add_argument("--rna_data_path", required=True, help="Path to RNA-seq track")
    parser.add_argument("--output_image_path", required=True, help="Path to save the heatmap (.png)")
    parser.add_argument("--output_hic_path", required=True, help="Path to save the matrix (.txt)")
    parser.add_argument("--iterations", type=int, default=5, help="Unrolled loop iterations N")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
