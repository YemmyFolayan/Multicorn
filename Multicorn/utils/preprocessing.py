"""Preprocessing utilities for Multicorn.

All four signals are aligned to the same genomic bins as the Hi-C map. Raw
contact maps are stabilized with ``log1p``; the regulatory tracks are reduced to
per-bin scalars (log-transformed ATAC read counts, H3K27ac signal intensity and
log-FPKM expression) and are length-matched to the number of Hi-C bins.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def load_hic_matrix(path: str) -> np.ndarray:
    """Load a dense Hi-C matrix from a ``.txt``, ``.npy`` or ``.npz`` file."""
    if path.endswith(".txt"):
        matrix = np.loadtxt(path)
    elif path.endswith(".npy"):
        matrix = np.load(path)
    elif path.endswith(".npz"):
        archive = np.load(path)
        matrix = archive[list(archive.keys())[0]]
    else:
        raise ValueError("Unsupported Hi-C file format. Use .txt, .npy or .npz.")
    return np.nan_to_num(matrix.astype(np.float32))


def normalize_hic(matrix: np.ndarray) -> np.ndarray:
    """Apply the variance-stabilizing ``log1p`` transform to a contact map."""
    return np.log1p(np.nan_to_num(matrix.astype(np.float32)))


def align_track_to_bins(track: np.ndarray, num_bins: int) -> np.ndarray:
    """Resample a 1D regulatory track to exactly ``num_bins`` entries."""
    track = np.nan_to_num(track.astype(np.float32)).ravel()
    if track.size == 0:
        return np.zeros(num_bins, dtype=np.float32)
    if track.size == num_bins:
        return track
    source = np.linspace(0.0, 1.0, num=track.size)
    target = np.linspace(0.0, 1.0, num=num_bins)
    return np.interp(target, source, track).astype(np.float32)


def load_atac_track(path: str, num_bins: int) -> np.ndarray:
    """Load an ATAC-seq accessibility track and align it to the Hi-C bins."""
    frame = pd.read_csv(path, sep="\t", header=0)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return np.zeros(num_bins, dtype=np.float32)
    values = np.log1p(np.abs(numeric.to_numpy(dtype=np.float32)).mean(axis=1))
    return align_track_to_bins(values, num_bins)


def load_chip_track(path: str, num_bins: int,
                    mark: str = "H3K27ac_encode") -> np.ndarray:
    """Load an H3K27ac ChIP-seq track and align it to the Hi-C bins."""
    frame = pd.read_csv(path, sep="\t")
    if mark in frame.columns:
        values = frame[mark].to_numpy(dtype=np.float32)
    else:
        numeric = frame.select_dtypes(include=[np.number])
        values = numeric.to_numpy(dtype=np.float32).mean(axis=1)
    return align_track_to_bins(np.nan_to_num(values), num_bins)


def load_rna_track(path: str, num_bins: int) -> np.ndarray:
    """Load an RNA-seq expression track (log-FPKM) and align it to the bins."""
    frame = pd.read_csv(path, sep="\t")
    fpkm_cols = [c for c in frame.columns if "FPKM" in c.upper()]
    if fpkm_cols:
        values = frame[fpkm_cols].to_numpy(dtype=np.float32).mean(axis=1)
    else:
        numeric = frame.select_dtypes(include=[np.number])
        values = numeric.to_numpy(dtype=np.float32).mean(axis=1)
    return align_track_to_bins(np.log1p(np.abs(np.nan_to_num(values))), num_bins)


def load_regulatory_tracks(atac_path: str, chip_path: str, rna_path: str,
                           num_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and bin-align the three regulatory tracks for one chromosome."""
    atac = load_atac_track(atac_path, num_bins)
    chip = load_chip_track(chip_path, num_bins)
    rna = load_rna_track(rna_path, num_bins)
    return atac, chip, rna
