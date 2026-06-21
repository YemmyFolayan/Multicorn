"""Dataset and loaders for Multicorn training and inference.

Wraps the ScUnicorn-style ``.npz`` tiles (``data`` low-resolution, ``target``
high-resolution, ``inds`` window indices) and optionally attaches the three
regulatory tracks, sliced to the genomic window of each tile.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HiCTileDataset(Dataset):
    """Tile dataset that optionally yields aligned regulatory tracks.

    Args:
        npz_path: Path to a ``.npz`` archive with ``data`` and ``target`` arrays
            and, optionally, an ``inds`` array of window indices.
        atac: Chromosome-level ATAC-seq track, or ``None`` for unimodal mode.
        chip: Chromosome-level H3K27ac ChIP-seq track, or ``None``.
        rna: Chromosome-level RNA-seq track, or ``None``.
        tile_size: Side length of each square tile in bins.
    """

    def __init__(self, npz_path: str, atac: Optional[np.ndarray] = None,
                 chip: Optional[np.ndarray] = None, rna: Optional[np.ndarray] = None,
                 tile_size: int = 40) -> None:
        archive = np.load(npz_path, allow_pickle=True)
        self.data = archive["data"].astype(np.float32)
        self.target = archive["target"].astype(np.float32)
        self.inds = archive["inds"] if "inds" in archive else None
        self.atac = atac
        self.chip = chip
        self.rna = rna
        self.tile_size = tile_size
        self.multimodal = atac is not None and chip is not None and rna is not None

    def __len__(self) -> int:
        return len(self.data)

    def _slice_track(self, track: np.ndarray, start: int) -> np.ndarray:
        end = start + self.tile_size
        window = track[start:end]
        if window.size < self.tile_size:
            window = np.pad(window, (0, self.tile_size - window.size))
        return window.astype(np.float32)

    def _tile_start(self, idx: int) -> int:
        if self.inds is None:
            return 0
        return int(self.inds[idx, -1])

    def __getitem__(self, idx: int):
        lr = torch.from_numpy(self.data[idx]).float()
        hr = torch.from_numpy(self.target[idx]).float()
        if lr.dim() == 2:
            lr = lr.unsqueeze(0)
        if hr.dim() == 2:
            hr = hr.unsqueeze(0)

        if not self.multimodal:
            return lr, hr

        start = self._tile_start(idx)
        atac = torch.from_numpy(self._slice_track(self.atac, start))
        chip = torch.from_numpy(self._slice_track(self.chip, start))
        rna = torch.from_numpy(self._slice_track(self.rna, start))
        return lr, hr, atac, chip, rna


def build_loader(npz_path: str, batch_size: int = 64, shuffle: bool = True,
                 atac: Optional[np.ndarray] = None, chip: Optional[np.ndarray] = None,
                 rna: Optional[np.ndarray] = None,
                 tile_size: int = 40) -> Tuple[DataLoader, HiCTileDataset]:
    """Create a :class:`DataLoader` for the tile dataset."""
    dataset = HiCTileDataset(npz_path, atac=atac, chip=chip, rna=rna, tile_size=tile_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset
