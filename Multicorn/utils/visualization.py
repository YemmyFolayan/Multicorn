"""Visualization helpers for Multicorn contact maps.

Provides single-map heatmaps and locus-level comparisons (raw, unimodal,
multimodal, ground truth) that mirror the qualitative figures in the paper.
"""

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_contact_map(matrix: np.ndarray, title: str = "Hi-C map",
                     cmap: str = "hot", save_path: Optional[str] = None,
                     dpi: int = 300) -> None:
    """Render a single contact map as a heatmap."""
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(label="Contact intensity")
    plt.title(title)
    plt.xlabel("Genomic position")
    plt.ylabel("Genomic position")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def compare_contact_maps(matrices: Sequence[np.ndarray], titles: Sequence[str],
                         cmap: str = "hot", save_path: Optional[str] = None,
                         dpi: int = 300) -> None:
    """Render several contact maps side by side for qualitative comparison."""
    panels = len(matrices)
    plt.figure(figsize=(6 * panels, 6))
    for index, (matrix, title) in enumerate(zip(matrices, titles), start=1):
        plt.subplot(1, panels, index)
        plt.imshow(matrix, cmap=cmap, aspect="auto")
        plt.title(title)
        plt.colorbar(label="Intensity")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    reference = np.random.rand(40, 40)
    enhanced = reference + np.random.normal(0, 0.05, reference.shape)
    compare_contact_maps(
        [reference, enhanced],
        ["Raw scHi-C", "Multicorn enhanced"],
        save_path="output/compare_demo.png",
    )
    print("Saved demo comparison plot.")
