"""Contact-map evaluation metrics for Multicorn.

Reports the image-restoration and correlation metrics used in the paper: MSE,
PSNR, SSIM, Pearson and Spearman correlation. All functions accept batched
tensors of shape ``(B, C, H, W)``.
"""

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim


def _flatten(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy().reshape(data.size(0), -1)


def _safe_correlation(func: Callable, x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    try:
        corr, _ = func(x, y)
    except ValueError:
        return 0.0
    return 0.0 if np.isnan(corr) else float(corr)


def compute_mse(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean squared error between target and restored maps."""
    return F.mse_loss(restored, target, reduction="mean").item()


def compute_psnr(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Peak signal-to-noise ratio relative to the target dynamic range."""
    mse = compute_mse(target, restored)
    if mse == 0.0:
        return 100.0
    max_i = torch.max(target).item()
    return float(10.0 * np.log10((max_i ** 2) / mse))


def compute_ssim(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean structural similarity index across the batch."""
    values = []
    for i in range(target.size(0)):
        ref = target[i].detach().cpu().numpy().squeeze()
        out = restored[i].detach().cpu().numpy().squeeze()
        data_range = out.max() - out.min()
        if data_range < 1e-8:
            values.append(1.0)
            continue
        try:
            values.append(ssim(ref, out, data_range=data_range))
        except ValueError:
            values.append(0.0)
    return float(np.mean(values))


def compute_pearson(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean Pearson correlation across the batch."""
    ref, out = _flatten(target), _flatten(restored)
    scores = [_safe_correlation(pearsonr, ref[i], out[i]) for i in range(ref.shape[0])]
    return float(np.mean(scores))


def compute_spearman(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean Spearman rank correlation across the batch."""
    ref, out = _flatten(target), _flatten(restored)
    scores = [_safe_correlation(spearmanr, ref[i], out[i]) for i in range(ref.shape[0])]
    return float(np.mean(scores))


def evaluate_all(target: torch.Tensor, restored: torch.Tensor) -> Dict[str, float]:
    """Compute the full metric suite as a dictionary."""
    return {
        "MSE": compute_mse(target, restored),
        "PSNR": compute_psnr(target, restored),
        "SSIM": compute_ssim(target, restored),
        "Pearson": compute_pearson(target, restored),
        "Spearman": compute_spearman(target, restored),
    }


if __name__ == "__main__":
    reference = torch.rand(4, 1, 40, 40)
    prediction = reference + 0.02 * torch.randn_like(reference)
    for key, value in evaluate_all(reference, prediction).items():
        print(f"{key}: {value:.6f}")
