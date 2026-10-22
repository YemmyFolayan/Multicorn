"""Contact-map evaluation metrics for Multicorn.

Reports the image-restoration and correlation metrics used in the paper: MSE,
PSNR, SSIM, Pearson, Spearman correlation and GenomeDisco. All functions accept
batched tensors of shape ``(B, C, H, W)``.

GenomeDisco is the biologically motivated metric: it compares two Hi-C contact
maps by running random walks of increasing length on each map (treated as a
graph) and measuring how quickly the smoothed maps diverge. The reported value
is a reproducibility/concordance score in roughly ``[-1, 1]`` where ``1`` means
the maps are identical after diffusion. See Ursu et al., Genome Research 2018.
"""

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim


def _upper_triangle(sample: np.ndarray) -> np.ndarray:
    # Use only the strict upper triangle for each correlation.
    matrix = np.squeeze(np.asarray(sample))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return matrix.astype(np.float64).ravel()
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols].astype(np.float64)


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
    """Mean per-tile PSNR, using that tile's HR maximum as the peak."""
    # PSNR uses each HR tile maximum as the reconstruction peak.
    scores = []
    for i in range(target.size(0)):
        mse = F.mse_loss(restored[i], target[i], reduction="mean").item()
        if mse == 0.0:
            scores.append(100.0)
            continue
        peak = float(torch.max(target[i]).item())
        if peak <= 0.0:
            scores.append(0.0)
            continue
        scores.append(float(10.0 * np.log10((peak ** 2) / mse)))
    return float(np.mean(scores)) if scores else 0.0


def compute_ssim(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean SSIM with a 7x7 window and the HR tile maximum as data_range."""
    # SSIM uses a 7x7 window and the HR tile peak.
    values = []
    for i in range(target.size(0)):
        ref = target[i].detach().cpu().numpy().squeeze()
        out = restored[i].detach().cpu().numpy().squeeze()
        side = int(min(ref.shape[-2:])) if ref.ndim >= 2 else int(ref.size)
        window = 7 if side >= 7 else side if side % 2 == 1 else side - 1
        if window < 3:
            values.append(0.0)
            continue
        peak = float(np.max(ref))
        data_range = peak if peak > 1e-8 else 1.0
        try:
            values.append(ssim(ref, out, data_range=data_range, win_size=window))
        except ValueError:
            values.append(0.0)
    return float(np.mean(values))


def _correlate_batch(func: Callable, target: torch.Tensor, restored: torch.Tensor) -> float:
    scores = []
    for i in range(target.size(0)):
        ref = _upper_triangle(target[i].detach().cpu().numpy())
        out = _upper_triangle(restored[i].detach().cpu().numpy())
        scores.append(_safe_correlation(func, ref, out))
    return float(np.mean(scores)) if scores else 0.0


def compute_pearson(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean Pearson correlation on the strict upper triangle of each tile."""
    return _correlate_batch(pearsonr, target, restored)


def compute_spearman(target: torch.Tensor, restored: torch.Tensor) -> float:
    """Mean Spearman correlation on the strict upper triangle of each tile."""
    return _correlate_batch(spearmanr, target, restored)


def _to_transition(matrix: np.ndarray, rowsums: np.ndarray) -> np.ndarray:
    """Row-normalise a contact map into a random-walk transition matrix.

    Empty rows (degree-0 nodes) are left as zero rows so that they do not inject
    probability mass during the walk.
    """
    safe = rowsums.copy()
    safe[safe == 0.0] = 1.0
    return matrix / safe[:, None]


def _genomedisco_single(
    ref: np.ndarray, out: np.ndarray, t_min: int, t_max: int
) -> float:
    """GenomeDisco reproducibility score for a single contact map pair."""
    ref = np.asarray(ref, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)

    # GenomeDisco operates on symmetric, non-negative contact maps.
    ref = np.clip(0.5 * (ref + ref.T), 0.0, None)
    out = np.clip(0.5 * (out + out.T), 0.0, None)

    # Normalise both maps to the same sequencing depth so the L1 distance
    # reflects topology rather than a difference in total contact counts.
    ref_sum, out_sum = ref.sum(), out.sum()
    if ref_sum > 0.0 and out_sum > 0.0:
        out = out * (ref_sum / out_sum)

    rowsums_ref = ref.sum(axis=1)
    rowsums_out = out.sum(axis=1)
    nonzero_total = 0.5 * (
        float(np.count_nonzero(rowsums_ref)) + float(np.count_nonzero(rowsums_out))
    )
    if nonzero_total == 0.0:
        return 0.0

    trans_ref = _to_transition(ref, rowsums_ref)
    trans_out = _to_transition(out, rowsums_out)

    walk_ref = trans_ref.copy()
    walk_out = trans_out.copy()
    diffs = []
    for step in range(1, t_max + 1):
        if step > 1:
            walk_ref = walk_ref.dot(trans_ref)
            walk_out = walk_out.dot(trans_out)
        if step >= t_min:
            diffs.append(float(np.abs(walk_ref - walk_out).sum()) / nonzero_total)

    if not diffs:
        return 0.0
    return 1.0 - float(np.mean(diffs))


def compute_genomedisco(
    target: torch.Tensor,
    restored: torch.Tensor,
    t_min: int = 3,
    t_max: int = 3,
) -> float:
    """Mean GenomeDisco reproducibility score across the batch.

    Each sample is treated as a square contact map. The random-walk length is
    controlled by ``t_min``/``t_max`` (GenomeDisco's published default is ``3``).
    Returns a score where ``1.0`` indicates identical maps.
    """
    scores = []
    for i in range(target.size(0)):
        ref = target[i].detach().cpu().numpy().squeeze()
        out = restored[i].detach().cpu().numpy().squeeze()
        if ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
            raise ValueError(
                "GenomeDisco expects square contact maps, got shape "
                f"{ref.shape}."
            )
        scores.append(_genomedisco_single(ref, out, t_min, t_max))
    return float(np.mean(scores)) if scores else 0.0


def _insulation_profile(matrix: np.ndarray, window: int) -> np.ndarray:
    size = matrix.shape[0]
    width = min(window, max(1, (size - 1) // 2))
    profile = np.full(size, np.nan, dtype=np.float64)
    for locus in range(width, size - width):
        block = matrix[locus - width:locus, locus:locus + width]
        profile[locus] = float(block.mean())
    return profile


def compute_dedoc2(target: torch.Tensor, restored: torch.Tensor, window: int = 5) -> float:
    """Mean L2 deviation of insulation profiles, the DeDoc2 domain score."""
    # DeDoc2 reports the L2 gap between the two insulation profiles.
    gaps = []
    for i in range(target.size(0)):
        ref = np.clip(np.squeeze(target[i].detach().cpu().numpy()), 0.0, None)
        out = np.clip(np.squeeze(restored[i].detach().cpu().numpy()), 0.0, None)
        ref_profile = _insulation_profile(ref, window)
        out_profile = _insulation_profile(out, window)
        valid = np.isfinite(ref_profile) & np.isfinite(out_profile)
        if not np.any(valid):
            gaps.append(0.0)
            continue
        gaps.append(float(np.linalg.norm(out_profile[valid] - ref_profile[valid])))
    return float(np.mean(gaps)) if gaps else 0.0


def _pair_correlation(func: Callable, left: np.ndarray, right: np.ndarray) -> float:
    return _safe_correlation(func, left.astype(np.float64), right.astype(np.float64))


def pearson_by_distance(
    target: torch.Tensor, restored: torch.Tensor, short_max: int = 5, mid_max: int = 20
) -> Dict[str, float]:
    """Pearson on short, mid, and long genomic-distance bands."""
    # Short mid and long bands split at five and twenty.
    bands = {"short": [], "mid": [], "long": []}
    for i in range(target.size(0)):
        ref = np.squeeze(target[i].detach().cpu().numpy())
        out = np.squeeze(restored[i].detach().cpu().numpy())
        size = ref.shape[0]
        rows, cols = np.triu_indices(size, k=1)
        distance = cols - rows
        groups = {
            "short": (distance >= 1) & (distance < short_max),
            "mid": (distance >= short_max) & (distance < mid_max),
            "long": distance >= mid_max,
        }
        for name, mask in groups.items():
            if int(mask.sum()) < 2:
                bands[name].append(0.0)
                continue
            bands[name].append(
                _pair_correlation(pearsonr, ref[rows[mask], cols[mask]], out[rows[mask], cols[mask]])
            )
    return {name: float(np.mean(values)) if values else 0.0 for name, values in bands.items()}


def pearson_by_compartment(
    target: torch.Tensor, restored: torch.Tensor, scores: torch.Tensor
) -> Dict[str, float]:
    """Pearson inside A-A, B-B, and A-B pairs from a median score split."""
    # High regulatory scores mark A and low scores mark B.
    groups: Dict[str, list] = {"AA": [], "BB": [], "AB": []}
    batch_scores = scores.expand(target.size(0), -1) if scores.dim() == 1 else scores
    for i in range(target.size(0)):
        ref = np.squeeze(target[i].detach().cpu().numpy())
        out = np.squeeze(restored[i].detach().cpu().numpy())
        signal = batch_scores[i].detach().cpu().numpy()
        active = signal >= np.median(signal)
        rows, cols = np.triu_indices(ref.shape[0], k=1)
        pair_groups = {
            "AA": active[rows] & active[cols],
            "BB": (~active[rows]) & (~active[cols]),
            "AB": active[rows] != active[cols],
        }
        for name, mask in pair_groups.items():
            if int(mask.sum()) < 2:
                groups[name].append(0.0)
                continue
            groups[name].append(
                _pair_correlation(pearsonr, ref[rows[mask], cols[mask]], out[rows[mask], cols[mask]])
            )
    return {name: float(np.mean(values)) if values else 0.0 for name, values in groups.items()}


def inactive_inactive_fraction(
    contact: np.ndarray, scores: np.ndarray, percentile: float = 0.25, top_fraction: float = 0.01
) -> float:
    """Fraction of the strongest contacts that join two low-support bins."""
    # Low support pairs among top contacts fill the HR row.
    matrix = np.squeeze(np.asarray(contact, dtype=np.float64))
    signal = np.asarray(scores, dtype=np.float64).ravel()
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    values = matrix[rows, cols]
    if values.size == 0:
        return 0.0
    keep = max(1, int(round(top_fraction * values.size)))
    chosen = np.argpartition(values, -keep)[-keep:]
    threshold = float(np.quantile(signal, percentile))
    both_low = (signal[rows[chosen]] < threshold) & (signal[cols[chosen]] < threshold)
    return float(np.mean(both_low))


def evaluate_all(target: torch.Tensor, restored: torch.Tensor) -> Dict[str, float]:
    """Compute the shared six-metric panel plus the DeDoc2 insulation score."""
    return {
        "Pearson": compute_pearson(target, restored),
        "Spearman": compute_spearman(target, restored),
        "MSE": compute_mse(target, restored),
        "SSIM": compute_ssim(target, restored),
        "PSNR": compute_psnr(target, restored),
        "GenomeDisco": compute_genomedisco(target, restored),
        "DeDoc2": compute_dedoc2(target, restored),
    }


if __name__ == "__main__":
    reference = torch.rand(4, 1, 40, 40)
    prediction = reference + 0.02 * torch.randn_like(reference)
    for key, value in evaluate_all(reference, prediction).items():
        print(f"{key}: {value:.6f}")
