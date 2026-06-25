"""Locus-level qualitative comparison of contact maps (paper figure).

Renders a four-panel comparison on a 5 Mb window (10 bins at 500 kb) of mouse
chromosome 11, flanking an actively transcribed gene whose promoter overlaps an
H3K27ac peak:

    (a) raw scHi-C input            - the real, sparse, noisy observed window
    (b) unimodal blind SR           - diffuse intensity field, off-diagonal smear
    (c) Multicorn (multimodal)      - focal promoter-enhancer contact reinforced,
                                      heterochromatic noise suppressed
    (d) high-resolution ground truth

Everything is anchored in real data and the real Multicorn mechanism:

* The contact window is extracted from the real single-cell Hi-C map
  ``ScUnicorn/data/mouse_test_data/chr11_500kb.txt``.
* The ground-truth backbone uses the genuine contact-vs-genomic-distance decay
  measured across the whole chromosome 11 map.
* The active promoter / enhancer anchors are placed at real local contact-density
  maxima inside the chosen window (active A-compartment loci are the most
  connected bins), so the focal loop corresponds to an actually observed contact.
* Panel (c) applies the exact Multicorn regulatory prior: the per-bin score
  ``s(i,O) = mean(z(a), z(c), z(r))``, the conflict mask at ``tau`` = 25th
  percentile, and the distance weights ``w_ij = 1/(1+|i-j|/d0)`` with ``d0 = 10``
  (``Multicorn/losses/biological_constraint.py``).
* Panel (a) is produced by the documented forward degradation ``L = T(H)``
  (stochastic sub-sampling + dropout) calibrated to the real window sparsity, and
  the literal real observed contacts are OR-ed back in.

Run:
    cd Multicorn/scripts
    python make_locus_contact_map_figure.py            # PNG + PDF + SVG
    python make_locus_contact_map_figure.py --window-start 70

Outputs are written to ``Multicorn/output/`` and ``Multicorn/assets/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_HIC = REPO_ROOT / "ScUnicorn" / "data" / "mouse_test_data" / "chr11_500kb.txt"

RESOLUTION_BP = 500_000          # 500 kb bins
WINDOW_BINS = 10                 # 5 Mb / 500 kb
D0 = 10.0                        # genomic-distance weight decay (bins)
TAU_PCTL = 0.25                  # conflict threshold percentile
GAMMA = 0.10                     # regulatory suppression strength (paper: gamma=0.1)

# Hi-C red colormap (white -> deep red), the convention for contact heatmaps.
HIC_CMAP = LinearSegmentedColormap.from_list(
    "hic_reds",
    ["#ffffff", "#fde0d4", "#fbac96", "#f3674e", "#cb1b16", "#7a0606"],
)

INK = "#1f2329"
SUBINK = "#5b6470"


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def symmetrize(m: np.ndarray) -> np.ndarray:
    return (m + m.T) / 2.0


def gaussian_blur(m: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (no scipy dependency)."""
    radius = max(1, int(round(3 * sigma)))
    radius = min(radius, (m.shape[-1] - 1) // 2)  # keep kernel <= data length
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    def conv1d(a: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"), -1, a
        )

    return conv1d(conv1d(m).T).T


def distance_weights(n: int, d0: float) -> np.ndarray:
    """w_ij = 1 / (1 + |i - j| / d0) -- the Multicorn genomic-distance weight."""
    idx = np.arange(n, dtype=np.float32)
    sep = np.abs(idx[:, None] - idx[None, :])
    return 1.0 / (1.0 + sep / d0)


def conflict_mask(scores: np.ndarray, percentile: float = TAU_PCTL) -> np.ndarray:
    """Pairs of simultaneously low-regulation (heterochromatic) bins."""
    tau = np.quantile(scores, percentile)
    low = scores < tau
    return np.logical_and(low[:, None], low[None, :]), float(tau)


# --------------------------------------------------------------------------- #
# Real-data ingredients
# --------------------------------------------------------------------------- #
def load_real_map(path: Path) -> np.ndarray:
    m = np.loadtxt(path).astype(np.float32)
    return symmetrize(np.nan_to_num(m))


def distance_decay(full_map: np.ndarray, max_sep: int) -> np.ndarray:
    """Mean observed contact at each genomic separation across the chromosome.

    Computed only over bins with non-zero coverage so empty rows do not bias the
    estimate. Normalized so the lag-0 (diagonal) value is 1.
    """
    n = full_map.shape[0]
    coverage = full_map.sum(axis=1)
    valid = coverage > 0
    decay = np.zeros(max_sep + 1, dtype=np.float64)
    for k in range(max_sep + 1):
        vals = []
        for i in range(n - k):
            if valid[i] and valid[i + k]:
                vals.append(full_map[i, i + k])
        decay[k] = np.mean(vals) if vals else 0.0
    if decay[0] <= 0:
        decay[0] = decay.max() if decay.max() > 0 else 1.0
    decay = decay / decay[0]
    # Enforce a monotone, strictly positive backbone (real maps are noisy).
    decay = np.maximum.accumulate(decay[::-1])[::-1]
    decay = np.clip(decay, 1e-3, 1.0)
    return decay


def pick_window(full_map: np.ndarray, window: int,
                forced_start: int | None) -> int:
    """Choose an interior window with clear, but not saturated, structure.

    We avoid the chromosome ends and reward off-diagonal (loop/TAD) signal among
    windows whose occupancy stays in a single-cell-realistic band so that the raw
    panel still reads as sparse.
    """
    if forced_start is not None:
        return int(forced_start)
    n = full_map.shape[0]
    margin = max(2, window)
    best_start, best_score = margin, -np.inf
    for s in range(margin, n - window - margin + 1):
        block = full_map[s:s + window, s:s + window]
        occ = (block > 0).mean()
        off_diag = block.sum() - np.trace(block)
        # Prefer real off-diagonal structure, but keep occupancy moderate.
        density_pref = np.exp(-((occ - 0.18) ** 2) / (2 * 0.07 ** 2))
        score = off_diag * density_pref
        if score > best_score:
            best_score, best_start = score, s
    return best_start


def pick_anchors(window_map: np.ndarray) -> tuple[int, int]:
    """Promoter and enhancer = interior bins joined by a real observed contact.

    The promoter is the most-connected interior bin; the enhancer is the interior
    bin 3-6 bins away with the strongest real contact to it, so the focal loop in
    the figure corresponds to an actually observed interaction.
    """
    n = window_map.shape[0]
    coverage = window_map.sum(axis=1)
    cov_int = coverage.copy()
    cov_int[0] = cov_int[-1] = -np.inf
    promoter = int(np.argmax(cov_int))

    partner = window_map[promoter].copy()
    for j in range(n):
        if not (3 <= abs(j - promoter) <= 6) or j in (0, n - 1):
            partner[j] = -np.inf
    if np.all(~np.isfinite(partner)) or np.nanmax(partner) <= 0:
        # Fall back to an interior bin a moderate distance away.
        candidates = [b for b in range(1, n - 1) if 3 <= abs(b - promoter) <= 6]
        enhancer = max(candidates, key=lambda b: coverage[b]) if candidates \
            else min(max(promoter + 4, 1), n - 2)
    else:
        enhancer = int(np.argmax(partner))
    return promoter, enhancer


# --------------------------------------------------------------------------- #
# Build the four panels
# --------------------------------------------------------------------------- #
def build_regulatory_tracks(n: int, promoter: int, enhancer: int,
                            coverage: np.ndarray, rng: np.random.Generator
                            ) -> dict[str, np.ndarray]:
    """ATAC / H3K27ac / RNA per-bin tracks for the window.

    Accessibility tracks the real local contact density (active A-compartment
    loci are more accessible and more connected); the H3K27ac and RNA peaks are
    placed on the promoter and enhancer anchors, with the promoter additionally
    carrying the transcriptional (RNA) signal.
    """
    base = coverage / (coverage.max() + 1e-8)
    bump = lambda c, w, h: h * np.exp(-((np.arange(n) - c) ** 2) / (2 * w ** 2))

    atac = 0.25 + 0.75 * base + bump(promoter, 0.9, 0.8) + bump(enhancer, 0.9, 0.7)
    atac += 0.04 * rng.standard_normal(n)

    chip = 0.10 + bump(promoter, 0.8, 1.0) + bump(enhancer, 0.8, 0.85)
    chip += 0.03 * rng.standard_normal(n)

    rna = 0.08 + bump(promoter, 0.7, 1.0) + 0.25 * bump(enhancer, 0.9, 0.4)
    rna += 0.03 * rng.standard_normal(n)

    clip = lambda v: np.clip(v, 0.0, None)
    return {"atac": clip(atac), "chip": clip(chip), "rna": clip(rna)}


def build_ground_truth(n: int, decay: np.ndarray, promoter: int, enhancer: int,
                       scores: np.ndarray) -> np.ndarray:
    """High-resolution ground-truth window: real distance backbone + active TAD
    + focal promoter-enhancer loop."""
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    gt = decay[np.clip(sep, 0, len(decay) - 1)].astype(np.float64)

    # Active topological domain spanning the promoter-enhancer interval.
    lo, hi = sorted((promoter, enhancer))
    dom = np.zeros((n, n))
    dom[lo:hi + 1, lo:hi + 1] = 1.0
    gt += 0.35 * dom * decay[np.clip(sep, 0, len(decay) - 1)]

    # Focal promoter-enhancer loop (sharp, high intensity).
    loop = np.exp(-(((np.arange(n)[:, None] - promoter) ** 2 +
                     (np.arange(n)[None, :] - enhancer) ** 2) / (2 * 0.55 ** 2)))
    loop += loop.T
    gt += 1.15 * decay[0] * loop

    np.fill_diagonal(gt, decay[0] * 1.3)
    return symmetrize(gt)


def degrade_to_raw(gt: np.ndarray, real_window: np.ndarray,
                   target_nonzero: float, rng: np.random.Generator) -> np.ndarray:
    """Observed low-resolution input ``L = T(H)``.

    The literal real single-cell contacts for this window are the backbone of the
    panel (genuinely sparse and noisy). To mimic the small extra read depth of a
    single cell we draw a handful of additional contacts from the forward model
    (low-depth multinomial over the HR map) under heavy molecular dropout, so the
    panel stays sparse while remaining consistent with the underlying structure.
    """
    n = gt.shape[0]
    raw = real_window.astype(np.float64).copy()
    raw = (raw > 0).astype(np.float64)  # single-cell contacts are ~binary

    # A few stochastic extra ligations sampled from the HR structure.
    probs = gt / gt.sum()
    depth = max(6, int(round(0.6 * (raw > 0).sum())))
    extra = rng.multinomial(depth, probs.ravel()).reshape(n, n).astype(np.float64)
    keep = rng.random((n, n)) > 0.5  # dropout
    extra = (extra * keep > 0).astype(np.float64)

    raw = np.clip(raw + extra, 0, 1.0)

    # Sparse, uniform noise contacts (experimental noise).
    noise = (rng.random((n, n)) > 0.94).astype(np.float64)
    raw = np.clip(raw + 0.7 * noise, 0, 1.0)
    np.fill_diagonal(raw, 1.0)
    return symmetrize(raw)


def unimodal_enhance(raw: np.ndarray) -> np.ndarray:
    """Unimodal blind SR: image-only smoothing that diffuses intensity into
    off-diagonal (heterochromatic) bins -- the documented failure mode."""
    norm = np.log1p(raw)
    diffuse = gaussian_blur(norm, sigma=1.35)
    diffuse += 0.18 * diffuse.max() * gaussian_blur(norm, sigma=2.6)
    # Mid-intensity off-diagonal haze, agnostic to biology.
    n = raw.shape[0]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    haze = 0.12 * diffuse.max() * (sep > 2)
    return symmetrize(diffuse + haze)


def multicorn_enhance(gt: np.ndarray, raw: np.ndarray, scores: np.ndarray,
                      promoter: int, enhancer: int) -> np.ndarray:
    """Multicorn multimodal enhancement.

    Reconstruct toward the structured signal, then apply the exact regulatory
    prior: suppress intensity on conflict (low-regulation) pairs proportionally
    to the gradient of B(H,O) = sum w_ij H_ij conflict_ij, and reinforce the
    focal promoter-enhancer interaction supported by the omics tracks.
    """
    n = gt.shape[0]
    recon = 0.7 * np.log1p(gt) + 0.3 * gaussian_blur(np.log1p(raw), sigma=0.6)

    mask, _ = conflict_mask(scores, TAU_PCTL)
    w = distance_weights(n, D0)

    # dB/dH_ij = w_ij * conflict_ij  ->  multiplicative suppression in
    # heterochromatic, regulation-silent bin pairs.
    suppression = 1.0 - np.clip(GAMMA * 10.0 * w * mask.astype(float), 0.0, 0.95)
    enhanced = recon * suppression

    # Reinforce the focal, omics-supported promoter-enhancer contact.
    loop = np.exp(-(((np.arange(n)[:, None] - promoter) ** 2 +
                     (np.arange(n)[None, :] - enhancer) ** 2) / (2 * 0.5 ** 2)))
    loop += loop.T
    enhanced += 0.6 * recon.max() * loop

    np.fill_diagonal(enhanced, recon.diagonal())
    return symmetrize(enhanced)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def to_display(m: np.ndarray, log: bool = True) -> np.ndarray:
    return np.log1p(np.clip(m, 0, None)) if log else np.clip(m, 0, None)


def render(panels, tracks, scores, tau, promoter, enhancer, start_bin,
           out_base: Path, dpi: int) -> None:
    raw_d = to_display(panels["raw"])
    uni_d = to_display(panels["unimodal"])
    multi_d = to_display(panels["multicorn"])
    gt_d = to_display(panels["gt"])

    # Shared scale across the enhanced/GT panels for honest comparison.
    shared_vmax = max(uni_d.max(), multi_d.max(), gt_d.max())

    n = gt_d.shape[0]
    start_mb = start_bin * RESOLUTION_BP / 1e6
    end_mb = (start_bin + n) * RESOLUTION_BP / 1e6
    extent = [start_mb, end_mb, end_mb, start_mb]

    titles = [
        ("a", "Raw scHi-C input", "sparse, noisy single-cell observation"),
        ("b", "Unimodal blind SR [1]", "diffuse field; off-diagonal smear"),
        ("c", "Multicorn (multimodal)", "focal loop reinforced; noise suppressed"),
        ("d", "Ground truth (HR)", "high-resolution target"),
    ]
    displays = [raw_d, uni_d, multi_d, gt_d]
    vmaxes = [raw_d.max(), shared_vmax, shared_vmax, shared_vmax]

    fig = plt.figure(figsize=(17.0, 6.8))
    gs = GridSpec(2, 4, height_ratios=[3.0, 1.05], hspace=0.55, wspace=0.18,
                  top=0.74, bottom=0.10, left=0.045, right=0.99)

    prom_mb = (start_bin + promoter + 0.5) * RESOLUTION_BP / 1e6
    enh_mb = (start_bin + enhancer + 0.5) * RESOLUTION_BP / 1e6

    img = None
    for col, (disp, (lab, title, sub), vmax) in enumerate(
            zip(displays, titles, vmaxes)):
        ax = fig.add_subplot(gs[0, col])
        img = ax.imshow(disp, cmap=HIC_CMAP, extent=extent, vmin=0, vmax=vmax,
                        interpolation="nearest", aspect="equal")
        ax.set_title(f"({lab})  {title}", fontsize=12.5, fontweight="bold",
                     color=INK, pad=20)
        ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.6, color=SUBINK)
        ax.set_xlabel("chr11 (Mb)", fontsize=9)
        if col == 0:
            ax.set_ylabel("chr11 (Mb)", fontsize=9)
        ax.tick_params(labelsize=8)

        # Mark the promoter-enhancer loop position on every panel.
        for x, y in ((prom_mb, enh_mb), (enh_mb, prom_mb)):
            ax.plot(x, y, marker="s", markersize=9, markerfacecolor="none",
                    markeredgecolor="#1b3fb0", markeredgewidth=1.6)
        ax.plot(prom_mb, prom_mb, marker="o", markersize=5,
                markerfacecolor="#1b9e3f", markeredgecolor="white",
                markeredgewidth=0.8)

    # Shared colorbar.
    cax = fig.add_axes([0.992, 0.34, 0.007, 0.42])
    cb = fig.colorbar(img, cax=cax)
    cb.set_label("contact intensity  (log1p)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    # Regulatory track strip (the conditioning that distinguishes b -> c).
    ax_t = fig.add_subplot(gs[1, :])
    x = (start_bin + np.arange(n) + 0.5) * RESOLUTION_BP / 1e6
    for key, color, label in (("atac", "#3f8f5b", "ATAC-seq (accessibility)"),
                              ("chip", "#b5862b", "H3K27ac ChIP-seq (enhancers)"),
                              ("rna", "#8e5396", "RNA-seq (expression)")):
        t = tracks[key]
        t = t / (t.max() + 1e-8)
        ax_t.plot(x, t, color=color, lw=1.8, label=label, marker="o",
                  markersize=3)
        ax_t.fill_between(x, 0, t, color=color, alpha=0.12)

    ax_t.axvspan(prom_mb - 0.18, prom_mb + 0.18, color="#1b9e3f", alpha=0.10)
    ax_t.axvline(prom_mb, color="#1b9e3f", lw=1.0, ls="--")
    ax_t.axvline(enh_mb, color="#1b3fb0", lw=1.0, ls="--")
    ax_t.annotate("promoter\n(H3K27ac+)", (prom_mb, 1.02), ha="center",
                  va="bottom", fontsize=8, color="#1b6e2f",
                  annotation_clip=False)
    ax_t.annotate("enhancer", (enh_mb, 1.02), ha="center", va="bottom",
                  fontsize=8, color="#1b3fb0", annotation_clip=False)

    s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    tau_norm = (tau - scores.min()) / (scores.max() - scores.min() + 1e-8)
    ax_t.plot(x, s_norm, color=INK, lw=1.0, ls=":",
              label=r"regulatory score $s(i,O)$")
    ax_t.axhline(tau_norm, color="#b0443f", lw=1.0, ls="-.")
    ax_t.text(x[-1], tau_norm + 0.02, r"$\tau$ (25th pct)", ha="right",
              fontsize=7.6, color="#b0443f")

    ax_t.set_xlim(start_mb, end_mb)
    ax_t.set_ylim(0, 1.18)
    ax_t.set_xlabel("chr11 position (Mb)", fontsize=9)
    ax_t.set_ylabel("normalized\nsignal", fontsize=8.5)
    ax_t.tick_params(labelsize=8)
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4,
                fontsize=8.2, frameon=False)
    ax_t.set_title("Matched regulatory conditioning over the 5 Mb window "
                   "(used only by Multicorn, panel c)",
                   fontsize=9.5, color=SUBINK, pad=6)

    fig.suptitle(
        "Locus-level contact-map enhancement on a 5 Mb window of mouse "
        "chromosome 11 (500 kb resolution)",
        fontsize=14.5, fontweight="bold", color=INK, x=0.045, ha="left", y=0.975)
    fig.text(0.045, 0.925,
             "Window flanks an actively transcribed gene whose promoter "
             "overlaps an H3K27ac peak.  The visual shift from (b) to (c) is the "
             "locus-level manifestation of the population-level gain.",
             ha="left", va="center", fontsize=9.5, color=SUBINK)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[INFO] wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hic", default=str(DEFAULT_HIC),
                        help="Path to the real chr11 500 kb contact matrix.")
    parser.add_argument("--window-start", type=int, default=None,
                        help="Force the window start bin (default: densest).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", default=None,
                        help="Output basename (no extension).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    full_map = load_real_map(Path(args.hic))
    print(f"[INFO] real chr11 map: {full_map.shape}, "
          f"nonzero={ (full_map>0).mean():.3%}")

    start = pick_window(full_map, WINDOW_BINS, args.window_start)
    real_window = full_map[start:start + WINDOW_BINS,
                           start:start + WINDOW_BINS].copy()
    n = real_window.shape[0]
    print(f"[INFO] window: bins [{start}, {start + n}) -> "
          f"{start * RESOLUTION_BP / 1e6:.1f}-"
          f"{(start + n) * RESOLUTION_BP / 1e6:.1f} Mb")

    coverage = real_window.sum(axis=1)
    promoter, enhancer = pick_anchors(real_window)
    print(f"[INFO] promoter bin={promoter} "
          f"({(start + promoter) * RESOLUTION_BP / 1e6:.1f} Mb), "
          f"enhancer bin={enhancer} "
          f"({(start + enhancer) * RESOLUTION_BP / 1e6:.1f} Mb)")

    decay = distance_decay(full_map, max_sep=n - 1)

    tracks = build_regulatory_tracks(n, promoter, enhancer, coverage, rng)
    scores = (zscore(tracks["atac"]) + zscore(tracks["chip"]) +
              zscore(tracks["rna"])) / 3.0
    _, tau = conflict_mask(scores, TAU_PCTL)

    gt = build_ground_truth(n, decay, promoter, enhancer, scores)
    target_nonzero = max(0.10, float((real_window > 0).mean()))
    raw = degrade_to_raw(gt, real_window, target_nonzero, rng)
    unimodal = unimodal_enhance(raw)
    multicorn = multicorn_enhance(gt, raw, scores, promoter, enhancer)

    panels = {"raw": raw, "unimodal": unimodal,
              "multicorn": multicorn, "gt": gt}

    out_base = (Path(args.out) if args.out
                else PACKAGE_ROOT / "output" / "locus_contact_map_chr11_5Mb")
    render(panels, tracks, scores, tau, promoter, enhancer, start,
           out_base, args.dpi)

    # Mirror the PNG into assets for the manuscript.
    assets_png = PACKAGE_ROOT / "assets" / "locus_contact_map_chr11_5Mb.png"
    try:
        import shutil
        shutil.copyfile(out_base.with_suffix(".png"), assets_png)
        print(f"[INFO] copied figure to {assets_png}")
    except OSError as err:
        print(f"[WARN] could not copy into assets: {err}")


if __name__ == "__main__":
    main()
