"""Render the Multicorn architecture as a flat, publication-style figure.

The diagram is generated programmatically with matplotlib so it stays in sync
with the implementation (tensor shapes, layer widths and iteration count are
read from the model code, not invented). The visual style is deliberately plain
- flat fills, thin uniform strokes, a restrained academic palette and monospace
shape annotations - so it reads like a hand-drawn methods figure rather than a
glossy auto-generated graphic.

Run:
    python generate_architecture.py            # writes PNG + PDF + SVG
    python generate_architecture.py --show      # also open an interactive window

Outputs are written next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
# Muted, print-friendly palette. Each stage gets one accent colour; fills are
# pale tints of that accent rather than gradients.
INK = "#1f2329"          # primary text / strokes
SUBINK = "#5b6470"       # secondary text
GRID = "#c9ced6"         # neutral connector colour

PALETTE = {
    "hic":   {"line": "#2f6db0", "fill": "#eaf1f8"},   # contact map (blue)
    "atac":  {"line": "#3f8f5b", "fill": "#eaf4ed"},   # accessibility (green)
    "chip":  {"line": "#b5862b", "fill": "#f7f0df"},   # enhancers (ochre)
    "rna":   {"line": "#8e5396", "fill": "#f2eaf4"},   # expression (purple)
    "fuse":  {"line": "#4b5bb0", "fill": "#ecedf7"},   # fusion (indigo)
    "loop":  {"line": "#b8643a", "fill": "#f8efe8"},   # DAN loop (terracotta)
    "out":   {"line": "#b0443f", "fill": "#f8eceb"},   # output (red)
    "next":  {"line": "#4a5560", "fill": "#eef0f2"},   # 3DUnicorn (slate)
}

# Prefer a clean sans for labels; matplotlib falls back gracefully.
for candidate in ("DejaVu Sans", "Arial", "Helvetica"):
    if any(candidate == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["mathtext.fontset"] = "cm"

MONO = {"family": "monospace"}


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def box(ax, xy, w, h, *, key, title, lines=None, title_size=10.5,
        line_size=8.6, pad=0.06, rounding=0.04, lw=1.3, alpha=1.0,
        fill=None, edge=None):
    """Draw a flat rounded rectangle with a bold title and detail lines."""
    x, y = xy
    style = PALETTE.get(key, {"line": INK, "fill": "#ffffff"})
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=edge or style["line"],
        facecolor=fill or style["fill"],
        alpha=alpha,
        mutation_aspect=1.0,
        zorder=2,
    )
    ax.add_patch(patch)

    cx = x + w / 2
    if lines:
        ax.text(cx, y + h - pad - 0.02, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=style["line"],
                zorder=3)
        ystep = (h - 2 * pad - 0.10) / max(len(lines), 1)
        ytop = y + h - pad - 0.18
        for i, ln in enumerate(lines):
            txt, mono = (ln if isinstance(ln, tuple) else (ln, False))
            ax.text(cx, ytop - i * ystep, txt, ha="center", va="top",
                    fontsize=line_size, color=INK,
                    fontfamily="monospace" if mono else None, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=style["line"],
                zorder=3)
    return patch


def arrow(ax, p0, p1, *, color=GRID, lw=1.5, ls="-", rad=0.0,
          label=None, label_off=(0, 0.10), label_size=8.0, mono=True,
          shrink=2.0):
    """Connector arrow with an optional shape label at its midpoint."""
    ax.add_patch(FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>", mutation_scale=12,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw, linestyle=ls, color=color,
        shrinkA=shrink, shrinkB=shrink, zorder=1.5,
    ))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + label_off[0], my + label_off[1], label,
                ha="center", va="center", fontsize=label_size,
                color=SUBINK, fontfamily="monospace" if mono else None,
                zorder=3)


def stage_label(ax, x, text):
    ax.text(x, 6.62, text, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=SUBINK)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure():
    fig, ax = plt.subplots(figsize=(14.5, 8.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title block
    ax.text(0.25, 8.62, "Multicorn", ha="left", va="center",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(0.27, 8.18,
            "Multimodal blind super-resolution for single-cell Hi-C "
            "enhancement and 3D genome reconstruction",
            ha="left", va="center", fontsize=10.5, color=SUBINK)

    # Stage headers
    stage_label(ax, 2.3, "1  Inputs & encoders")
    stage_label(ax, 5.9, "2  Multimodal fusion")
    stage_label(ax, 10.2, "3  Deep alternating restoration  (N = 5)")
    stage_label(ax, 14.3, "4  Output")

    # ---- Stage 1: inputs and encoders -------------------------------------- #
    ax.text(2.3, 6.18, "input", ha="center", fontsize=8.2, color=SUBINK,
            style="italic")
    ax.text(3.55, 6.18, "encoder", ha="center", fontsize=8.2, color=SUBINK,
            style="italic")

    # scHi-C contact map -> 2D CNN
    box(ax, (0.55, 5.35), 1.55, 0.78, key="hic",
        title="scHi-C map  L", lines=[("1 x 40 x 40", True)])
    box(ax, (2.85, 5.35), 1.55, 0.78, key="hic",
        title="2D CNN", lines=[("Conv+2xResBlock", False), ("-> 128 x 8 x 8", True)],
        line_size=8.0)

    # Three regulatory tracks -> MLP encoders
    tracks = [
        ("atac", "ATAC-seq", "accessibility a"),
        ("chip", "ChIP H3K27ac", "enhancers c"),
        ("rna",  "RNA-seq", "expression r"),
    ]
    y0 = 4.35
    for i, (key, name, sub) in enumerate(tracks):
        yy = y0 - i * 0.92
        box(ax, (0.55, yy), 1.55, 0.74, key=key, title=name,
            lines=[(sub, False)], line_size=8.2)
        box(ax, (2.85, yy), 1.55, 0.74, key=key, title="MLP",
            lines=[("Linear 128->64", True)], line_size=7.8)
        arrow(ax, (2.12, yy + 0.37), (2.83, yy + 0.37),
              color=PALETTE[key]["line"], lw=1.2, label=None)

    # input -> encoder arrows for Hi-C
    arrow(ax, (2.12, 5.74), (2.83, 5.74), color=PALETTE["hic"]["line"], lw=1.3)

    # ---- Stage 2: fusion ---------------------------------------------------- #
    # latents flowing right out of encoders
    arrow(ax, (4.42, 5.74), (5.05, 5.74), color=PALETTE["hic"]["line"],
          lw=1.4, label="F_HiC", label_off=(0, 0.16))
    latent_labels = ["z_a (64)", "z_c (64)", "z_r (64)"]
    for i, lab in enumerate(latent_labels):
        yy = (y0 - i * 0.92) + 0.37
        arrow(ax, (4.42, yy), (5.05, yy),
              color=PALETTE[tracks[i][0]]["line"], lw=1.2,
              label=lab, label_off=(0, 0.15), label_size=7.4)

    box(ax, (5.05, 1.65), 1.75, 3.05, key="fuse",
        title="Multimodal fusion",
        lines=[
            ("z = [z_a||z_c||z_r]", True),
            ("z in R^192", True),
            ("", False),
            ("phi: 192 -> 128", True),
            ("", False),
            ("F_cond =", False),
            ("F_HiC + phi(z)", True),
            ("", False),
            ("reinjected each", False),
            ("iteration", False),
        ],
        title_size=10.0, line_size=8.0)

    arrow(ax, (6.82, 3.18), (7.55, 3.18), color=PALETTE["fuse"]["line"],
          lw=1.6, label="F_cond", label_off=(0, 0.17))

    # ---- Stage 3: DAN restoration loop ------------------------------------- #
    # outer container
    ax.add_patch(FancyBboxPatch(
        (7.55, 1.35), 5.05, 5.05,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=1.6, edgecolor=PALETTE["loop"]["line"],
        facecolor=PALETTE["loop"]["fill"], zorder=1.2))
    ax.text(10.07, 6.10, "Deep Alternating Network (DAN)", ha="center",
            va="center", fontsize=10.2, fontweight="bold",
            color=PALETTE["loop"]["line"])

    # Estimator and Restorer
    box(ax, (7.85, 4.55), 1.95, 1.20, key="loop", fill="#ffffff",
        title="Estimator",
        lines=[("refine kernel T", False), ("MLP -> 512", True),
               ("min ||L - T(H)||^2", False)],
        line_size=8.0, title_size=10.0)
    box(ax, (10.35, 4.55), 1.95, 1.20, key="loop", fill="#ffffff",
        title="Restorer",
        lines=[("HiCEnc + fusion", False), ("+ Conv decoder", False),
               ("reconstruct H", False)],
        line_size=8.0, title_size=10.0)

    # alternation arrows between the two
    arrow(ax, (9.82, 5.35), (10.33, 5.35), color=PALETTE["loop"]["line"],
          lw=1.5, label="T", label_off=(0, 0.16), mono=False)
    arrow(ax, (10.33, 4.85), (9.82, 4.85), color=PALETTE["loop"]["line"],
          lw=1.5, label="H", label_off=(0, -0.20), mono=False)

    # recurrence arrow: H feeds back as the next LR estimate
    ax.add_patch(FancyArrowPatch(
        (11.32, 4.53), (11.32, 3.55),
        arrowstyle="-|>", mutation_scale=12,
        connectionstyle="arc3,rad=0", linewidth=1.5,
        color=PALETTE["loop"]["line"], zorder=1.6))
    ax.add_patch(FancyArrowPatch(
        (11.32, 3.55), (8.83, 3.55),
        arrowstyle="-", mutation_scale=12,
        connectionstyle="arc3,rad=0", linewidth=1.5,
        color=PALETTE["loop"]["line"], zorder=1.6))
    ax.add_patch(FancyArrowPatch(
        (8.83, 3.55), (8.83, 4.53),
        arrowstyle="-|>", mutation_scale=12,
        connectionstyle="arc3,rad=0", linewidth=1.5,
        color=PALETTE["loop"]["line"], zorder=1.6))
    ax.text(10.07, 3.40, "H_t  ->  H_{t+1}   (blend 0.5)", ha="center",
            va="center", fontsize=8.0, color=SUBINK, fontfamily="monospace")

    # biologically constrained objective panel
    box(ax, (7.85, 1.62), 4.45, 1.55, key="loop", fill="#fbf4ee",
        title="Biologically constrained objective", title_size=9.6,
        lines=[
            (r"$J = \|L - T(H)\|^2 + \beta\,P(H) + \gamma\,B(H,O)$", False),
            ("B penalizes contacts between inaccessible,", False),
            ("unmarked, silent bins   w_ij = 1/(1+|i-j|/d0)", False),
            ("O = (a, c, r) normalized tracks      gamma = 0.1", False),
        ],
        line_size=8.0)

    # ---- Stage 4: output + downstream -------------------------------------- #
    arrow(ax, (12.62, 5.15), (13.45, 5.15), color=PALETTE["out"]["line"],
          lw=1.7, label="H*", label_off=(0, 0.18), mono=False)

    box(ax, (13.45, 4.45), 1.95, 1.30, key="out",
        title="Enhanced HR map",
        lines=[("H*  (super-resolved", False), ("contact matrix)", False),
               ("+ degradation tail", False)],
        line_size=8.0, title_size=9.8)

    arrow(ax, (14.42, 4.43), (14.42, 3.55), color=PALETTE["next"]["line"],
          lw=1.6)

    box(ax, (13.45, 2.20), 1.95, 1.30, key="next",
        title="3DUnicorn",
        lines=[("max-likelihood", False), ("3D reconstruction", False),
               ("vs 3D-FISH", False)],
        line_size=8.0, title_size=9.8)

    # ---- Legend ------------------------------------------------------------ #
    legend_items = [
        ("hic",  "Hi-C contact map"),
        ("atac", "ATAC accessibility"),
        ("chip", "H3K27ac enhancers"),
        ("rna",  "RNA expression"),
        ("loop", "DAN restoration"),
    ]
    lx = 0.55
    ly = 0.85
    ax.text(lx, ly + 0.42, "Legend", fontsize=9.0, fontweight="bold",
            color=SUBINK)
    for i, (key, label) in enumerate(legend_items):
        x = lx + i * 2.55
        ax.add_patch(mpatches.Rectangle(
            (x, ly), 0.28, 0.20, linewidth=1.1,
            edgecolor=PALETTE[key]["line"], facecolor=PALETTE[key]["fill"]))
        ax.text(x + 0.38, ly + 0.10, label, fontsize=8.2, va="center",
                color=INK)

    ax.text(15.45, 0.30,
            "Multicorn = Multimodal Unicorn  (extends ScUnicorn blind SR)",
            ha="right", va="center", fontsize=7.8, color=SUBINK,
            style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="Output basename (without extension).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window after saving.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    base = Path(args.out) if args.out else here / "multicorn_architecture"

    fig = build_figure()
    for ext in ("png", "pdf", "svg"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"wrote {path}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
