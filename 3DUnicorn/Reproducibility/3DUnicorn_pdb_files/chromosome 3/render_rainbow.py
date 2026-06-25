"""Render a 3DUnicorn chromosome PDB backbone as a rainbow-colored trace.

Colors the Calpha trace from the N-terminus (blue) to the C-terminus (red),
matching UCSF Chimera's `rainbow` coloring, and saves a high-resolution PNG.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    from scipy.interpolate import splprep, splev
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def read_ca_coords(pdb_path: str) -> np.ndarray:
    coords = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
    return np.asarray(coords, dtype=float)


def smooth(coords: np.ndarray, factor: int = 8) -> np.ndarray:
    """Spline-smooth the trace for a clean tube-like appearance."""
    if not HAVE_SCIPY or len(coords) < 4:
        return coords
    tck, _ = splprep(coords.T, s=0, k=3)
    u = np.linspace(0.0, 1.0, len(coords) * factor)
    return np.asarray(splev(u, tck)).T


def make_segments(coords: np.ndarray) -> np.ndarray:
    pts = coords.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def render(pdb_path: str, out_path: str, linewidth: float = 3.0,
           cmap_name: str = "rainbow", elev: float = 18.0, azim: float = -60.0):
    coords = read_ca_coords(pdb_path)
    if coords.size == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    sm = smooth(coords)
    segs = make_segments(sm)

    t = np.linspace(0.0, 1.0, len(segs))
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segs, cmap=cmap, linewidths=linewidth,
                          capstyle="round", joinstyle="round")
    lc.set_array(t)
    ax.add_collection3d(lc)

    # Tight, equal-aspect bounds
    mins = sm.min(axis=0)
    maxs = sm.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0 * 1.05
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}  ({len(coords)} CA atoms)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", default=os.path.join(here, "chr3_tuple_500kb_new_1.pdb"))
    p.add_argument("--out", default=os.path.join(here, "chromosome 3 chimera rainbow.png"))
    p.add_argument("--linewidth", type=float, default=3.0)
    p.add_argument("--cmap", default="rainbow")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    args = p.parse_args()
    render(args.pdb, args.out, args.linewidth, args.cmap, args.elev, args.azim)
