"""Rotation-equivariance error curves (thesis Fig: rotation_error).

Plots the relative latent error against rotation angle for the eval-fit
optimiser (Adam) and plain gradient descent, from the dense sweep produced
by make_rotation_error_table.py (rotated-directions target, full-image
batches). Adam's error rises smoothly with angle and collapses at 90/180/270
degrees - the rotations whose latent representation is a signed permutation,
the only rotations that commute with Adam's per-coordinate second-moment
normalisation. Gradient descent is rotation-equivariant at every angle.

    PYTHONPATH=.:scripts/figures python scripts/figures/fig_rotation_error.py \
        --sweep publication/tables/rotation_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman", "Times New Roman", "Times",
        "Liberation Serif", "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

ADAM_COLOR = "#3A5FAD"   # SphereJEPA blue
GD_COLOR = "#3D8B2F"     # SphereJEPA green


def series(metrics: dict, optimiser: str):
    angles, mean, std = [], [], []
    for key, m in metrics.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        angle, target, batch, opt = parts
        if target != "rotated_directions" or batch != "full_image" or opt != optimiser:
            continue
        angles.append(float(angle))
        mu, sd = m["rel_err_procrustes"]
        mean.append(mu)
        std.append(sd)
    order = np.argsort(angles)
    return (np.asarray(angles)[order], np.asarray(mean)[order],
            np.asarray(std)[order])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path,
                        default=REPO_ROOT / "publication" / "tables" / "rotation_sweep.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "rotation_error_thesis",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    metrics = json.loads(args.sweep.read_text())["metrics"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=args.dpi)
    for opt, color, label in (("adam", ADAM_COLOR, "Adam (eval-fit optimiser)"),
                              ("gd", GD_COLOR, "Gradient descent")):
        a, mu, sd = series(metrics, opt)
        ax.plot(a, mu, color=color, lw=1.8, label=label, zorder=3)
        ax.fill_between(a, mu - sd, mu + sd, color=color, alpha=0.18, lw=0,
                        zorder=2)

    for x in (90, 180, 270):
        ax.axvline(x, color="0.75", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(90, ax.get_ylim()[1], " signed-permutation rotations",
            fontsize=8, color="0.45", va="top", ha="left")

    ax.set_xlabel(r"Rotation angle about the invariance axis (degrees)")
    ax.set_ylabel(r"Relative latent error")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.output}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight")
    print(f"[saved] {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
