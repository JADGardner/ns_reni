"""Programmatic conditioning-methods figure (thesis Fig: model_types).

Recreates Figures/model_types.pdf with the labels baked in (they were
previously TikZ overlay nodes in the chapter): the three neural-field
conditioning schemes compared in RENI++ — condition-by-concatenation,
hypernetwork, and attention.

Run from the ns_reni repo root (CPU, no checkpoint needed):

    PYTHONPATH=. python scripts/figures/fig_model_types.py
"""

from __future__ import annotations

import argparse
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]

# SphereJEPA palette (as in fig_model_overview): pastel fills with saturated
# matching strokes. Cross-figure semantics: blue = latent (z), pale yellow =
# direction input (lambda(d)), green = network bodies.
BLUE_CELL, BLUE_CELL_STROKE = "#EAF1FF", (0.23, 0.37, 0.68, 0.55)
DIR_CELL, DIR_CELL_STROKE = "#FFF3D6", (0.79, 0.60, 0.18, 0.60)
CELL_STROKE = (0.0, 0.0, 0.0, 0.35)
GREEN_FILL, GREEN_STROKE, GREEN_TEXT = "#E1F5D9", "#3D8B2F", "#1F4914"
BLUE_FILL, BLUE_STROKE, BLUE_TEXT = "#D7E5FF", "#3A5FAD", "#0B2A66"
CONTAINER_FILL, CONTAINER_STROKE = "#F4F4F4", (0.0, 0.0, 0.0, 0.40)
NET_FILL = GREEN_FILL
NET_STROKE = GREEN_STROKE
ARROW_COLOR = "#333333"


def cell(ax, x, y, s, facecolor, rounding=0.05, edgecolor=None):
    if edgecolor is None:
        edgecolor = {BLUE_CELL: BLUE_CELL_STROKE, DIR_CELL: DIR_CELL_STROKE}.get(
            facecolor, CELL_STROKE)
    ax.add_patch(FancyBboxPatch(
        (x, y), s, s, boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.9))


def cell_column(ax, x, y_center, n, s=0.42, colors=None, gap=0.03):
    """n stacked cells centred on y_center; returns (top_y, bottom_y)."""
    total = n * s + (n - 1) * gap
    y0 = y_center + total / 2 - s
    for i in range(n):
        c = colors[i] if colors else "white"
        cell(ax, x, y0 - i * (s + gap), s, c)
    return x, y_center


def trapezoid(ax, x0, x1, y_center, half_h0, half_h1,
              facecolor=None, edgecolor=None):
    ax.add_patch(Polygon(
        [(x0, y_center - half_h0), (x0, y_center + half_h0),
         (x1, y_center + half_h1), (x1, y_center - half_h1)],
        closed=True, facecolor=facecolor or NET_FILL,
        edgecolor=edgecolor or NET_STROKE, linewidth=1.2,
        joinstyle="round"))


def arrow(ax, start, end, connectionstyle="arc3", lw=1.1):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=ARROW_COLOR, shrinkA=0, shrinkB=0,
        connectionstyle=connectionstyle))


def text(ax, x, y, s, size=9, rotation=0, color="black", **kw):
    ax.text(x, y, s, fontsize=size, rotation=rotation,
            ha="center", va="center", color=color, **kw)


def output_column(ax, x, y_center, n=7, s=0.36):
    total = n * s + (n - 1) * 0.03
    y0 = y_center + total / 2 - s
    for i in range(n):
        cell(ax, x, y0 - i * (s + 0.03), s, "white", rounding=0.04)
    return x + s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "publication" / "figures" / "model_types")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--svg", action="store_true")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(13.6, 4.4), dpi=args.dpi)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Panel 1: condition-by-concatenation (MLP widens toward outputs) ──
    cy = 2.3
    cell_column(ax, 0.72, cy, 4, colors=[BLUE_CELL] * 3 + [DIR_CELL])
    text(ax, 0.58, cy + 0.25, r"$\mathbf{z}$", size=12)
    text(ax, 0.50, cy - 0.68, r"$\lambda(\mathbf{d})$", size=11)
    trapezoid(ax, 1.28, 3.05, cy, 0.62, 1.10)
    text(ax, 2.16, cy, "MLP", size=12, color=GREEN_TEXT)
    output_column(ax, 3.20, cy)
    text(ax, 3.74, cy, "outputs", size=11, rotation=90)
    text(ax, 2.05, 0.42, "Condition-by-Concat", size=12)

    # ── Panel 2: hypernetwork ───────────────────────────────────────────
    hy = 3.35   # hyper row
    my = 1.95   # main row
    cell_column(ax, 4.62, hy, 3, colors=[BLUE_CELL] * 3)
    text(ax, 4.50, hy, r"$\mathbf{z}$", size=12)
    trapezoid(ax, 5.26, 6.66, hy, 0.36, 0.68,
              facecolor=BLUE_FILL, edgecolor=BLUE_STROKE)
    text(ax, 6.00, hy + 0.16, "Hyper", size=10.5, color=BLUE_TEXT)
    text(ax, 6.00, hy - 0.20, "MLP", size=10.5, color=BLUE_TEXT)
    # weights flow into the main MLP: right, then straight down (elbow)
    arrow(ax, (6.66, hy), (7.72, my + 0.86),
          connectionstyle="angle,angleA=0,angleB=90,rad=0.15", lw=1.2)
    text(ax, 8.04, hy, r"$\mathbf{W}$", size=12)
    trapezoid(ax, 6.90, 8.52, my, 0.55, 1.05)
    text(ax, 7.72, my, "MLP", size=12, color=GREEN_TEXT)
    cell(ax, 5.94, my - 0.21, 0.42, DIR_CELL)
    text(ax, 5.70, my, r"$\lambda(\mathbf{d})$", size=11)
    arrow(ax, (6.38, my), (6.88, my))
    output_column(ax, 8.68, my)
    text(ax, 9.22, my, "outputs", size=11, rotation=90)
    text(ax, 7.05, 0.42, "Hypernetwork", size=12)

    # ── Panel 3: attention ──────────────────────────────────────────────
    ay = 2.15
    container = FancyBboxPatch(
        (10.62, ay - 1.05), 2.02, 2.10,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor=CONTAINER_FILL, edgecolor=CONTAINER_STROKE, linewidth=1.1)
    ax.add_patch(container)
    # attention block with K / V / Q inside
    ax.add_patch(FancyBboxPatch(
        (10.80, ay - 0.85), 0.92, 1.70,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=BLUE_FILL, edgecolor=BLUE_STROKE, linewidth=1.0))
    text(ax, 11.02, ay + 0.58, "K", size=10, color=BLUE_TEXT)
    text(ax, 11.02, ay + 0.16, "V", size=10, color=BLUE_TEXT)
    text(ax, 11.02, ay - 0.62, "Q", size=10, color=BLUE_TEXT)
    # small MLP trapezoid feeding the outputs
    trapezoid(ax, 11.88, 12.48, ay, 0.55, 0.28)
    text(ax, 12.16, ay, "MLP", size=9.5, rotation=90, color=GREEN_TEXT)
    # z (keys/values) and lambda(d) (queries)
    cell_column(ax, 9.78, ay + 0.45, 3, colors=[BLUE_CELL] * 3)
    text(ax, 9.66, ay + 0.45, r"$\mathbf{z}$", size=12)
    arrow(ax, (10.20, ay + 0.45), (10.78, ay + 0.45))
    cell(ax, 9.78, ay - 1.00, 0.42, DIR_CELL)
    text(ax, 9.54, ay - 0.79, r"$\lambda(\mathbf{d})$", size=11)
    arrow(ax, (10.20, ay - 0.79), (10.78, ay - 0.66))
    output_column(ax, 12.80, ay)
    text(ax, 13.34, ay, "outputs", size=11, rotation=90)
    text(ax, 11.62, 0.42, "Attention", size=12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.output}.png", dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight", pad_inches=0.02)
    if args.svg:
        fig.savefig(f"{args.output}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"[saved] {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
