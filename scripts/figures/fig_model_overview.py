"""Programmatic RENI++ model overview figure.

Recreates the conditional spherical neural field overview diagram without
TikZ or external drawing tools. The output sphere is textured from an EXR
environment map, which can be overridden with --envmap.

Run from the ns_reni repo root:

    PYTHONPATH=. python scripts/figures/fig_model_overview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman",
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPO_ROOT.parents[1]


def default_envmap() -> Path:
    candidates = [
        PHD_ROOT / "data/RENI_HDR_512x1024/openfootage_hdri/00266_OpenfootageNET_Pinzgau_LOW.exr",
        PHD_ROOT / "data/RENI_HDR_512x1024/iHDRI/Lookout-Point-Enderndorf-4K.exr",
        PHD_ROOT / "data/RENI_HDR_512x1024/iHDRI/Road-Island-Vis-4K.exr",
        PHD_ROOT / "data/RENI_HDR_512x1024/iHDRI/Snowy-Path-Uhlberg-4K.exr",
        REPO_ROOT / "data/RENI_HDR/test/00001.exr",
        REPO_ROOT / "data/RENI_HDR/val/00001.exr",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No default RENI_HDR EXR found; pass --envmap /path/to/file.exr"
    )


def clean_hdr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image[:, :, :3].astype(np.float32, copy=False)
    finite = np.isfinite(image)
    finite_max = float(np.max(image[finite])) if np.any(finite) else 0.0
    image = np.nan_to_num(image, nan=0.0, posinf=finite_max, neginf=0.0)
    positive = image[image > 0]
    floor = float(np.min(positive)) if positive.size else 0.0
    image[image <= 0] = floor
    return image


def read_exr(path: Path) -> np.ndarray:
    try:
        import pyexr
    except ImportError as exc:
        raise SystemExit(
            "pyexr is required to read EXR inputs. Run this script in the "
            "RENI++/research environment or container."
        ) from exc
    return clean_hdr(pyexr.read(str(path)))


def tonemap(image: np.ndarray, percentile: float = 99.4) -> np.ndarray:
    scale = float(np.percentile(image, percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(image)) if image.size else 1.0
    image = np.log1p(np.maximum(image, 0.0)) / np.log1p(scale)
    return np.clip(image, 0.0, 1.0) ** (1.0 / 2.2)


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    ryaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rroll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ryaw @ rx @ rroll


def sample_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    u = np.mod(u, w)
    v = np.clip(v, 0, h - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = (x0 + 1) % w
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (u - x0)[..., None]
    wy = (v - y0)[..., None]
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def envmap_sphere_image(
    envmap: np.ndarray,
    size: int = 520,
    yaw: float = 210.0,
    pitch: float = -8.0,
    roll: float = 0.0,
) -> np.ndarray:
    """Orthographically project an equirectangular envmap onto a sphere."""
    display = tonemap(envmap)
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rr = xx**2 + yy**2
    mask = rr <= 1.0
    zz = np.sqrt(np.maximum(1.0 - rr, 0.0))
    dirs = np.stack([xx, yy, zz], axis=-1)
    dirs = dirs @ rotation_matrix(yaw, pitch, roll).T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
    u = (lon / (2 * np.pi) + 0.5) * display.shape[1]
    v = (0.5 + lat / np.pi) * display.shape[0]
    rgb = sample_bilinear(display, u, v)

    light = np.array([-0.35, 0.55, 0.75])
    light = light / np.linalg.norm(light)
    shade = 0.72 + 0.35 * np.clip(np.sum(dirs * light, axis=-1), 0.0, 1.0)
    edge = np.clip((1.0 - rr) / 0.18, 0.0, 1.0)
    rgb = np.clip(rgb * shade[..., None] * (0.72 + 0.28 * edge[..., None]), 0, 1)

    alpha = np.zeros((size, size, 1), dtype=np.float32)
    alpha[mask] = 1.0
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return rgba


def gray_sphere_image(size: int = 420) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rr = xx**2 + yy**2
    mask = rr <= 1
    zz = np.sqrt(np.maximum(1 - rr, 0))
    light = np.array([-0.5, 0.65, 0.75])
    light = light / np.linalg.norm(light)
    dirs = np.stack([xx, yy, zz], axis=-1)
    shade = 0.72 + 0.25 * np.clip(np.sum(dirs * light, axis=-1), 0, 1)
    edge = np.clip((1 - rr) / 0.22, 0, 1)
    gray = np.clip(shade * (0.82 + 0.18 * edge), 0, 1)
    rgba = np.dstack([gray, gray, gray, mask.astype(np.float32)])
    return rgba


def arrow(ax, start, end, color="black", lw=1.4, style="-|>",
          mutation_scale=16, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    return patch


def draw_text(ax, x, y, text, size=12, rotation=0, ha="center", va="center",
              **kwargs):
    defaults = dict(fontsize=size, rotation=rotation, ha=ha, va=va, color="black")
    defaults.update(kwargs)
    return ax.text(x, y, text, **defaults)


# SphereJEPA figure palette (scripts/sphere_jepa/figures/svg_pieces.py):
# pastel fills with saturated matching strokes, rx = 12/130 of block height,
# cell rounding = 18% of cell, soft rgba(0,0,0,0.3) strokes on grids.
BLUE_FILL, BLUE_STROKE, BLUE_TEXT = "#D7E5FF", "#3A5FAD", "#0B2A66"
GREEN_FILL, GREEN_STROKE = "#E1F5D9", "#3D8B2F"
CELL_FILL, CELL_STROKE = "#EAF1FF", (0.23, 0.37, 0.68, 0.55)
STACK_FILL, STACK_STROKE = "#FFF3D6", (0.79, 0.60, 0.18, 0.60)
ARROW_COLOR = "#333333"
VEC_COLOR = "#7C9AD6"  # pastel blue for on-sphere direction/latent vectors


def rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, rounding=None,
                zorder=2):
    rounding = rounding if rounding is not None else 0.18 * min(w, h)
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_axes(ax, cx, cy, r):
    origin = (cx, cy)
    endpoints = {
        "x": (cx + 1.05 * r, cy - 0.22 * r),
        "y": (cx, cy + 1.08 * r),
        "z": (cx - 0.72 * r, cy - 0.82 * r),
    }
    for label, end in endpoints.items():
        arrow(ax, origin, end, lw=1.1, mutation_scale=14)
        offsets = {
            "x": (0.08, -0.04),
            "y": (0.0, 0.10),
            "z": (-0.08, -0.11),
        }
        dx, dy = offsets[label]
        draw_text(ax, end[0] + dx, end[1] + dy, label, size=13)


def draw_query_sphere(ax, cx, cy, r):
    ax.imshow(gray_sphere_image(), extent=(cx - r, cx + r, cy - r, cy + r),
              zorder=0)
    draw_axes(ax, cx, cy, r)
    arrow(ax, (cx, cy), (cx + 0.35 * r, cy + 0.56 * r),
          color=VEC_COLOR, lw=1.3, mutation_scale=13)
    draw_text(ax, cx + 0.44 * r, cy + 0.64 * r, r"$\mathbf{d}$", size=12)
    ax.plot([cx + 0.35 * r, cx + 0.35 * r],
            [cy + 0.56 * r, cy - 0.53 * r],
            "--", color="black", linewidth=0.9, dashes=(5, 5), alpha=0.8)
    ax.plot([cx - 0.36 * r, cx + 0.35 * r],
            [cy - 0.38 * r, cy - 0.53 * r],
            "--", color="black", linewidth=0.9, dashes=(5, 5), alpha=0.8)
    ax.plot([cx + 0.35 * r, cx + 0.69 * r],
            [cy - 0.53 * r, cy - 0.18 * r],
            "--", color="black", linewidth=0.9, dashes=(5, 5), alpha=0.8)
    draw_text(ax, cx - 0.78 * r, cy + 0.86 * r, "Query", size=12, rotation=42)


def draw_latent_sphere(ax, cx, cy, r):
    ax.imshow(gray_sphere_image(), extent=(cx - r, cx + r, cy - r, cy + r),
              zorder=0)
    draw_axes(ax, cx, cy, r)
    vectors = [
        (-0.72, -0.28), (-0.56, 0.28), (-0.18, 0.64), (0.14, 0.70),
        (0.42, 0.34), (0.60, -0.03), (0.12, -0.42), (-0.05, -0.25),
    ]
    for vx, vy in vectors:
        arrow(ax, (cx, cy), (cx + vx * r, cy + vy * r),
              color=VEC_COLOR, lw=1.2, mutation_scale=12)
    draw_text(ax, cx + 0.34 * r, cy + 0.50 * r, r"$\mathbf{Z}$", size=13)
    draw_text(ax, cx - 0.85 * r, cy + 0.82 * r, "Latent Code",
              size=12, rotation=48)


def draw_vector_stack(ax, x, y, w, h):
    labels = [r"$x$", r"$y$", r"$z$"]
    gap = 0.02
    for i, label in enumerate(labels):
        yy = y + (2 - i) * h / 3
        rounded_box(ax, x, yy + gap / 2, w, h / 3 - gap,
                    facecolor=STACK_FILL, edgecolor=STACK_STROKE, lw=0.9)
        draw_text(ax, x + w / 2, yy + h / 6, label, size=11)
    return x + w, y + h / 2


def draw_latent_matrix(ax, x, y, cell=0.22):
    gap = 0.018
    for row in range(3):
        for col in range(4):
            rounded_box(ax, x + col * cell + gap / 2,
                        y + (2 - row) * cell + gap / 2,
                        cell - gap, cell - gap,
                        facecolor=CELL_FILL, edgecolor=CELL_STROKE, lw=0.8)
    draw_text(ax, x + 4.55 * cell, y + 1.5 * cell, "...", size=17)
    for row in range(3):
        rounded_box(ax, x + 5.35 * cell + gap / 2,
                    y + (2 - row) * cell + gap / 2,
                    cell - gap, cell - gap,
                    facecolor=CELL_FILL, edgecolor=CELL_STROKE, lw=0.8)
    draw_text(ax, x + 2.65 * cell, y - 0.18, r"$3\times N$", size=12)
    top_center_x = x + (5.35 + 1.0) * cell / 2
    top_y = y + 3 * cell
    return top_center_x, top_y


def draw_mlp(ax, x0, y0, h, layer_w=0.28, gap=0.48, n_layers=5):
    xs = []
    for i in range(n_layers):
        x = x0 + i * (layer_w + gap)
        xs.append(x)
        rounded_box(ax, x, y0, layer_w, h, facecolor=GREEN_FILL,
                    edgecolor=GREEN_STROKE, lw=1.1, rounding=0.07)
        if i:
            arrow(ax, (xs[i - 1] + layer_w, y0 + h / 2),
                  (x, y0 + h / 2), lw=1.1, mutation_scale=13)
    return xs


def draw_output_sphere(ax, sphere_img, cx, cy, r, red_line_start_x):
    ax.imshow(sphere_img, extent=(cx - r, cx + r, cy - r, cy + r), zorder=1)
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="0.15",
                        linewidth=1.0, zorder=2))
    curve_start = (cx + 0.06 * r, cy)
    ax.plot([red_line_start_x, curve_start[0]], [cy, cy],
            color="red", linestyle=(0, (8, 7)), linewidth=1.4, zorder=3)
    ax.plot([curve_start[0] - 0.05 * r, curve_start[0]], [cy, cy],
            color="red", linewidth=1.6, solid_capstyle="round", zorder=4)
    path = MplPath(
        [
            curve_start,
            (cx + 0.42 * r, cy),
            (cx + 0.36 * r, cy + 0.40 * r),
            (cx + 0.32 * r, cy + 0.59 * r),
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=17,
        linewidth=1.6,
        color="red",
        zorder=4,
        shrinkA=0,
        shrinkB=0,
    ))


def build_figure(args):
    envmap = read_exr(args.envmap)
    output_sphere = envmap_sphere_image(
        envmap,
        size=args.sphere_size,
        yaw=args.sphere_yaw,
        pitch=args.sphere_pitch,
        roll=args.sphere_roll,
    )

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=args.dpi)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_query_sphere(ax, 1.35, 4.18, 1.05)
    draw_latent_sphere(ax, 1.35, 1.52, 1.05)

    stack_right, stack_mid_y = draw_vector_stack(ax, 3.62, 3.55, 0.36, 0.80)
    matrix_top_cx, matrix_top_y = draw_latent_matrix(ax, 3.02, 1.36, cell=0.24)

    box_x, box_y, box_w, box_h = 4.82, 2.21, 1.94, 1.14
    box_center_y = box_y + box_h / 2
    upper_input_y = (box_center_y + box_y + box_h) / 2
    lower_input_y = (box_center_y + box_y) / 2
    arrow(ax, (stack_right + 0.04, stack_mid_y), (box_x, upper_input_y),
          color=ARROW_COLOR, lw=1.1, mutation_scale=14)
    arrow(ax, (matrix_top_cx, matrix_top_y + 0.04), (box_x, lower_input_y),
          color=ARROW_COLOR, lw=1.1, mutation_scale=14)
    rounded_box(ax, box_x, box_y, box_w, box_h, facecolor=BLUE_FILL,
                edgecolor=BLUE_STROKE, lw=1.4, rounding=0.10)
    draw_text(ax, box_x + box_w / 2, box_y + box_h / 2,
              "Invariant\nTransformation", size=12, color=BLUE_TEXT)

    mlp_y = 1.65
    mlp_h = 2.25
    mlp_center_y = mlp_y + mlp_h / 2
    mlp_x0 = 7.62
    mlp_layer_w = 0.27
    arrow(ax, (box_x + box_w, box_y + box_h / 2), (mlp_x0 - 0.03, mlp_center_y),
          lw=1.15, mutation_scale=15)
    draw_text(ax, (box_x + box_w + mlp_x0) / 2, mlp_center_y + 0.13,
              r"$\mathbf{d}'$", size=13)

    mlp_xs = draw_mlp(ax, mlp_x0, mlp_y, mlp_h, layer_w=mlp_layer_w, gap=0.33,
                      n_layers=5)
    draw_text(ax, 8.85, 4.35,
              "Rotation-Equivariant Conditional\nSpherical Neural Field",
              size=12)

    z_start_x = box_x + box_w / 2
    z_end_x = mlp_xs[2] + mlp_layer_w / 2
    z_y = 1.02
    ax.plot([z_start_x, z_start_x, z_end_x],
            [box_y, z_y, z_y], color="black", linewidth=1.0)
    arrow(ax, (z_end_x, z_y), (z_end_x, mlp_y),
          lw=1.0, mutation_scale=13)
    draw_text(ax, (z_start_x + z_end_x) / 2, z_y + 0.17,
              r"$\mathbf{Z}'$", size=13)

    last_layer_x = mlp_xs[-1] + mlp_layer_w
    c_x = 11.18
    arrow(ax, (last_layer_x, mlp_center_y), (c_x - 0.20, mlp_center_y),
          lw=1.15, mutation_scale=15)
    draw_text(ax, c_x, mlp_center_y, r"$\mathbf{C}$", size=13)
    sphere_cx = 12.85
    draw_output_sphere(ax, output_sphere, sphere_cx, mlp_center_y, 0.95,
                       red_line_start_x=c_x + 0.22)

    if args.envmap_label:
        label = args.envmap.name
        text = draw_text(ax, sphere_cx, 1.55, label, size=7)
        text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envmap", type=Path, default=None,
                        help="EXR environment map to project onto the output sphere")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "publication" / "figures" / "model_overview",
                        help="Output stem without extension")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--sphere_size", type=int, default=700)
    parser.add_argument("--sphere_yaw", type=float, default=210.0)
    parser.add_argument("--sphere_pitch", type=float, default=-8.0)
    parser.add_argument("--sphere_roll", type=float, default=0.0)
    parser.add_argument("--envmap_label", action="store_true",
                        help="Write the input EXR filename beneath the output sphere")
    parser.add_argument("--svg", action="store_true", help="Also write SVG")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.envmap is None:
        args.envmap = default_envmap()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args)
    fig.savefig(f"{args.output}.png", dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight", pad_inches=0.02)
    if args.svg:
        fig.savefig(f"{args.output}.svg", bbox_inches="tight", pad_inches=0.02)
    print(f"[saved] {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
