"""Programmatic RENI++ gravity-axis invariant model overview.

The diagram follows ``RENIField.vn_invariance`` for the SO(2) model. It shows
the axial/planar decomposition, the direction-relative inner products, the
Vector Neuron frame used to make the planar latent invariant, and the two
inputs to the attention decoder. The output sphere is textured from an EXR
environment map, which can be overridden with ``--envmap``.

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
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)


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


def sphere_shadow_image(size: int = 240, max_alpha: float = 0.38) -> np.ndarray:
    """Soft radial shadow disc (transparent black, gaussian-ish falloff)."""
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rr = np.sqrt(xx**2 + yy**2)
    alpha = np.clip(1.0 - rr, 0.0, 1.0) ** 1.8 * max_alpha
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[..., 3] = alpha
    return rgba


def draw_sphere_shadow(ax, cx, cy, r, dx=0.12, dy=-0.16, scale=1.05,
                       zorder=-2):
    s = scale * r
    ax.imshow(sphere_shadow_image(),
              extent=(cx + dx * r - s, cx + dx * r + s,
                      cy + dy * r - s, cy + dy * r + s),
              zorder=zorder)


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


TEXT_SCALE = 1.15


def draw_text(ax, x, y, text, size=12, rotation=0, ha="center", va="center",
              **kwargs):
    defaults = dict(fontsize=size * TEXT_SCALE, rotation=rotation, ha=ha,
                    va=va, color="black")
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
    draw_sphere_shadow(ax, cx, cy, r)
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
    draw_sphere_shadow(ax, cx, cy, r)
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


def draw_output_sphere(ax, sphere_img, cx, cy, r, red_line_start_x):
    draw_sphere_shadow(ax, cx, cy, r, zorder=0)
    ax.imshow(sphere_img, extent=(cx - r, cx + r, cy - r, cy + r), zorder=1)
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


def draw_tensor(
    ax,
    x,
    y,
    rows,
    cols,
    *,
    cell=0.22,
    gap=0.035,
    facecolor=CELL_FILL,
    edgecolor=CELL_STROKE,
    tail=False,
    zorder=3,
):
    """Draw a small tensor glyph and return its bounding box.

    ``tail`` replaces the final columns with an ellipsis and one last column,
    which keeps the latent dimension legible without implying a small model.
    """
    shown_cols = cols
    if tail:
        shown_cols = cols + 2

    for row in range(rows):
        for col in range(cols):
            rounded_box(
                ax,
                x + col * (cell + gap),
                y + (rows - 1 - row) * (cell + gap),
                cell,
                cell,
                facecolor=facecolor,
                edgecolor=edgecolor,
                lw=0.8,
                rounding=0.035,
                zorder=zorder,
            )
        if tail:
            rounded_box(
                ax,
                x + (cols + 1) * (cell + gap),
                y + (rows - 1 - row) * (cell + gap),
                cell,
                cell,
                facecolor=facecolor,
                edgecolor=edgecolor,
                lw=0.8,
                rounding=0.035,
                zorder=zorder,
            )

    if tail:
        draw_text(
            ax,
            x + (cols + 0.45) * (cell + gap),
            y + (rows * cell + (rows - 1) * gap) / 2,
            r"$\cdots$",
            size=15,
            zorder=zorder + 1,
        )

    width = shown_cols * cell + (shown_cols - 1) * gap
    height = rows * cell + (rows - 1) * gap
    return {
        "left": x,
        "right": x + width,
        "bottom": y,
        "top": y + height,
        "cx": x + width / 2,
        "cy": y + height / 2,
    }


def draw_vn_glyph(ax, x, y, w=0.72, h=0.72):
    """Horizontal Vector Neuron branch, wide at input and narrow at output."""
    points = [
        (x, y),
        (x, y + h),
        (x + w, y + 0.72 * h),
        (x + w, y + 0.28 * h),
    ]
    ax.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor=BLUE_STROKE,
            edgecolor=BLUE_TEXT,
            linewidth=1.0,
            zorder=3,
        )
    )
    draw_text(ax, x + 0.39 * w, y + h / 2, "VN", size=12,
              color="white", zorder=4)
    return x + w, y + h / 2


def draw_directional_invariant(ax, d_parallel, d_perp, z_perp):
    """Draw the SO(2) directional invariant used as the decoder query."""
    # The inner-product operands repeat the relevant decomposed inputs so the
    # matrix operation remains readable when the figure is scaled in LaTeX.
    zt = draw_tensor(ax, 4.18, 4.76, 3, 2, cell=0.19, gap=0.025,
                     facecolor=CELL_FILL, edgecolor=CELL_STROKE)
    draw_text(ax, zt["cx"], zt["top"] + 0.16,
              r"$\mathbf{Z}_{\perp}^{\mathsf{T}}$", size=13)
    draw_text(ax, zt["cx"], zt["bottom"] - 0.14,
              r"$N_z\!\times\!2$", size=10, color="#555555")

    draw_text(ax, 4.85, zt["cy"], r"$\times$", size=18)
    dp = draw_tensor(ax, 5.16, 4.89, 2, 1, cell=0.22, gap=0.035,
                     facecolor=CELL_FILL, edgecolor=CELL_STROKE)
    draw_text(ax, dp["cx"], dp["top"] + 0.16,
              r"$\mathbf{d}_{\perp}$", size=13)
    draw_text(ax, 5.62, zt["cy"], r"$=$", size=17)
    inner = draw_tensor(ax, 5.91, 4.76, 3, 1, cell=0.19, gap=0.025,
                        facecolor=STACK_FILL, edgecolor=STACK_STROKE)
    draw_text(ax, inner["cx"], inner["top"] + 0.16,
              r"$\mathbf{Z}_{\perp}^{\mathsf{T}}\mathbf{d}_{\perp}$",
              size=13)
    draw_text(ax, inner["cx"], inner["bottom"] - 0.14,
              r"$N_z\!\times\!1$", size=10, color="#555555")

    norm_x, norm_y, norm_w, norm_h = 6.45, 4.40, 1.18, 0.52
    rounded_box(ax, norm_x, norm_y, norm_w, norm_h,
                facecolor=STACK_FILL, edgecolor=STACK_STROKE,
                lw=1.0, rounding=0.08)
    draw_text(ax, norm_x + norm_w / 2, norm_y + norm_h / 2,
              r"$\Vert\mathbf{d}_{\perp}\Vert_2$", size=13)

    # Assemble (d_parallel, Z_perp^T d_perp, ||d_perp||) as one invariant
    # directional input. Different cell heights distinguish the N_z block.
    out_x = 8.04
    top_cell = draw_tensor(ax, out_x, 5.72, 1, 1, cell=0.28,
                           facecolor=STACK_FILL, edgecolor=STACK_STROKE)
    middle = draw_tensor(ax, out_x, 4.91, 3, 1, cell=0.20, gap=0.025,
                         facecolor=STACK_FILL, edgecolor=STACK_STROKE)
    bottom_cell = draw_tensor(ax, out_x, 4.43, 1, 1, cell=0.28,
                              facecolor=STACK_FILL, edgecolor=STACK_STROKE)
    draw_text(ax, out_x + 0.50, top_cell["cy"], r"$d_{\parallel}$",
              size=12, ha="left")
    draw_text(ax, out_x + 0.50, middle["cy"],
              r"$\mathbf{Z}_{\perp}^{\mathsf{T}}\mathbf{d}_{\perp}$",
              size=12, ha="left")
    draw_text(ax, out_x + 0.50, bottom_cell["cy"],
              r"$\Vert\mathbf{d}_{\perp}\Vert_2$", size=12, ha="left")
    draw_text(ax, 8.65, 4.06,
              r"$\mathbf{z}_{d,\mathbf{g}}\in\mathbb{R}^{N_z+2}$",
              size=13)

    # Routed dependencies. The lower latent branch supplies Z_perp.
    arrow(ax, (d_parallel["right"] + 0.05, d_parallel["cy"]),
          (top_cell["left"] - 0.05, top_cell["cy"]),
          lw=1.0, mutation_scale=12,
          connectionstyle="arc3,rad=-0.04")
    arrow(ax, (d_perp["right"] + 0.05, d_perp["cy"]),
          (dp["left"] - 0.05, dp["cy"]), lw=1.0, mutation_scale=12)
    arrow(ax, (z_perp["right"] + 0.04, z_perp["cy"]),
          (zt["left"] - 0.04, zt["cy"]), lw=0.95, mutation_scale=11,
          connectionstyle="angle,angleA=0,angleB=90,rad=5")
    arrow(ax, (inner["right"] + 0.05, inner["cy"]),
          (middle["left"] - 0.05, middle["cy"]),
          lw=1.0, mutation_scale=12)
    arrow(ax, (d_perp["right"] + 0.05, d_perp["cy"] - 0.06),
          (norm_x - 0.05, norm_y + norm_h / 2),
          lw=0.95, mutation_scale=11,
          connectionstyle="arc3,rad=0.08")
    arrow(ax, (norm_x + norm_w + 0.05, norm_y + norm_h / 2),
          (bottom_cell["left"] - 0.05, bottom_cell["cy"]),
          lw=1.0, mutation_scale=12)
    return {
        "left": out_x,
        "right": 9.55,
        "cy": 5.18,
    }


def draw_latent_invariant(ax, z_parallel, z_perp):
    """Draw the VN planar frame and axial bypass used for conditioning."""
    # Z_perp takes a copy path and an equivariant VN path that predicts Q_2.
    vn_right, vn_cy = draw_vn_glyph(ax, 4.34, 1.98, w=0.72, h=0.72)
    arrow(ax, (z_perp["right"] + 0.05, z_perp["cy"] + 0.10),
          (4.34 - 0.05, vn_cy), lw=1.0, mutation_scale=12,
          connectionstyle="arc3,rad=-0.12")

    basis = draw_tensor(ax, 5.38, 2.03, 2, 2, cell=0.22, gap=0.03,
                        facecolor=CELL_FILL, edgecolor=CELL_STROKE)
    arrow(ax, (vn_right + 0.05, vn_cy),
          (basis["left"] - 0.05, basis["cy"]),
          lw=1.0, mutation_scale=12)
    draw_text(ax, basis["cx"], basis["top"] + 0.17,
              r"$\mathbf{Q}_2(\mathbf{Z}_{\perp})^{\mathsf{T}}$", size=12)
    draw_text(ax, 6.16, basis["cy"], r"$\times$", size=17)

    copied = draw_tensor(ax, 6.48, 1.86, 2, 3, cell=0.20, gap=0.025,
                         facecolor=CELL_FILL, edgecolor=CELL_STROKE, tail=True)
    draw_text(ax, copied["cx"], copied["top"] + 0.17,
              r"$\mathbf{Z}_{\perp}$", size=12)
    arrow(ax, (z_perp["right"] + 0.05, z_perp["cy"] - 0.12),
          (copied["left"] - 0.05, copied["cy"]),
          lw=0.95, mutation_scale=11,
          connectionstyle="angle,angleA=0,angleB=90,rad=5")
    draw_text(ax, 5.65, 1.29, "copy", size=10, color="#555555")

    draw_text(ax, 7.82, copied["cy"], r"$=$", size=17)
    planar_inv = draw_tensor(
        ax, 8.16, 1.86, 2, 3, cell=0.20, gap=0.025,
        facecolor=STACK_FILL, edgecolor=STACK_STROKE, tail=True,
    )
    draw_text(ax, planar_inv["cx"], planar_inv["top"] + 0.17,
              r"$\mathrm{VNInv}_2(\mathbf{Z}_{\perp})$", size=12)

    # Reattach the axial component, which is already SO(2)-invariant.
    latent_out = draw_tensor(
        ax, 9.72, 1.72, 3, 3, cell=0.20, gap=0.025,
        facecolor=STACK_FILL, edgecolor=STACK_STROKE, tail=True,
    )
    draw_text(ax, latent_out["cx"], latent_out["bottom"] - 0.22,
              r"$\mathbf{Z}_{\mathbf{g},\mathrm{inv}}\in"
              r"\mathbb{R}^{3\times N_z}$", size=13)
    arrow(ax, (planar_inv["right"] + 0.05, planar_inv["cy"]),
          (latent_out["left"] - 0.05, latent_out["bottom"] + 0.31),
          lw=1.0, mutation_scale=12)
    bypass_x = z_parallel["right"] + 0.27
    bypass_y = 3.12
    ax.plot(
        [z_parallel["right"] + 0.05, bypass_x, bypass_x, latent_out["left"] - 0.38],
        [z_parallel["cy"], z_parallel["cy"], bypass_y, bypass_y],
        color=ARROW_COLOR,
        linewidth=0.95,
        zorder=1,
    )
    arrow(ax, (latent_out["left"] - 0.38, bypass_y),
          (latent_out["left"] - 0.05, latent_out["top"] - 0.10),
          lw=0.95, mutation_scale=11)
    return latent_out


def draw_attention_decoder(ax, x, y, w, h):
    rounded_box(ax, x, y, w, h, facecolor="#F5F5F5",
                edgecolor="#8C8C8C", lw=1.2, rounding=0.14, zorder=2)
    draw_text(ax, x + w / 2, y + h - 0.27, "Attention Decoder", size=14)
    attn_x, attn_y, attn_w, attn_h = x + 0.27, y + 0.48, 1.08, 1.12
    rounded_box(ax, attn_x, attn_y, attn_w, attn_h,
                facecolor=BLUE_FILL, edgecolor=BLUE_STROKE,
                lw=1.1, rounding=0.10, zorder=3)
    draw_text(ax, attn_x + attn_w / 2, attn_y + attn_h / 2,
              "Multi-Head\nAttention", size=10, color=BLUE_TEXT)
    ffn_x, ffn_y, ffn_w, ffn_h = x + 1.58, y + 0.48, 0.72, 1.12
    rounded_box(ax, ffn_x, ffn_y, ffn_w, ffn_h,
                facecolor=GREEN_FILL, edgecolor=GREEN_STROKE,
                lw=1.1, rounding=0.10, zorder=3)
    draw_text(ax, ffn_x + ffn_w / 2, ffn_y + ffn_h / 2,
              "FFN", size=12)
    arrow(ax, (attn_x + attn_w, attn_y + attn_h / 2),
          (ffn_x - 0.04, ffn_y + ffn_h / 2),
          lw=1.0, mutation_scale=12)
    draw_text(ax, x + w / 2, y + 0.22,
              r"$\times\,N_{\mathrm{layer}}$", size=11, color="#555555")
    return {
        "left": x,
        "right": x + w,
        "query_y": attn_y + 0.77 * attn_h,
        "condition_y": attn_y + 0.27 * attn_h,
        "output_y": ffn_y + ffn_h / 2,
    }


def build_figure(args):
    envmap = read_exr(args.envmap)
    output_sphere = envmap_sphere_image(
        envmap,
        size=args.sphere_size,
        yaw=args.sphere_yaw,
        pitch=args.sphere_pitch,
        roll=args.sphere_roll,
    )

    fig, ax = plt.subplots(figsize=(16.8, 7.2), dpi=args.dpi)
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Column headings and colour key.
    draw_text(ax, 1.35, 6.92, "Equivariant Inputs", size=15,
              fontweight="bold")
    draw_text(ax, 6.55, 6.92, "Gravity-Axis Invariants", size=15,
              fontweight="bold")
    draw_text(ax, 12.68, 6.92, "Conditional Decoder", size=15,
              fontweight="bold")
    ax.plot([0.15, 3.25], [6.69, 6.69], color="#B0B0B0", linewidth=0.8)
    ax.plot([3.55, 10.75], [6.69, 6.69], color="#B0B0B0", linewidth=0.8)
    ax.plot([11.05, 16.65], [6.69, 6.69], color="#B0B0B0", linewidth=0.8)
    rounded_box(ax, 6.80, 6.37, 0.20, 0.20, CELL_FILL, CELL_STROKE,
                lw=0.7, rounding=0.03)
    draw_text(ax, 7.11, 6.47, "rotates in the horizontal plane", size=10,
              ha="left", color="#555555")
    rounded_box(ax, 10.18, 6.37, 0.20, 0.20, STACK_FILL, STACK_STROKE,
                lw=0.7, rounding=0.03)
    draw_text(ax, 10.49, 6.47, "invariant", size=10, ha="left",
              color="#555555")

    # Inputs and their decomposition about the thesis-wide gravity axis g=e_y.
    draw_query_sphere(ax, 1.18, 5.33, 0.76)
    draw_latent_sphere(ax, 1.18, 1.88, 0.76)
    draw_text(ax, 1.18, 4.25, r"$\mathbf{g}=\mathbf{e}_y$", size=12)

    split_x = 2.18
    arrow(ax, (1.95, 5.33), (split_x, 5.33), lw=1.0, mutation_scale=12)
    ax.plot([split_x, split_x], [5.02, 5.90], color=ARROW_COLOR, linewidth=1.0)
    d_parallel = draw_tensor(
        ax, 2.53, 5.76, 1, 1, cell=0.30,
        facecolor=STACK_FILL, edgecolor=STACK_STROKE,
    )
    draw_text(ax, d_parallel["cx"], d_parallel["top"] + 0.18,
              r"$d_{\parallel}=\mathbf{g}^{\mathsf{T}}\mathbf{d}$", size=12)
    d_perp = draw_tensor(
        ax, 2.53, 4.82, 2, 1, cell=0.27, gap=0.035,
        facecolor=CELL_FILL, edgecolor=CELL_STROKE,
    )
    draw_text(ax, d_perp["cx"], d_perp["bottom"] - 0.18,
              r"$\mathbf{d}_{\perp}=\mathbf{B}_{\mathbf{g}}\mathbf{d}$", size=12)
    arrow(ax, (split_x, 5.86), (d_parallel["left"] - 0.04, d_parallel["cy"]),
          lw=0.9, mutation_scale=11)
    arrow(ax, (split_x, 5.06), (d_perp["left"] - 0.04, d_perp["cy"]),
          lw=0.9, mutation_scale=11)

    arrow(ax, (1.95, 1.88), (split_x, 1.88), lw=1.0, mutation_scale=12)
    ax.plot([split_x, split_x], [1.28, 2.58], color=ARROW_COLOR, linewidth=1.0)
    z_parallel = draw_tensor(
        ax, 2.53, 2.43, 1, 3, cell=0.20, gap=0.025,
        facecolor=STACK_FILL, edgecolor=STACK_STROKE, tail=True,
    )
    draw_text(ax, z_parallel["cx"], z_parallel["top"] + 0.18,
              r"$\mathbf{Z}_{\parallel}=\mathbf{g}^{\mathsf{T}}\mathbf{Z}$",
              size=12)
    z_perp = draw_tensor(
        ax, 2.53, 1.12, 2, 3, cell=0.20, gap=0.025,
        facecolor=CELL_FILL, edgecolor=CELL_STROKE, tail=True,
    )
    draw_text(ax, z_perp["cx"], z_perp["bottom"] - 0.20,
              r"$\mathbf{Z}_{\perp}=\mathbf{B}_{\mathbf{g}}\mathbf{Z}$",
              size=12)
    arrow(ax, (split_x, 2.53), (z_parallel["left"] - 0.04, z_parallel["cy"]),
          lw=0.9, mutation_scale=11)
    arrow(ax, (split_x, 1.31), (z_perp["left"] - 0.04, z_perp["cy"]),
          lw=0.9, mutation_scale=11)

    directional = draw_directional_invariant(ax, d_parallel, d_perp, z_perp)
    latent = draw_latent_invariant(ax, z_parallel, z_perp)

    # Only the directional invariant receives the NeRF positional encoding.
    lambda_x, lambda_y, lambda_w, lambda_h = 10.08, 4.91, 0.58, 0.58
    rounded_box(ax, lambda_x, lambda_y, lambda_w, lambda_h,
                facecolor="#FFF0C8", edgecolor="#D1A94B",
                lw=1.0, rounding=0.09)
    draw_text(ax, lambda_x + lambda_w / 2, lambda_y + lambda_h / 2,
              r"$\lambda$", size=16)
    arrow(ax, (directional["right"] + 0.03, directional["cy"]),
          (lambda_x - 0.04, lambda_y + lambda_h / 2),
          lw=1.0, mutation_scale=12)

    decoder = draw_attention_decoder(ax, 11.28, 2.72, 2.42, 2.22)
    arrow(ax, (lambda_x + lambda_w + 0.04, lambda_y + lambda_h / 2),
          (decoder["left"] - 0.04, decoder["query_y"]),
          lw=1.05, mutation_scale=13)
    draw_text(ax, 10.94, decoder["query_y"] + 0.13, r"$Q$", size=12)
    arrow(ax, (latent["right"] + 0.04, latent["cy"]),
          (decoder["left"] - 0.04, decoder["condition_y"]),
          lw=1.05, mutation_scale=13,
          connectionstyle="arc3,rad=0.08")
    draw_text(ax, 10.94, decoder["condition_y"] - 0.15,
              r"$K,V$", size=12)

    c_x = 14.08
    arrow(ax, (decoder["right"] + 0.04, decoder["output_y"]),
          (c_x - 0.22, decoder["output_y"]), lw=1.1, mutation_scale=14)
    draw_text(ax, c_x, decoder["output_y"],
              r"$\hat{\mathbf{c}}(\mathbf{d})$", size=13)
    sphere_cx = 15.64
    draw_output_sphere(ax, output_sphere, sphere_cx, decoder["output_y"], 0.88,
                       red_line_start_x=c_x + 0.30)
    draw_text(ax, sphere_cx, 2.24,
              r"$f_{\boldsymbol{\Theta}}(\mathbf{d},\mathbf{Z})$", size=13)

    if args.envmap_label:
        label = args.envmap.name
        text = draw_text(ax, sphere_cx, 1.98, label, size=7)
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
