"""Programmatic RENI++ gravity-axis invariant model overview.

The diagram follows ``RENIField.vn_invariance`` for the SO(2) model. The
query direction and latent are split about the gravity axis into axial and
planar parts, which assemble into the two invariant decoder inputs (thesis
eqns for z_{d,g} and Z_{g,inv}); each invariant is drawn as one box holding
its defining equation. The Vector Neuron branch sits on the planar-latent
path, with a dashed callout explaining the shared Gram-Schmidt-orthonormalised
joint frame and its co-rotating per-channel readouts.
The output sphere is textured from an EXR environment map, which can be
overridden with ``--envmap``.

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
          mutation_scale=16, connectionstyle="arc3", zorder=None):
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
        zorder=zorder,
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
                zorder=2, linestyle="solid"):
    rounding = rounding if rounding is not None else 0.18 * min(w, h)
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder,
        linestyle=linestyle,
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


def draw_attention_decoder(ax, x, y, w, h):
    rounded_box(ax, x, y, w, h, facecolor="#F5F5F5",
                edgecolor="#8C8C8C", lw=1.2, rounding=0.14, zorder=2)
    draw_text(ax, x + w / 2, y + h - 0.27, "Attention Decoder", size=13.5)
    blk_y = y + 0.44
    blk_h = h - 0.89
    attn_x, attn_w = x + 0.24, 1.06
    rounded_box(ax, attn_x, blk_y, attn_w, blk_h,
                facecolor=BLUE_FILL, edgecolor=BLUE_STROKE,
                lw=1.1, rounding=0.10, zorder=3)
    draw_text(ax, attn_x + attn_w / 2, blk_y + blk_h / 2,
              "Multi-Head\nAttention", size=10, color=BLUE_TEXT)
    ffn_x, ffn_w = x + 1.62, 0.64
    rounded_box(ax, ffn_x, blk_y, ffn_w, blk_h,
                facecolor=GREEN_FILL, edgecolor=GREEN_STROKE,
                lw=1.1, rounding=0.10, zorder=3)
    draw_text(ax, ffn_x + ffn_w / 2, blk_y + blk_h / 2, "FFN", size=12)
    arrow(ax, (attn_x + attn_w, blk_y + blk_h / 2),
          (ffn_x - 0.04, blk_y + blk_h / 2),
          lw=1.0, mutation_scale=12, zorder=4)
    draw_text(ax, x + w / 2, y + 0.21,
              r"$\times\,N_{\mathrm{layer}}$", size=11, color="#555555")
    return {
        "left": x,
        "right": x + w,
        "output_y": y + h / 2,
    }


def arrow_through_chip(ax, start, end, chip_cx, chip_cy, chip_r, label,
                       label_size=16):
    """Split a straight arrow around a small rounded chip drawn on the path."""
    rounded_box(ax, chip_cx - chip_r, chip_cy - chip_r, 2 * chip_r, 2 * chip_r,
                facecolor="#FFF0C8", edgecolor="#D1A94B",
                lw=1.0, rounding=0.09, zorder=3)
    draw_text(ax, chip_cx, chip_cy, label, size=label_size, zorder=4)
    vec = np.array(end, dtype=float) - np.array(start, dtype=float)
    unit = vec / np.linalg.norm(vec)
    pad = 1.25 * chip_r
    ax.plot([start[0], chip_cx - pad * unit[0]],
            [start[1], chip_cy - pad * unit[1]],
            color=ARROW_COLOR, linewidth=1.05, zorder=1)
    arrow(ax, (chip_cx + pad * unit[0], chip_cy + pad * unit[1]), end,
          lw=1.05, mutation_scale=13)


def build_figure(args):
    envmap = read_exr(args.envmap)
    output_sphere = envmap_sphere_image(
        envmap,
        size=args.sphere_size,
        yaw=args.sphere_yaw,
        pitch=args.sphere_pitch,
        roll=args.sphere_roll,
    )

    fig, ax = plt.subplots(figsize=(15.6, 6.62), dpi=args.dpi)
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 6.62)
    ax.set_aspect("equal")
    ax.axis("off")

    # Layout: one lane per decoder input. The direction lane runs along
    # y=Y_DPAR/Y_DPERP into the query box; the latent lane runs along
    # y=Y_ZPERP/Y_ZPAR into the conditioning box. Z_perp fans out at one
    # junction dot to serve both boxes, the only cross-lane dependency.
    x_sphere, r_sphere = 1.30, 0.78
    x_fork = 2.44
    x_tok = 2.80
    y_dpar, y_dperp = 5.32, 4.60
    y_zperp, y_zpar = 2.12, 1.40
    y_query = (y_dpar + y_dperp) / 2
    y_latent = (y_zperp + y_zpar) / 2
    x_box, w_box = 5.15, 3.65
    qbox_y, qbox_h = 4.22, 1.28
    kbox_y, kbox_h = 1.06, 1.28
    x_junction = 4.15
    y_riser = qbox_y + 0.14
    dec_x, dec_w, dec_h = 10.35, 2.50, 2.40
    lane_q = qbox_y + qbox_h / 2
    lane_k = kbox_y + kbox_h / 2
    dec_y = (lane_q + lane_k) / 2 - dec_h / 2  # centred between the lanes

    # Column headings and colour key.
    for cx_head, text in ((2.15, "Inputs"),
                          (6.85, "Gravity-Axis Invariants"),
                          (12.80, "Conditional Decoder")):
        draw_text(ax, cx_head, 6.36, text, size=15, fontweight="bold")
    for x0, x1 in ((0.20, 4.10), (4.40, 9.85), (10.15, 15.40)):
        ax.plot([x0, x1], [6.12, 6.12], color="#B0B0B0", linewidth=0.8)
    # Legend chips, centred as a group on the canvas midline.
    rounded_box(ax, 5.86, 0.24, 0.20, 0.20, CELL_FILL, CELL_STROKE,
                lw=0.7, rounding=0.03)
    draw_text(ax, 6.17, 0.34, "rotates in the horizontal plane", size=10,
              ha="left", color="#555555")
    rounded_box(ax, 8.82, 0.24, 0.20, 0.20, STACK_FILL, STACK_STROKE,
                lw=0.7, rounding=0.03)
    draw_text(ax, 9.13, 0.34, "invariant", size=10, ha="left",
              color="#555555")

    # Inputs and their decomposition about the thesis-wide gravity axis g=e_y.
    draw_query_sphere(ax, x_sphere, y_query, r_sphere)
    draw_latent_sphere(ax, x_sphere, y_latent, r_sphere)
    draw_text(ax, x_sphere, (y_query + y_latent) / 2, r"$\mathbf{g}=\mathbf{e}_y$",
              size=12)

    # Direction split: axial coordinate and planar projection.
    ax.plot([x_sphere + r_sphere + 0.02, x_fork], [y_query, y_query],
            color=ARROW_COLOR, linewidth=1.0)
    ax.plot([x_fork, x_fork], [y_dperp, y_dpar], color=ARROW_COLOR,
            linewidth=1.0)
    d_parallel = draw_tensor(
        ax, x_tok, y_dpar - 0.15, 1, 1, cell=0.30,
        facecolor=STACK_FILL, edgecolor=STACK_STROKE,
    )
    draw_text(ax, d_parallel["cx"], d_parallel["top"] + 0.20,
              r"$d_{\parallel}=\mathbf{g}^{\mathsf{T}}\mathbf{d}$", size=11.5)
    d_perp = draw_tensor(
        ax, x_tok, y_dperp - 0.2775, 2, 1, cell=0.26, gap=0.035,
        facecolor=CELL_FILL, edgecolor=CELL_STROKE,
    )
    draw_text(ax, d_perp["cx"], d_perp["bottom"] - 0.20,
              r"$\mathbf{d}_{\perp}=\mathbf{B}_{\mathbf{g}}\mathbf{d}$",
              size=11.5)
    arrow(ax, (x_fork, y_dpar), (x_tok - 0.04, y_dpar),
          lw=0.9, mutation_scale=11)
    arrow(ax, (x_fork, y_dperp), (x_tok - 0.04, y_dperp),
          lw=0.9, mutation_scale=11)

    # Latent split. Z_perp sits nearest the query lane it also feeds.
    ax.plot([x_sphere + r_sphere + 0.02, x_fork], [y_latent, y_latent],
            color=ARROW_COLOR, linewidth=1.0)
    ax.plot([x_fork, x_fork], [y_zpar, y_zperp], color=ARROW_COLOR,
            linewidth=1.0)
    z_perp = draw_tensor(
        ax, x_tok, y_zperp - 0.2125, 2, 3, cell=0.20, gap=0.025,
        facecolor=CELL_FILL, edgecolor=CELL_STROKE, tail=True,
    )
    draw_text(ax, z_perp["cx"], z_perp["top"] + 0.20,
              r"$\mathbf{Z}_{\perp}=\mathbf{B}_{\mathbf{g}}\mathbf{Z}$",
              size=11.5)
    z_parallel = draw_tensor(
        ax, x_tok, y_zpar - 0.10, 1, 3, cell=0.20, gap=0.025,
        facecolor=STACK_FILL, edgecolor=STACK_STROKE, tail=True,
    )
    draw_text(ax, z_parallel["cx"], z_parallel["bottom"] - 0.20,
              r"$\mathbf{Z}_{\parallel}=\mathbf{g}^{\mathsf{T}}\mathbf{Z}$",
              size=11.5)
    arrow(ax, (x_fork, y_zperp), (x_tok - 0.04, y_zperp),
          lw=0.9, mutation_scale=11)
    arrow(ax, (x_fork, y_zpar), (x_tok - 0.04, y_zpar),
          lw=0.9, mutation_scale=11)

    # Invariant query: thesis eqn for z_{d,g}, one box.
    rounded_box(ax, x_box, qbox_y, w_box, qbox_h,
                facecolor=STACK_FILL, edgecolor=STACK_STROKE,
                lw=1.2, rounding=0.12)
    draw_text(ax, x_box + w_box / 2, qbox_y + 0.66 * qbox_h,
              r"$\mathbf{z}_{d,\mathbf{g}}=\left(d_{\parallel},\ "
              r"\mathbf{Z}_{\perp}^{\mathsf{T}}\mathbf{d}_{\perp},\ "
              r"\Vert\mathbf{d}_{\perp}\Vert_2\right)$", size=13)
    draw_text(ax, x_box + w_box / 2, qbox_y + 0.28 * qbox_h,
              r"$\in\mathbb{R}^{N_z+2}$", size=11, color="#555555")
    arrow(ax, (d_parallel["right"] + 0.05, y_dpar), (x_box - 0.03, y_dpar),
          lw=1.0, mutation_scale=12)
    arrow(ax, (d_perp["right"] + 0.05, y_dperp), (x_box - 0.03, y_dperp),
          lw=1.0, mutation_scale=12)

    # Invariant condition: thesis eqn for Z_{g,inv}.
    rounded_box(ax, x_box, kbox_y, w_box, kbox_h,
                facecolor=STACK_FILL, edgecolor=STACK_STROKE,
                lw=1.2, rounding=0.12)
    draw_text(ax, x_box + w_box / 2, kbox_y + 0.62 * kbox_h,
              r"$\mathbf{Z}_{\mathbf{g},\mathrm{inv}}=\left[\,"
              r"\mathbf{Z}_{\parallel}\,;\ \mathbf{Z}_{\perp}"
              r"\hat{\mathbf{Q}}^{\mathsf{T}}\right]$", size=13)
    draw_text(ax, x_box + w_box / 2, kbox_y + 0.30 * kbox_h,
              r"$\in\mathbb{R}^{3\times N_z}$", size=11, color="#555555")
    arrow(ax, (z_parallel["right"] + 0.05, y_zpar), (x_box - 0.03, y_zpar),
          lw=1.0, mutation_scale=12)

    # Z_perp fans out at a junction dot: through the channelwise Vector
    # Neuron branch into the conditioning box, and up one elbow into the
    # query box as the raw planar inner products.
    ax.plot([z_perp["right"] + 0.05, x_junction], [y_zperp, y_zperp],
            color=ARROW_COLOR, linewidth=1.0)
    ax.plot([x_junction], [y_zperp], marker="o", color=ARROW_COLOR,
            markersize=4.2, zorder=4)
    vn_w, vn_h = 0.55, 0.58
    vn_x = (x_junction + x_box) / 2 - vn_w / 2 - 0.09
    ax.plot([x_junction, vn_x + 0.02], [y_zperp, y_zperp],
            color=ARROW_COLOR, linewidth=1.0)
    draw_vn_glyph(ax, vn_x, y_zperp - vn_h / 2, w=vn_w, h=vn_h)
    arrow(ax, (vn_x + vn_w, y_zperp), (x_box - 0.03, y_zperp),
          lw=1.0, mutation_scale=12)
    ax.plot([x_junction, x_junction], [y_zperp, y_riser],
            color=ARROW_COLOR, linewidth=1.0)
    arrow(ax, (x_junction, y_riser), (x_box - 0.03, y_riser),
          lw=1.0, mutation_scale=12)

    # Annotation callout: ONE shared planar frame is predicted from all
    # latent channels by the Vector Neuron branch and Gram-Schmidt
    # orthonormalised; every channel is read out against it, so the inner
    # products cancel the shared rotation while retaining inter-channel
    # structure. The mini states below show three channels and the
    # orthonormal frame before and after R_2: everything co-rotates, the
    # readouts are unchanged.
    note_x, note_y, note_w, note_h = x_box, 2.52, w_box, 1.50
    rounded_box(ax, note_x, note_y, note_w, note_h,
                facecolor="white", edgecolor="#999999", lw=0.9,
                rounding=0.10, linestyle=(0, (4, 3)))
    note_cx = note_x + note_w / 2
    draw_text(ax, note_cx, note_y + 0.87 * note_h,
              r"$\hat{\mathbf{Q}}=\mathrm{GS}\!\left(\mathrm{VN}"
              r"(\mathbf{Z}_{\perp})\right),\quad"
              r"\tilde{z}_{ik}=\hat{\mathbf{q}}_k^{\mathsf{T}}"
              r"\mathbf{z}_{\perp,i}$", size=10)
    draw_text(ax, note_cx, note_y + 0.66 * note_h,
              r"$\hat{\mathbf{Q}}$ co-rotates:  "
              r"$(\mathbf{R}_2\hat{\mathbf{q}}_k)^{\mathsf{T}}"
              r"(\mathbf{R}_2\mathbf{z}_{\perp,i})=\tilde{z}_{ik}$",
              size=9.3, color="#444444")
    ax.plot([vn_x + vn_w - 0.06, note_x], [y_zperp + vn_h / 2, note_y + 0.26],
            color="#999999", linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)

    fan_y = note_y + 0.24 * note_h
    chan_angs = (40.0, 105.0, 215.0)
    chan_lens = (0.46, 0.34, 0.26)
    q_ang, q_len = 68.0, 0.36     # orthonormal frame: q1 at q_ang, q2 at +90
    for fx, rot_deg in ((note_cx - 1.00, 0.0), (note_cx + 1.00, 50.0)):
        fo = (fx, fan_y)
        ax.plot([fo[0]], [fo[1]], marker="o", color="#333333",
                markersize=2.2, zorder=5)
        # The shared orthonormal frame (unit, perpendicular), co-rotating.
        for qa in (q_ang + rot_deg, q_ang + 90.0 + rot_deg):
            qr = np.deg2rad(qa)
            arrow(ax, fo, (fo[0] + q_len * np.cos(qr),
                           fo[1] + q_len * np.sin(qr)),
                  color=BLUE_STROKE, lw=1.2, mutation_scale=8, zorder=5)
        # The latent channels, co-rotating with it.
        for ca, cl in zip(chan_angs, chan_lens):
            rad = np.deg2rad(ca + rot_deg)
            arrow(ax, fo, (fo[0] + cl * np.cos(rad), fo[1] + cl * np.sin(rad)),
                  color=VEC_COLOR, lw=1.2, mutation_scale=9, zorder=6)
    u0 = np.array([np.cos(np.deg2rad(chan_angs[0])),
                   np.sin(np.deg2rad(chan_angs[0]))])
    draw_text(ax, note_cx - 1.00 + 0.58 * u0[0] + 0.08,
              fan_y + 0.58 * u0[1],
              r"$\mathbf{z}_{\perp,i}$", size=7.5)
    q1 = np.array([np.cos(np.deg2rad(q_ang + 90.0)),
                   np.sin(np.deg2rad(q_ang + 90.0))])
    draw_text(ax, note_cx - 1.00 + 0.50 * q1[0] - 0.05,
              fan_y + 0.50 * q1[1] - 0.10,
              r"$\hat{\mathbf{q}}_k$", size=7.5)
    arrow(ax, (note_cx - 0.40, fan_y + 0.13), (note_cx + 0.40, fan_y + 0.13),
          color="#888888", lw=0.8, mutation_scale=8,
          connectionstyle="arc3,rad=-0.28", zorder=5)
    draw_text(ax, note_cx, fan_y - 0.16, r"$\mathbf{R}_2$", size=8.5,
              color="#666666")

    # Decoder, centred between the two lanes; entries mirror about it.
    decoder = draw_attention_decoder(ax, dec_x, dec_y, dec_w, dec_h)
    query_entry = (dec_x, dec_y + dec_h / 2 + 0.80)
    condition_entry = (dec_x, dec_y + dec_h / 2 - 0.80)

    # Only the query receives the NeRF positional encoding lambda.
    q_start = (x_box + w_box, qbox_y + qbox_h / 2)
    chip = (np.array(q_start) + np.array(query_entry)) / 2
    arrow_through_chip(ax, q_start, query_entry, chip[0], chip[1], 0.28,
                       r"$\lambda$")

    def label_beside(start, end, frac, dist, text):
        """Place text at ``frac`` along start->end, offset ``dist`` along
        the left-hand normal (negative for the right-hand side)."""
        s, e = np.array(start, dtype=float), np.array(end, dtype=float)
        unit = (e - s) / np.linalg.norm(e - s)
        normal = np.array([-unit[1], unit[0]])
        pos = s + frac * (e - s) + dist * normal
        draw_text(ax, pos[0], pos[1], text, size=12)

    label_beside(q_start, query_entry, 0.80, 0.27, r"$Q$")
    kv_start = (x_box + w_box, kbox_y + kbox_h / 2)
    arrow(ax, kv_start, condition_entry, lw=1.05, mutation_scale=13)
    label_beside(kv_start, condition_entry, 0.72, -0.28, r"$K,V$")

    c_x = 13.34
    arrow(ax, (decoder["right"], decoder["output_y"]),
          (c_x - 0.24, decoder["output_y"]), lw=1.1, mutation_scale=14)
    draw_text(ax, c_x, decoder["output_y"],
              r"$\hat{\mathbf{c}}(\mathbf{d})$", size=13)
    sphere_cx, sphere_r = 14.52, 0.85
    draw_output_sphere(ax, output_sphere, sphere_cx, decoder["output_y"],
                       sphere_r, red_line_start_x=c_x + 0.28)
    draw_text(ax, sphere_cx, decoder["output_y"] - sphere_r - 0.28,
              r"$f_{\boldsymbol{\Theta}}(\mathbf{d},\mathbf{Z})$", size=13)

    if args.envmap_label:
        label = args.envmap.name
        text = draw_text(ax, sphere_cx,
                         decoder["output_y"] - sphere_r - 0.54, label, size=7)
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
