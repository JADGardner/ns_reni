"""Scratch explainer for the Vector Neuron invariance mechanism (SO(2)).

Temporary companion to fig_model_overview.py, used to refine the VNInv_2
callout and build intuition. Not a thesis asset; iterate freely and delete
when done.

The VN operations below are numpy mirrors of the shipped implementation in
``reni/field_components/vn_layers.py`` (VNLinear, VNReLU, VNInvariant), so
every number printed in the figure is computed by the real rules, and the
invariance claims are asserted at run time.

Run from the ns_reni repo root:

    PYTHONPATH=. python scripts/figures/scratch_vn_invariance.py
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
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]

VEC_COLOR = "#7C9AD6"      # latent vectors (matches fig_model_overview)
FRAME_COLOR = "#3A5FAD"    # VN-predicted frame vectors
DIR_COLOR = "#3D8B2F"      # direction d_perp
GHOST_ALPHA = 0.30
AXIS_COLOR = "#B0B0B0"


# --------------------------------------------------------------------------
# Numpy mirrors of reni/field_components/vn_layers.py. Vector features are
# arrays [channels, 2]: each row is one 2D vector living in the horizontal
# plane.
# --------------------------------------------------------------------------

def vn_linear(weight: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """VNLinear: mix vectors with scalar weights (no bias)."""
    return weight @ vecs


def vn_relu(w: np.ndarray, u: np.ndarray, vecs: np.ndarray,
            eps: float = 1e-6) -> np.ndarray:
    """VNReLU, mirroring the shipped forward() exactly."""
    q = w @ vecs
    k = u @ vecs
    qk = (q * k).sum(-1, keepdims=True)
    k_norm = np.sqrt((k ** 2).sum(-1, keepdims=True)).clip(min=eps)
    q_projected_on_k = q - (q * (k / k_norm)).sum(-1, keepdims=True) * k
    return np.where(qk >= 0.0, q, q_projected_on_k)


def vn_frame(vecs: np.ndarray, w_lift: np.ndarray, w_relu: np.ndarray,
             u_relu: np.ndarray) -> np.ndarray:
    """VNInvariant.mlp: VNLinear(c_in->2) then VNReLU(2). Rows are the two
    predicted frame vectors q_1, q_2."""
    return vn_relu(w_relu, u_relu, vn_linear(w_lift, vecs))


def readout(vecs: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """VNInvariant einsum: entries <z_i, q_j>."""
    return vecs @ frame.T


def rot(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def dir_vec(deg: float, length: float = 1.0) -> np.ndarray:
    a = np.deg2rad(deg)
    return length * np.array([np.cos(a), np.sin(a)])


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------

def arrow(ax, start, end, color, lw=1.6, mutation=13, alpha=1.0, zorder=4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=to_rgba(color, alpha), shrinkA=0, shrinkB=0,
        zorder=zorder,
    ))


def vec_arrow(ax, origin, vec, color, scale=1.0, alpha=1.0, lw=1.6,
              label=None, label_pad=0.16, label_size=11, label_off=None,
              zorder=4):
    tip = (origin[0] + scale * vec[0], origin[1] + scale * vec[1])
    arrow(ax, origin, tip, color, lw=lw, alpha=alpha, zorder=zorder)
    if label:
        if label_off is None:
            n = np.linalg.norm(vec)
            label_off = (vec / n) * label_pad if n > 0 else (0.0, label_pad)
        ax.text(tip[0] + label_off[0], tip[1] + label_off[1], label,
                ha="center", va="center", fontsize=label_size,
                color=to_rgba("black", max(alpha, 0.55)))


def rotation_arc(ax, origin, radius, a_from, a_to, color="#888888", lw=1.0,
                 label=None):
    start = (origin[0] + radius * np.cos(np.deg2rad(a_from)),
             origin[1] + radius * np.sin(np.deg2rad(a_from)))
    end = (origin[0] + radius * np.cos(np.deg2rad(a_to)),
           origin[1] + radius * np.sin(np.deg2rad(a_to)))
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
        color=color, connectionstyle="arc3,rad=0.32", shrinkA=0, shrinkB=0,
        zorder=3,
    ))
    if label:
        mid = np.deg2rad((a_from + a_to) / 2)
        ax.text(origin[0] + (radius + 0.22) * np.cos(mid),
                origin[1] + (radius + 0.22) * np.sin(mid), label,
                ha="center", va="center", fontsize=10, color="#666666")


def plane_axes(ax, origin, r=1.15):
    """Unlabelled fixed world axes for the horizontal plane."""
    for dx, dy in ((r, 0), (0, r)):
        arrow(ax, origin, (origin[0] + dx, origin[1] + dy), AXIS_COLOR,
              lw=0.9, mutation=9, zorder=2)


def dotted_projection(ax, origin, vec, axis_unit, color="#999999"):
    """Dotted drop from the tip of vec onto the line through axis_unit."""
    tip = np.array(origin) + vec
    foot = np.array(origin) + np.dot(vec, axis_unit) * axis_unit
    ax.plot([tip[0], foot[0]], [tip[1], foot[1]], linestyle=(0, (2, 3)),
            color=color, linewidth=0.9, zorder=2)


def fmt(x) -> str:
    return np.array2string(np.asarray(x), separator=", ",
                           formatter={"float_kind": lambda v: f"{v:.2f}"})


def panel_title(ax, cx, y, text):
    ax.text(cx, y, text, ha="center", va="center", fontsize=12.5,
            fontweight="bold")


def panel_note(ax, cx, y, lines, size=9.5, dy=0.30, color="#444444"):
    for i, line in enumerate(lines):
        ax.text(cx, y - i * dy, line, ha="center", va="center",
                fontsize=size, color=color)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def build_figure():
    rng = np.random.default_rng(12)
    R_DEG = 50.0
    R = rot(R_DEG)

    fig, ax = plt.subplots(figsize=(21.0, 5.6), dpi=200)
    ax.set_xlim(0, 21.0)
    ax.set_ylim(0, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")

    centers = [2.1, 6.3, 10.5, 14.5, 18.8]
    cy = 3.0
    title_y = 5.25
    note_y = 1.15

    # ---- Panel 1: world coordinates are not invariant -------------------
    cx = centers[0]
    panel_title(ax, cx, title_y, "1. the problem")
    origin = (cx - 0.15, cy - 0.55)
    plane_axes(ax, origin)
    z = dir_vec(22.0, 1.05)
    zr = R @ z
    vec_arrow(ax, origin, z, VEC_COLOR, label=r"$\mathbf{z}_{\perp}$",
              label_off=(0.16, 0.10))
    vec_arrow(ax, origin, zr, VEC_COLOR, alpha=GHOST_ALPHA,
              label=r"$\mathbf{R}\mathbf{z}_{\perp}$",
              label_off=(-0.26, 0.10))
    dotted_projection(ax, origin, z, np.array([1.0, 0.0]))
    dotted_projection(ax, origin, z, np.array([0.0, 1.0]))
    rotation_arc(ax, origin, 1.18, 22.0, 22.0 + R_DEG, label=r"$\mathbf{R}$")
    panel_note(ax, cx, note_y, [
        f"coords {fmt(z)}  →  {fmt(zr)}",
        "fixed-axis coordinates change",
        "under a gravity-axis rotation",
    ])

    # ---- Panel 2: VN layers predict a frame ------------------------------
    cx = centers[1]
    panel_title(ax, cx, title_y, "2. predict a frame from the vectors")
    origin = (cx, cy - 0.55)
    z1 = dir_vec(20.0, 0.95)
    z2 = dir_vec(115.0, 0.75)
    Z = np.stack([z1, z2])
    w_lift = rng.normal(size=(2, 2))
    w_relu = rng.normal(size=(2, 2))
    u_relu = rng.normal(size=(2, 2))
    Q = vn_frame(Z, w_lift, w_relu, u_relu)
    vec_arrow(ax, origin, z1, VEC_COLOR, label=r"$\mathbf{z}_1$")
    vec_arrow(ax, origin, z2, VEC_COLOR, label=r"$\mathbf{z}_2$")
    vec_arrow(ax, origin, Q[0], FRAME_COLOR, label=r"$\mathbf{q}_1$")
    vec_arrow(ax, origin, Q[1], FRAME_COLOR, label=r"$\mathbf{q}_2$")
    panel_note(ax, cx, note_y, [
        r"$\mathbf{z}_1,\mathbf{z}_2$: two whole vectors (channels),"
        " not components;",
        "VNLinear mixes them, VNReLU gates them, giving",
        r"frame $\mathbf{Q}(\mathbf{Z})=(\mathbf{q}_1,\mathbf{q}_2)$;"
        r" the output is $\langle\mathbf{z}_i,\mathbf{q}_j\rangle$",
    ])

    # ---- Panel 3: co-rotation makes the read-out invariant ---------------
    cx = centers[2]
    panel_title(ax, cx, title_y, "3. rotate: the read-out is unchanged")
    Zr = Z @ R.T
    Qr = vn_frame(Zr, w_lift, w_relu, u_relu)
    assert np.allclose(Qr, Q @ R.T), "frame is not equivariant!"
    M = readout(Z, Q)
    Mr = readout(Zr, Qr)
    assert np.allclose(M, Mr), "read-out is not invariant!"
    s3 = 0.70
    for o_x, vz, vq in ((cx - 1.15, Z, Q), (cx + 1.15, Zr, Qr)):
        o = (o_x, cy - 0.45)
        for v in vz:
            vec_arrow(ax, o, s3 * v, VEC_COLOR, lw=1.4)
        for v in vq:
            vec_arrow(ax, o, s3 * v, FRAME_COLOR, lw=1.4)
        ax.plot([o[0]], [o[1]], marker="o", color="#333333", markersize=2.0,
                zorder=5)
    ax.add_patch(FancyArrowPatch(
        (cx - 0.42, cy + 0.62), (cx + 0.42, cy + 0.62), arrowstyle="-|>",
        mutation_scale=11, linewidth=1.0, color="#888888",
        connectionstyle="arc3,rad=-0.30", shrinkA=0, shrinkB=0, zorder=3))
    ax.text(cx, cy + 1.02, r"$\mathbf{R}$", fontsize=10, color="#666666",
            ha="center", va="center")
    panel_note(ax, cx, note_y, [
        r"frame recomputed from the rotated vectors;"
        r" $\langle\mathbf{z}_i,\mathbf{q}_j\rangle$:",
        f"left {fmt(M.ravel())}",
        f"right {fmt(Mr.ravel())}  (identical, asserted)",
    ])

    # ---- Panel 4: the RENI++ single-vector channel -----------------------
    cx = centers[3]
    panel_title(ax, cx, title_y, "4. RENI++: one vector per channel")
    origin = (cx - 0.35, cy - 0.55)
    z = dir_vec(38.0, 1.0)
    w_proj = rng.normal(size=(1, 1))          # vn_proj_in: VNLinear(1->1)
    w_lift1 = rng.normal(size=(2, 1))         # VNInvariant: VNLinear(1->2)
    z_hat = vn_linear(w_proj, z[None, :])
    Q1 = vn_frame(z_hat, w_lift1, w_relu, u_relu)
    cross = float(Q1[0, 0] * Q1[1, 1] - Q1[0, 1] * Q1[1, 0])
    assert abs(cross) < 1e-9, "single-channel frame should be parallel"
    inv_a = readout(z_hat, Q1)
    # Same norm, different angle: the read-out cannot tell them apart.
    z_alt = dir_vec(170.0, 1.0)
    z_hat_alt = vn_linear(w_proj, z_alt[None, :])
    inv_b = readout(z_hat_alt, vn_frame(z_hat_alt, w_lift1, w_relu, u_relu))
    assert np.allclose(inv_a, inv_b), "read-out should depend on norm only"
    d_perp = dir_vec(-15.0, 0.85)
    vec_arrow(ax, origin, z, VEC_COLOR, label=r"$\mathbf{z}_{\perp}$")
    # True q lengths are tiny here; draw unit directions (the parallelism is
    # the point) with a hair of perpendicular offset so both stay visible.
    q_dirs = Q1 / np.linalg.norm(Q1, axis=1, keepdims=True)
    perp = np.array([-q_dirs[0, 1], q_dirs[0, 0]])
    for qd, length, side, lab in ((q_dirs[0], 0.85, 1.0, r"$\mathbf{q}_1$"),
                                  (q_dirs[1], 0.62, -1.0, r"$\mathbf{q}_2$")):
        o = (origin[0] + side * 0.025 * perp[0],
             origin[1] + side * 0.025 * perp[1])
        vec_arrow(ax, o, length * qd, FRAME_COLOR, lw=1.3, label=lab,
                  label_off=(side * 0.20 * perp[0] + 0.02,
                             side * 0.20 * perp[1] - 0.04))
    vec_arrow(ax, origin, d_perp, DIR_COLOR, label=r"$\mathbf{d}_{\perp}$")
    ax.add_patch(FancyArrowPatch(
        (origin[0] + 0.55 * np.cos(np.deg2rad(-15)),
         origin[1] + 0.55 * np.sin(np.deg2rad(-15))),
        (origin[0] + 0.55 * np.cos(np.deg2rad(38)),
         origin[1] + 0.55 * np.sin(np.deg2rad(38))),
        arrowstyle="-", linewidth=0.9, color="#999999",
        connectionstyle="arc3,rad=0.18", zorder=3))
    ax.text(origin[0] + 0.78 * np.cos(np.deg2rad(11)),
            origin[1] + 0.78 * np.sin(np.deg2rad(11)), r"$\theta$",
            fontsize=10, color="#666666", ha="center", va="center")
    panel_note(ax, cx, note_y, [
        "every equivariant prediction from one vector",
        r"is parallel to it, so $\mathrm{VNInv}_2$ is a function of "
        r"$\Vert\mathbf{z}_{\perp}\Vert$ only;",
        r"the angle $\theta$ reaches the decoder via "
        r"$\mathbf{Z}_{\perp}^{\mathsf{T}}\mathbf{d}_{\perp}$ in the query",
    ], size=9)

    # ---- Panel 5: per-channel vs joint frame (invariant_function=VNJoint) -
    cx = centers[4]
    panel_title(ax, cx, title_y, "5. per-channel vs joint frame")

    def per_channel_readout(vecs):
        """Mirror of the shipped per-channel path (dim=1 per channel)."""
        rows = []
        for v in vecs:
            v_hat = vn_linear(w_proj, v[None, :])
            rows.append(readout(v_hat, vn_frame(v_hat, w_lift1, w_relu,
                                                u_relu)).ravel())
        return np.concatenate(rows)

    ZA = np.stack([dir_vec(15.0, 1.0), dir_vec(105.0, 1.0)])
    ZB = np.stack([dir_vec(15.0, 1.0), dir_vec(160.0, 1.0)])
    pcA, pcB = per_channel_readout(ZA), per_channel_readout(ZB)
    jA = readout(ZA, vn_frame(ZA, w_lift, w_relu, u_relu)).ravel()
    jB = readout(ZB, vn_frame(ZB, w_lift, w_relu, u_relu)).ravel()
    assert np.allclose(pcA, pcB), "per-channel should be blind to shape"
    assert not np.allclose(jA, jB), "joint frame should see shape"

    s5 = 0.75
    for o_x, vecs, lab in ((cx - 0.95, ZA, "A"), (cx + 0.95, ZB, "B")):
        o = (o_x, cy - 0.45)
        for v in vecs:
            vec_arrow(ax, o, s5 * v, VEC_COLOR, lw=1.5)
        ax.plot([o[0]], [o[1]], marker="o", color="#333333", markersize=2.0,
                zorder=5)
        ax.text(o_x, cy - 0.95, lab, fontsize=10, color="#666666",
                ha="center", va="center")
    panel_note(ax, cx, note_y, [
        "A and B: equal norms, different inter-channel angle",
    ], size=9)
    ax.text(cx, note_y - 0.30, f"per-channel: A = B = {fmt(pcA)}  (blind)",
            fontsize=8.8, color="#C58B00", ha="center", va="center")
    ax.text(cx, note_y - 0.60, f"joint frame: A {fmt(jA)}",
            fontsize=8.8, color="#3D8B2F", ha="center", va="center")
    ax.text(cx, note_y - 0.90, r"$\neq$ B " + f"{fmt(jB)}",
            fontsize=8.8, color="#3D8B2F", ha="center", va="center")

    for x0, colr, lab in ((8.3, VEC_COLOR, "input vectors"),
                          (10.5, FRAME_COLOR, "predicted frame"),
                          (12.8, DIR_COLOR, "query direction")):
        arrow(ax, (x0, 0.14), (x0 + 0.34, 0.14), colr, lw=1.5, mutation=10)
        ax.text(x0 + 0.46, 0.14, lab, fontsize=9, color="#555555",
                ha="left", va="center")
    ax.text(20.9, 0.12, "scratch explainer, not a thesis asset",
            fontsize=7.5, color="#AAAAAA", ha="right", va="center")

    print("frame before:", fmt(Q))
    print("frame after (rotated inputs):", fmt(Qr))
    print("read-out before:", fmt(M))
    print("read-out after:", fmt(Mr))
    print("panel-4 frame cross product (parallel => 0):", cross)
    print("panel-4 read-out:", fmt(inv_a),
          "| same norm, angle 38->170 deg:", fmt(inv_b))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "publication" / "figures" / "scratch_vn_invariance",
        help="Output stem without extension")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(f"{args.output}.png", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"[saved] {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
