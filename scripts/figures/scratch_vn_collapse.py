"""Scratch explainer #2: WHY the per-channel VN invariant collapses.

Companion to scratch_vn_invariance.py, digging one level deeper: the layer
algebra that forces single-channel frames to be parallel (panel 1), what the
joint frame computes instead (panel 2), the pointwise aliasing the collapse
causes (panel 3), why the lost inter-channel angles are nevertheless still
available to the decoder through the query stream (panel 4), and what the
RENI training runs said (panel 5).

All VN ops are the same numpy mirrors of reni/field_components/vn_layers.py
used in scratch_vn_invariance.py; every claim is asserted at run time.

Run from the ns_reni repo root:

    python scripts/figures/scratch_vn_collapse.py
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
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]

VEC_COLOR = "#7C9AD6"      # latent vectors
FRAME_COLOR = "#3A5FAD"    # VN-predicted frame vectors
DIR_COLOR = "#3D8B2F"      # query direction
ALT_COLOR = "#C58B00"      # the aliasing twin
GHOST_ALPHA = 0.30
AXIS_COLOR = "#B0B0B0"


# ---- numpy mirrors of vn_layers.py (as in scratch_vn_invariance.py) ------

def vn_linear(weight: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return weight @ vecs


def vn_relu(w, u, vecs, eps=1e-6):
    q = w @ vecs
    k = u @ vecs
    qk = (q * k).sum(-1, keepdims=True)
    k_norm = np.sqrt((k ** 2).sum(-1, keepdims=True)).clip(min=eps)
    q_projected_on_k = q - (q * (k / k_norm)).sum(-1, keepdims=True) * k
    return np.where(qk >= 0.0, q, q_projected_on_k)


def vn_frame(vecs, w_lift, w_relu, u_relu):
    return vn_relu(w_relu, u_relu, vn_linear(w_lift, vecs))


def readout(vecs, frame):
    return vecs @ frame.T


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def reflect(deg):
    """Reflection across the line at angle deg."""
    return rot(deg) @ np.diag([1.0, -1.0]) @ rot(-deg)


def dir_vec(deg, length=1.0):
    a = np.deg2rad(deg)
    return length * np.array([np.cos(a), np.sin(a)])


# ---- drawing helpers (as in scratch_vn_invariance.py) --------------------

def arrow(ax, start, end, color, lw=1.6, mutation=13, alpha=1.0, zorder=4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=to_rgba(color, alpha), shrinkA=0, shrinkB=0,
        zorder=zorder))


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


def panel_title(ax, cx, y, text):
    ax.text(cx, y, text, ha="center", va="center", fontsize=12.5,
            fontweight="bold")


def panel_note(ax, cx, y, lines, size=9.5, dy=0.30, color="#444444"):
    for i, line in enumerate(lines):
        ax.text(cx, y - i * dy, line, ha="center", va="center",
                fontsize=size, color=color)


def fmt(x):
    return np.array2string(np.asarray(x), separator=", ",
                           formatter={"float_kind": lambda v: f"{v:.2f}"})


# --------------------------------------------------------------------------

def build_figure():
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(21.0, 5.9), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 21.0)
    ax.set_ylim(0, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")

    centers = [2.1, 6.3, 10.5, 14.6, 18.7]
    cy = 3.15
    title_y = 5.55
    note_y = 1.35

    # ---- Panel 1: the algebra of the collapse ---------------------------
    cx = centers[0]
    panel_title(ax, cx, title_y, "1. one channel in, parallel out")
    origin = (cx - 0.2, cy - 0.35)
    v = dir_vec(35.0, 0.95)
    w_lift1 = np.array([[1.35], [-0.7]])        # VNLinear(1->2)
    w_relu = rng.normal(size=(2, 2))
    u_relu = rng.normal(size=(2, 2))
    lifted = vn_linear(w_lift1, v[None, :])     # rows: w_a v, w_b v
    Q1 = vn_frame(v[None, :], w_lift1, w_relu, u_relu)
    for q in np.concatenate([lifted, Q1]):
        cross = float(v[0] * q[1] - v[1] * q[0])
        assert abs(cross) < 1e-9, "single-channel outputs must be parallel"
    inv = readout(v[None, :], Q1).ravel()
    vec_arrow(ax, origin, v, VEC_COLOR, label=r"$\mathbf{v}$",
              label_off=(-0.05, 0.20))
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    o_off = (origin[0] + 0.05 * perp[0], origin[1] + 0.05 * perp[1])
    vec_arrow(ax, o_off, lifted[0], FRAME_COLOR, lw=1.3, alpha=0.85,
              label=r"$w_a\mathbf{v}$", label_off=(0.30, 0.02))
    vec_arrow(ax, origin, lifted[1], FRAME_COLOR, lw=1.3, alpha=0.5,
              label=r"$w_b\mathbf{v}$", label_off=(-0.32, -0.06))
    panel_note(ax, cx, note_y, [
        r"VNLinear mixes channels with real scalars:",
        r"one channel $\Rightarrow$ every output is $w\,\mathbf{v}$"
        r" (rescale or flip);",
        r"VNReLU keeps or zeroes it (never rotates);",
        r"read-out $\langle\mathbf{v}, w\mathbf{v}\rangle"
        r" = w\Vert\mathbf{v}\Vert^2$: the angle is gone",
    ], size=9)

    # ---- Panel 2: the joint frame escapes the ray ------------------------
    cx = centers[1]
    panel_title(ax, cx, title_y, "2. many channels: new directions")
    origin = (cx - 0.1, cy - 0.35)
    Z = np.stack([dir_vec(15.0, 0.95), dir_vec(95.0, 0.8),
                  dir_vec(205.0, 0.6)])
    w_liftN = rng.normal(size=(1, 3))
    q_joint = vn_linear(w_liftN, Z)[0]
    for z in Z:
        assert abs(z[0] * q_joint[1] - z[1] * q_joint[0]) > 1e-3
    c = Z @ q_joint                     # readout against the joint frame
    gram = Z @ Z.T
    w = w_liftN.ravel()
    assert np.allclose(c, gram @ w), "read-out must be a Gram slice"
    for i, z in enumerate(Z):
        vec_arrow(ax, origin, z, VEC_COLOR, lw=1.4,
                  label=rf"$\mathbf{{z}}_{{{i + 1}}}$")
    vec_arrow(ax, origin, q_joint, FRAME_COLOR,
              label=r"$\mathbf{q}=\sum_j w_j\mathbf{z}_j$",
              label_off=(0.05, -0.24))
    panel_note(ax, cx, note_y, [
        r"the frame is a learned combination of ALL channels,",
        r"so it points where no single channel does;",
        r"$c_i=\langle\mathbf{z}_i,\mathbf{q}\rangle=\sum_j w_j"
        r"\langle\mathbf{z}_i,\mathbf{z}_j\rangle$:",
        r"each read-out is a slice of the Gram matrix (asserted)",
    ], size=9)

    # ---- Panel 3: pointwise aliasing --------------------------------------
    cx = centers[2]
    panel_title(ax, cx, title_y, "3. what the collapse costs")
    L_DEG = 0.0
    z1 = dir_vec(30.0, 1.0)
    z2 = dir_vec(100.0, 0.8)
    z2_alt = reflect(L_DEG) @ z2               # same norm, new angle to z1
    d = dir_vec(L_DEG, 0.9)                    # on the mirror line
    d_far = dir_vec(70.0, 0.9)
    assert np.isclose(np.linalg.norm(z2), np.linalg.norm(z2_alt))
    assert abs(z1 @ z2 - z1 @ z2_alt) > 1e-2          # Gram differs
    assert np.isclose(z2 @ d, z2_alt @ d)             # same input at d
    assert abs(z2 @ d_far - z2_alt @ d_far) > 1e-2    # differs elsewhere
    origin = (cx - 0.15, cy - 0.35)
    vec_arrow(ax, origin, z1, VEC_COLOR, label=r"$\mathbf{z}_1$")
    vec_arrow(ax, origin, z2, VEC_COLOR, label=r"$\mathbf{z}_2$")
    vec_arrow(ax, origin, z2_alt, ALT_COLOR, label=r"$\mathbf{z}_2'$")
    vec_arrow(ax, origin, d, DIR_COLOR, label=r"$\mathbf{d}$",
              label_off=(0.18, -0.06))
    vec_arrow(ax, origin, d_far, DIR_COLOR, alpha=GHOST_ALPHA,
              label=r"$\mathbf{d}'$", label_off=(0.10, 0.16))
    ax.plot([origin[0] - 1.05 * np.cos(np.deg2rad(L_DEG)),
             origin[0] + 1.15 * np.cos(np.deg2rad(L_DEG))],
            [origin[1] - 1.05 * np.sin(np.deg2rad(L_DEG)),
             origin[1] + 1.15 * np.sin(np.deg2rad(L_DEG))],
            linestyle=(0, (2, 3)), color="#999999", lw=0.9, zorder=2)
    panel_note(ax, cx, note_y, [
        r"$\mathbf{z}_2'$: reflect $\mathbf{z}_2$ across the dotted line:"
        r" norms equal, Gram differs;",
        r"norm-only conditioning is identical for both, AND at"
        r" $\mathbf{d}$ on the line",
        r"the projections agree too, so the decoder MUST emit the same"
        r" value there",
        r"for two different environments (asserted). At $\mathbf{d}'$"
        r" they differ.",
    ], size=9)

    # ---- Panel 4: the query stream still carries the angles --------------
    cx = centers[3]
    panel_title(ax, cx, title_y, "4. the angles survive in the query stream")
    theta = np.linspace(0.0, 2 * np.pi, 721)
    D = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    p1, p2, p2a = D @ z1, D @ z2, D @ z2_alt
    gram_rec = np.trapezoid(p1 * p2, theta) / np.pi
    assert np.isclose(gram_rec, z1 @ z2, atol=1e-6), "Gram not recovered"
    sub = fig.add_axes([0.635, 0.30, 0.118, 0.40])
    sub.plot(np.rad2deg(theta), p1, color=VEC_COLOR, lw=1.4,
             label=r"$\langle\mathbf{z}_1,\mathbf{d}(\theta)\rangle$")
    sub.plot(np.rad2deg(theta), p2, color=FRAME_COLOR, lw=1.4,
             label=r"$\langle\mathbf{z}_2,\mathbf{d}(\theta)\rangle$")
    sub.plot(np.rad2deg(theta), p2a, color=ALT_COLOR, lw=1.2,
             linestyle="--",
             label=r"$\langle\mathbf{z}_2',\mathbf{d}(\theta)\rangle$")
    sub.axhline(0.0, color="#CCCCCC", lw=0.6)
    sub.set_xlim(0, 360)
    sub.set_xticks([0, 180, 360])
    sub.set_xlabel(r"query azimuth $\theta$ (deg)", fontsize=8)
    sub.tick_params(labelsize=7)
    sub.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    for s in ("top", "right"):
        sub.spines[s].set_visible(False)
    panel_note(ax, cx, note_y, [
        r"$\langle\mathbf{z}_i,\mathbf{d}(\theta)\rangle="
        r"\Vert\mathbf{z}_i\Vert\cos(\theta-\varphi_i)$:"
        r" amplitude = norm, phase gap = angle;",
        r"$\frac{1}{\pi}\int_0^{2\pi}\langle\mathbf{z}_1,\mathbf{d}\rangle"
        r"\langle\mathbf{z}_2,\mathbf{d}\rangle\,d\theta"
        r"=\langle\mathbf{z}_1,\mathbf{z}_2\rangle$"
        f"  (recovered: {gram_rec:.3f} = {z1 @ z2:.3f}, asserted);",
        r"across the sphere the decoder already sees everything the joint"
        r" frame adds,",
        r"just not at any single query point",
    ], size=9)

    # ---- Panel 5: what training said --------------------------------------
    cx = centers[4]
    panel_title(ax, cx, title_y, "5. why RENI training does not move")
    panel_note(ax, cx, cy + 1.45, [
        "D=100 two-bracket latent-reset 2cyc, only the",
        "invariant function changed (wandb 704f7kk7):",
        "",
        "full refit (21 imgs):  joint 14.17/21.72",
        "vs per-channel 14.46/22.00 psnr hdr/ldr  (tie)",
        "",
        "frustum outpaint:  joint fits visible best (+1.4 dB)",
        "but completes hidden WORSE (−0.47 dB LDR)",
    ], size=9.3, dy=0.34)
    panel_note(ax, cx, note_y - 0.30, [
        "the recovered angles are redundant (panel 4), the",
        "latents are free to dodge the aliasing (panel 3),",
        "and the norm bottleneck regularises completion",
    ], size=9.3, dy=0.30, color="#8A2F2F")

    ax.text(20.9, 0.12, "scratch explainer, not a thesis asset",
            fontsize=7.5, color="#AAAAAA", ha="right", va="center")

    print("panel 1 read-out (norm-only):", fmt(inv))
    print("panel 2 joint read-out:", fmt(c), "= Gram @ w:", fmt(gram @ w))
    print("panel 3 Gram:", f"{z1 @ z2:.3f}", "vs twin:", f"{z1 @ z2_alt:.3f}",
          "| proj at d:", f"{z2 @ d:.3f}", "=", f"{z2_alt @ d:.3f}")
    print("panel 4 Gram recovered from query stream:",
          f"{gram_rec:.4f} vs {z1 @ z2:.4f}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "publication" / "figures" / "scratch_vn_collapse",
        help="Output stem without extension")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(f"{args.output}.png", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"[saved] {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
