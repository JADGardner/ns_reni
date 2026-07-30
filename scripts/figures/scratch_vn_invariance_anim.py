"""Animated companion to scratch_vn_invariance.py (temporary, not a thesis
asset). One continuous gravity-axis rotation drives three panels:

  A. fixed-axis coordinates of a latent vector (they change);
  B. the general two-channel VN frame read-out (frozen while spinning);
  C. the RENI++ single-vector channel: VNInv_2 frozen, while the query
     inner product with a fixed direction d_perp sweeps (equivariance).

All quantities are recomputed every frame with the real VNLinear/VNReLU
rules and the invariants are asserted against frame 0.

Run from the ns_reni repo root:

    PYTHONPATH=. python scripts/figures/scratch_vn_invariance_anim.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scratch_vn_invariance import (
    AXIS_COLOR, DIR_COLOR, FRAME_COLOR, VEC_COLOR,
    dir_vec, dotted_projection, fmt, plane_axes, rot,
    readout, vec_arrow, vn_frame, vn_linear,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGING = "#B0413E"
FROZEN = "#3D8B2F"
BLIND = "#C58B00"


def build_animation(n_frames: int = 90, fps: int = 15):
    rng = np.random.default_rng(12)
    w_lift = rng.normal(size=(2, 2))
    w_relu = rng.normal(size=(2, 2))
    u_relu = rng.normal(size=(2, 2))
    w_proj = rng.normal(size=(1, 1))
    w_lift1 = rng.normal(size=(2, 1))

    z_a0 = dir_vec(22.0, 1.05)
    Z0 = np.stack([dir_vec(20.0, 0.95), dir_vec(115.0, 0.75)])
    z_c0 = dir_vec(38.0, 1.0)
    d_perp = dir_vec(-15.0, 0.90)

    M0 = readout(Z0, vn_frame(Z0, w_lift, w_relu, u_relu))
    zc_hat0 = vn_linear(w_proj, z_c0[None, :])
    inv0 = readout(zc_hat0, vn_frame(zc_hat0, w_lift1, w_relu, u_relu))

    def per_channel_readout(vecs):
        rows = []
        for v in vecs:
            v_hat = vn_linear(w_proj, v[None, :])
            rows.append(readout(v_hat, vn_frame(v_hat, w_lift1, w_relu,
                                                u_relu)).ravel())
        return np.concatenate(rows)

    pc0 = per_channel_readout(np.stack([dir_vec(25.0, 0.95),
                                        dir_vec(115.0, 0.75)]))

    fig, ax = plt.subplots(figsize=(19.8, 5.6), dpi=90)

    centers = [2.4, 7.0, 11.6, 16.4]
    cy = 3.15
    title_y = 5.28
    text_y = 1.30

    def draw(frame_idx):
        ax.clear()
        ax.set_xlim(0, 19.8)
        ax.set_ylim(0, 5.6)
        ax.set_aspect("equal")
        ax.axis("off")
        theta = 360.0 * frame_idx / n_frames
        R = rot(theta)

        # ---- A: fixed-axis coordinates change ---------------------------
        cx = centers[0]
        ax.text(cx, title_y, "fixed-axis coordinates", ha="center",
                fontsize=13, fontweight="bold")
        o = (cx - 0.2, cy - 0.7)
        plane_axes(ax, o, r=1.2)
        z_a = R @ z_a0
        vec_arrow(ax, o, z_a, VEC_COLOR, label=r"$\mathbf{z}_{\perp}$",
                  lw=1.8)
        dotted_projection(ax, o, z_a, np.array([1.0, 0.0]))
        dotted_projection(ax, o, z_a, np.array([0.0, 1.0]))
        ax.text(cx, text_y, f"coords = {fmt(z_a)}", ha="center",
                fontsize=11, color=CHANGING)
        ax.text(cx, text_y - 0.42, "changes", ha="center", fontsize=10,
                style="italic", color=CHANGING)

        # ---- B: general VN frame read-out is frozen ----------------------
        cx = centers[1]
        ax.text(cx, title_y, "VN frame read-out", ha="center",
                fontsize=13, fontweight="bold")
        o = (cx, cy - 0.7)
        Zt = Z0 @ R.T
        Qt = vn_frame(Zt, w_lift, w_relu, u_relu)
        Mt = readout(Zt, Qt)
        assert np.allclose(Mt, M0), "read-out drifted!"
        for v in Zt:
            vec_arrow(ax, o, v, VEC_COLOR, lw=1.8)
        for v in Qt:
            vec_arrow(ax, o, v, FRAME_COLOR, lw=1.8)
        ax.plot([o[0]], [o[1]], marker="o", color="#333333", markersize=2.4)
        ax.text(cx, text_y,
                r"$\langle\mathbf{z}_i,\mathbf{q}_j\rangle$ = "
                + fmt(Mt.ravel()),
                ha="center", fontsize=11, color=FROZEN)
        ax.text(cx, text_y - 0.42, "unchanged", ha="center", fontsize=10,
                style="italic", color=FROZEN)

        # ---- C: RENI++ channel, conditioning vs query --------------------
        cx = centers[2]
        ax.text(cx, title_y, "RENI++ channel: conditioning vs query",
                ha="center", fontsize=13, fontweight="bold")
        o = (cx - 0.2, cy - 0.7)
        z_c = R @ z_c0
        zc_hat = vn_linear(w_proj, z_c[None, :])
        Qc = vn_frame(zc_hat, w_lift1, w_relu, u_relu)
        inv = readout(zc_hat, Qc)
        assert np.allclose(inv, inv0), "VNInv_2 drifted!"
        q_dirs = Qc / np.linalg.norm(Qc, axis=1, keepdims=True)
        vec_arrow(ax, o, z_c, VEC_COLOR, label=r"$\mathbf{z}_{\perp}$",
                  lw=1.8)
        vec_arrow(ax, o, 0.72 * q_dirs[0], FRAME_COLOR, lw=1.4)
        vec_arrow(ax, o, 0.55 * q_dirs[1], FRAME_COLOR, lw=1.4)
        vec_arrow(ax, o, d_perp, DIR_COLOR, label=r"$\mathbf{d}_{\perp}$",
                  lw=1.8)
        zd = float(z_c @ d_perp)
        ax.text(cx, text_y,
                r"$\mathrm{VNInv}_2$ = " + fmt(inv.ravel())
                + "   unchanged",
                ha="center", fontsize=11, color=FROZEN)
        ax.text(cx, text_y - 0.42,
                r"$\langle\mathbf{z}_{\perp},\mathbf{d}_{\perp}\rangle$ = "
                + f"{zd:+.2f}   changes",
                ha="center", fontsize=11, color=CHANGING)

        # ---- D: latent shape change, per-channel vs joint frame ----------
        cx = centers[3]
        ax.text(cx, title_y, "shape change: per-channel vs joint",
                ha="center", fontsize=13, fontweight="bold")
        o = (cx - 0.2, cy - 0.7)
        phi = 115.0 + 40.0 * np.sin(np.deg2rad(theta))
        Zd = np.stack([dir_vec(25.0, 0.95), dir_vec(phi, 0.75)])
        frame_d = vn_frame(Zd, w_lift, w_relu, u_relu)
        joint_d = readout(Zd, frame_d)
        pc_d = per_channel_readout(Zd)
        assert np.allclose(pc_d, pc0), "per-channel moved despite fixed norms"
        for v in Zd:
            vec_arrow(ax, o, v, VEC_COLOR, lw=1.8)
        for v in frame_d:
            vec_arrow(ax, o, v, FRAME_COLOR, lw=1.4)
        ax.plot([o[0]], [o[1]], marker="o", color="#333333", markersize=2.4)
        ax.text(cx, text_y,
                f"per-channel = {fmt(pc_d)}   blind to the change",
                ha="center", fontsize=10.5, color=BLIND)
        ax.text(cx, text_y - 0.42,
                f"joint frame = {fmt(joint_d.ravel())}   sees it",
                ha="center", fontsize=10.5, color=FROZEN)

        ax.text(9.9, 0.28,
                r"A–C: one rigid rotation of everything (invariants freeze)."
                r"  D: norms fixed, the inter-channel angle varies.",
                ha="center", fontsize=10.5, color="#555555")
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    return fig, anim, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "publication" / "figures"
        / "scratch_vn_invariance.gif")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, anim, fps = build_animation(args.frames, args.fps)
    anim.save(str(args.output), writer=PillowWriter(fps=fps))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
