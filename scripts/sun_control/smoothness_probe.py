"""Quantify sun-control smoothness: decode along dragged ch9 paths and
record the decoded-sun trajectory plus frame-to-frame image change.

    PYTHONPATH=.:scripts/figures python scripts/sun_control/smoothness_probe.py \
        <run_dir> <tag>

Outputs outputs/sun_control/smoothness_<tag>.png (trajectory + delta plots)
and smoothness_<tag>.gif (tone-mapped azimuth sweep).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/figures")
from _common import equirect_ray_bundle, load_model
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.tonemap import two_bracket_to_linear, luminance

RUN = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "probe"
CH, H, W, DEVICE = 9, 96, 192, "cuda:0"
OUT = Path("outputs/sun_control")
OUT.mkdir(parents=True, exist_ok=True)

_, _, model = load_model(RUN, device=DEVICE)
model.eval()
bank = model.field.train_mu.detach()
rb = equirect_ray_bundle(DEVICE, idx=0, height=H)
norm = float(bank[:, CH].norm(dim=-1).median())
base = bank.mean(0)


def decode(z):
    outs = []
    for s in range(0, rb.origins.shape[0], 65536):
        e = s + 65536
        sm = model.create_ray_samples(rb.origins[s:e], rb.directions[s:e],
                                      rb.camera_indices[s:e])
        o = model.field.forward(sm, rotation=None,
            latent_codes=z.unsqueeze(0).repeat(sm.shape[0], 1, 1))[RENIFieldHeadNames.RGB]
        outs.append(two_bracket_to_linear(o, m_ldr=model.tonemap_m_ldr,
                                          m_log=model.tonemap_m_log))
    return torch.cat(outs).reshape(H, W, 3)


def el_az_dir(el_deg, az_deg):
    el, az = np.radians(el_deg), np.radians(az_deg)
    return torch.tensor([np.cos(el) * np.sin(az), np.sin(el),
                         np.cos(el) * np.cos(az)],
                        dtype=torch.float32, device=DEVICE)


def peak(lin):
    lum = luminance(lin).reshape(-1)
    sel = lum >= torch.quantile(lum, 0.999)
    d = F.normalize((rb.directions[sel] * lum[sel][:, None]).sum(0), dim=0)
    return (np.degrees(np.arcsin(np.clip(float(d[1]), -1, 1))),
            np.degrees(np.arctan2(float(d[0]), float(d[2]))))


def sweep(path_dirs):
    imgs, peaks, deltas = [], [], []
    prev = None
    for d in path_dirs:
        z = base.clone(); z[CH] = norm * d
        with torch.no_grad():
            lin = decode(z)
        imgs.append(lin)
        peaks.append(peak(lin))
        if prev is not None:
            deltas.append(float(((lin - prev).pow(2).mean().sqrt()
                                 / prev.pow(2).mean().sqrt().clamp(min=1e-8))))
        prev = lin
    return imgs, peaks, deltas


az_grid = np.linspace(-180, 180, 65)
el_fixed = 25.0
az_imgs, az_peaks, az_deltas = sweep([el_az_dir(el_fixed, a) for a in az_grid])

el_grid = np.linspace(2, 85, 49)
az_fixed = 0.0
_, el_peaks, el_deltas = sweep([el_az_dir(e, az_fixed) for e in el_grid])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
axs[0].plot(az_grid, [p[1] for p in az_peaks], ".-", ms=3, label="decoded")
axs[0].plot(az_grid, az_grid, "k--", lw=0.8, label="commanded")
axs[0].set(xlabel="commanded azimuth (deg)", ylabel="decoded peak azimuth",
           title=f"azimuth sweep @ el={el_fixed:.0f}")
axs[0].legend(fontsize=8)
axs[1].plot(el_grid, [p[0] for p in el_peaks], ".-", ms=3, label="decoded")
axs[1].plot(el_grid, el_grid, "k--", lw=0.8, label="commanded")
axs[1].set(xlabel="commanded elevation (deg)", ylabel="decoded peak elevation",
           title=f"elevation sweep @ az={az_fixed:.0f}")
axs[1].legend(fontsize=8)
axs[2].plot(az_grid[1:], az_deltas, label="azimuth path")
axs[2].plot(el_grid[1:], el_deltas, label="elevation path")
axs[2].set(xlabel="command (deg)", ylabel="relative RMS frame delta",
           title="step-to-step image change")
axs[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / f"smoothness_{TAG}.png", dpi=130)
print(f"[saved] {OUT}/smoothness_{TAG}.png")
print(f"[az] decoded-peak azimuth err median "
      f"{np.median(np.abs(((np.array([p[1] for p in az_peaks]) - az_grid + 180) % 360) - 180)):.1f} deg; "
      f"delta mean {np.mean(az_deltas):.4f} max {np.max(az_deltas):.4f}")
print(f"[el] decoded-peak elevation err median "
      f"{np.median(np.abs(np.array([p[0] for p in el_peaks]) - el_grid)):.1f} deg; "
      f"delta mean {np.mean(el_deltas):.4f} max {np.max(el_deltas):.4f}")

import imageio.v3 as iio
frames = []
for lin in az_imgs:
    x = lin / torch.quantile(lin.reshape(-1, 3).max(-1).values, 0.97).clamp(min=1e-6)
    frames.append((x.clamp(0, 1) ** (1 / 2.2) * 255).byte().cpu().numpy())
iio.imwrite(OUT / f"smoothness_{TAG}.gif", frames, duration=80, loop=0)
print(f"[saved] {OUT}/smoothness_{TAG}.gif")
