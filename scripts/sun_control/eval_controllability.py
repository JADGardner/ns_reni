"""Measure sun-position controllability of the supervised channel.

1. Alignment: cosine between each train-bank channel vector and its GT sun.
2. Control: set the channel to commanded directions on a grid (base = mean
   latent), decode, and measure the decoded luminance peak's angular error
   against the command.
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

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/reni_sun_synth_d100")
CH = 9
H, W = 64, 128
DEVICE = "cuda:0"

import json
labels = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "data/RENI_SUN_SYNTH/train/sun_labels.json").read_text())
names = sorted(labels)
gt = torch.tensor([labels[n]["sun_direction"] for n in names])
gt = torch.cat([gt, gt * torch.tensor([-1.0, 1.0, 1.0])])
gt = F.normalize(gt, dim=-1)

_, _, model = load_model(RUN, device=DEVICE)
model.eval()
bank = model.field.train_mu.detach().cpu()
z9 = F.normalize(bank[:, CH, :], dim=-1)
cos = (z9 * gt).sum(-1)
print(f"[align] train-bank ch{CH} vs GT sun: mean cos {cos.mean():.4f}, "
      f"median angular err {torch.rad2deg(torch.acos(cos.clamp(-1,1))).median():.2f} deg")

rb = equirect_ray_bundle(DEVICE, idx=0, height=H)
base = bank.mean(0).to(DEVICE)
norm = float(bank[:, CH].norm(dim=-1).median())
print(f"[control] base=mean latent, channel norm {norm:.3f}")

errs = []
for el_deg in (10.0, 25.0, 40.0, 55.0):
    for az_deg in np.linspace(-150, 150, 8):
        el, az = np.radians(el_deg), np.radians(az_deg)
        d = torch.tensor([np.cos(el)*np.sin(az), np.sin(el),
                          np.cos(el)*np.cos(az)], dtype=torch.float32,
                         device=DEVICE)
        z = base.clone(); z[CH] = norm * d
        with torch.no_grad():
            outs = []
            for s in range(0, rb.origins.shape[0], 65536):
                e = s + 65536
                sm = model.create_ray_samples(rb.origins[s:e], rb.directions[s:e],
                                              rb.camera_indices[s:e])
                o = model.field.forward(sm, rotation=None,
                    latent_codes=z.unsqueeze(0).repeat(sm.shape[0],1,1))[RENIFieldHeadNames.RGB]
                outs.append(two_bracket_to_linear(o, m_ldr=model.tonemap_m_ldr,
                                                  m_log=model.tonemap_m_log))
            lin = torch.cat(outs).reshape(H, W, 3)
        lum = luminance(lin).reshape(-1)
        thresh = torch.quantile(lum, 0.9995)
        sel = lum >= thresh
        w_ = lum[sel]
        peak = F.normalize((rb.directions[sel] * w_[:, None]).sum(0), dim=0)
        err = float(torch.rad2deg(torch.acos((peak @ d).clamp(-1, 1))))
        errs.append(err)
errs = np.array(errs)
print(f"[control] peak-tracking error over {len(errs)} commanded directions: "
      f"mean {errs.mean():.2f} deg, median {np.median(errs):.2f} deg, "
      f"max {errs.max():.2f} deg")
