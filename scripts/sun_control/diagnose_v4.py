"""Diagnose the v4 control failure: tie integrity, train-pair reconstruction,
and controllability from fitted (on-manifold) bases vs the mean latent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/figures")
from _common import equirect_ray_bundle, load_model
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.tonemap import two_bracket_to_linear, luminance

RUN = Path("outputs/reni_sun_synth_v4_d100")
CH, H, W, DEVICE = 9, 64, 128, "cuda:0"

labels = json.loads(Path("data/RENI_SUN_SYNTH_V4/train/sun_labels.json").read_text())
names = sorted(labels)
gt = F.normalize(torch.tensor([labels[n]["sun_direction"] for n in names]), dim=-1)
pairs = {}
for i, n in enumerate(names):
    pairs.setdefault(labels[n]["pair_id"], {})[labels[n]["member"]] = i

_, _, model = load_model(RUN, device=DEVICE)
model.eval()
bank = model.field.train_mu.detach()
rb = equirect_ray_bundle(DEVICE, idx=0, height=H)

# 1. tie integrity: within-pair vs random-pair non-ch9 distance
keep = [c for c in range(bank.shape[1]) if c != CH]
a_idx = torch.tensor([m["a"] for m in pairs.values()])
b_idx = torch.tensor([m["b"] for m in pairs.values()])
d_pair = (bank[a_idx][:, keep] - bank[b_idx][:, keep]).norm(dim=-1).mean()
perm = torch.randperm(len(a_idx))
d_rand = (bank[a_idx][:, keep] - bank[b_idx[perm]][:, keep]).norm(dim=-1).mean()
print(f"[tie] within-pair non-ch9 dist {d_pair:.4f} vs shuffled {d_rand:.4f}")

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

def peak_dir(lin):
    lum = luminance(lin).reshape(-1)
    sel = lum >= torch.quantile(lum, 0.9995)
    return F.normalize((rb.directions[sel] * lum[sel][:, None]).sum(0), dim=0)

# 2. train-latent reconstruction: does the decode place the sun at GT?
errs = []
for i in np.random.default_rng(0).choice(len(names), 24, replace=False):
    with torch.no_grad():
        p = peak_dir(decode(bank[int(i)]))
    errs.append(float(torch.rad2deg(torch.acos(
        (p @ gt[int(i)].to(DEVICE)).clamp(-1, 1)))))
errs = np.array(errs)
print(f"[recon] train-latent decoded peak vs GT sun: "
      f"median {np.median(errs):.2f} deg, mean {errs.mean():.2f} deg")

# 3. control from fitted bases vs the mean base
def sweep(base, tag):
    norm = float(bank[:, CH].norm(dim=-1).median())
    es = []
    for el_deg in (15.0, 40.0):
        for az_deg in np.linspace(-150, 150, 6):
            el, az = np.radians(el_deg), np.radians(az_deg)
            d = torch.tensor([np.cos(el)*np.sin(az), np.sin(el),
                              np.cos(el)*np.cos(az)],
                             dtype=torch.float32, device=DEVICE)
            z = base.clone(); z[CH] = norm * d
            with torch.no_grad():
                p = peak_dir(decode(z))
            es.append(float(torch.rad2deg(torch.acos((p @ d).clamp(-1, 1)))))
    es = np.array(es)
    print(f"[control:{tag}] median {np.median(es):.2f} deg, "
          f"mean {es.mean():.2f} deg over {len(es)}")

for j in (0, 500, 1500):
    sweep(bank[j].clone(), f"fitted{j}")
sweep(bank.mean(0), "mean")
