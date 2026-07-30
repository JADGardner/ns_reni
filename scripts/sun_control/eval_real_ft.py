"""Evaluate the real-data sun-control fine-tune.

1. Alignment: bank ch9 vs pseudo-label sun direction, split by label
   confidence (weight 1.0 = clearly sunny).
2. Visual command sweep: from sunny fitted real bases, overwrite ch9 with
   commanded directions and save a tone-mapped grid with the command marked.
"""
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

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/reni_sun_real_ft_d100")
LABELS = Path(sys.argv[2] if len(sys.argv) > 2
              else "data/RENI_HDR/train/sun_labels_pseudo.json")
TAG = sys.argv[3] if len(sys.argv) > 3 else "real_ft"
CH, H, W, DEVICE = 9, 96, 192, "cuda:0"
OUT = Path("outputs/sun_control")
OUT.mkdir(parents=True, exist_ok=True)

labels = json.loads(LABELS.read_text())
names = sorted(labels)
gt = torch.tensor([labels[n]["sun_direction"] for n in names])
gt = torch.cat([gt, gt * torch.tensor([-1.0, 1.0, 1.0])])
gt = F.normalize(gt, dim=-1)
wts = torch.tensor([float(labels[n].get("weight", 1.0)) for n in names])
wts = torch.cat([wts, wts])

_, _, model = load_model(RUN, device=DEVICE)
model.eval()
bank = model.field.train_mu.detach()
z9 = F.normalize(bank[:, CH, :], dim=-1).cpu()
cos = (z9 * gt).sum(-1)
err = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
real = torch.tensor([not n.startswith("synth") for n in names])
real = torch.cat([real, real])
sunny = (wts >= 0.999) & real
print(f"[align] all {len(err)} rows: median err {err.median():.2f} deg")
print(f"[align] real sunny (w=1, n={int(sunny.sum())}): "
      f"median err {err[sunny].median():.2f} deg, mean {err[sunny].mean():.2f} deg")

rb = equirect_ray_bundle(DEVICE, idx=0, height=H)

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

def tonemap(lin):
    x = lin / torch.quantile(lin.reshape(-1, 3).max(-1).values, 0.97).clamp(min=1e-6)
    return (x.clamp(0, 1) ** (1 / 2.2) * 255).byte().cpu().numpy()

def mark(img, d):
    az = np.arctan2(d[0], d[2])
    pol = np.arccos(np.clip(d[1], -1, 1))
    c = int((az / (2 * np.pi) + 0.5) * W) % W
    r = int(pol / np.pi * H)
    r0, r1 = max(r - 1, 0), min(r + 2, H)
    img[r0:r1, max(c - 6, 0):max(c - 2, 1)] = (255, 40, 40)
    img[r0:r1, min(c + 3, W - 1):min(c + 7, W)] = (255, 40, 40)
    img[max(r - 6, 0):max(r - 2, 1), max(c - 1, 0):min(c + 2, W)] = (255, 40, 40)
    img[min(r + 3, H - 1):min(r + 7, H), max(c - 1, 0):min(c + 2, W)] = (255, 40, 40)
    return img

# sunny fitted bases: full-weight REAL rows from the original half
n_imgs = len(names)
cand = [i for i in range(n_imgs)
        if wts[i] >= 0.999 and not names[i].startswith("synth")]
picks = [cand[j] for j in np.linspace(0, len(cand) - 1, 3).astype(int)]
print(f"[sweep] bases: {[names[i] for i in picks]}")

ELS = (10.0, 30.0, 55.0, 80.0)
AZS = np.linspace(-135, 135, 5)
norm = float(bank[:, CH].norm(dim=-1).median())
rows = []
for i in picks:
    base = bank[i].clone()
    with torch.no_grad():
        recon = tonemap(decode(base))
    row = [mark(recon.copy(), gt[i].numpy())]
    for el_deg, az_deg in [(e, a) for e in ELS for a in AZS]:
        el, az = np.radians(el_deg), np.radians(az_deg)
        d = torch.tensor([np.cos(el) * np.sin(az), np.sin(el),
                          np.cos(el) * np.cos(az)],
                         dtype=torch.float32, device=DEVICE)
        z = base.clone(); z[CH] = norm * d
        with torch.no_grad():
            img = tonemap(decode(z))
        row.append(mark(img, d.cpu().numpy()))
    rows.append(row)

ncol = 1 + len(ELS) * len(AZS)
pad = 2
canvas = np.full((len(rows) * (H + pad) - pad, ncol * (W + pad) - pad, 3),
                 255, dtype=np.uint8)
for r, row in enumerate(rows):
    for c, img in enumerate(row):
        canvas[r * (H + pad):r * (H + pad) + H,
               c * (W + pad):c * (W + pad) + W] = img

import imageio.v3 as iio
path = OUT / f"{TAG}_command_sweep.png"
iio.imwrite(path, canvas)
print(f"[sweep] wrote {path} (col 0 = reconstruction with pseudo-label mark; "
      f"then elevations {ELS} x azimuths {list(AZS)})")
