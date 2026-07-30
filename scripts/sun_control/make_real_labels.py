"""Pseudo sun labels for RENI_HDR: luminance-peak direction + confidence.

Confidence = clamp(peak/p99 luminance ratio / 15, 0, 1): concentrated suns
supervise at full weight, overcast skies barely at all.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/figures")
from _common import REPO_ROOT, equirect_ray_bundle
from reni.utils.tonemap import apply_fixed_gauge, luminance

H, W, DEVICE = 64, 128, "cuda:0"
rb = equirect_ray_bundle(DEVICE, idx=0, height=H)

labels = {}
files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / "train" / "*.exr")))
for i, f in enumerate(files):
    import pyexr
    img = pyexr.read(f).astype("float32")[..., :3]
    fin = np.isfinite(img)
    if not fin.all():
        img[~fin] = img[fin].max()
    img[img <= 0] = img[img > 0].min()
    t = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None], size=(H, W),
                      mode="bilinear")[0].permute(1, 2, 0)
    g = apply_fixed_gauge(t.to(DEVICE))
    lum = luminance(g).reshape(-1)
    # Peak-pixel-first: the sun disc is orders brighter than anything else,
    # so anchor on the brightest near-or-above-horizon pixel, then take a
    # solid-angle-weighted centroid of a small window around it. A global
    # top-quantile centroid instead drags toward sun-lit ground (biasing
    # elevation below the horizon), and a hard sky-only cut drops horizon
    # suns at this resolution and grabs bright clouds.
    y = rb.directions[:, 1]
    cand = y > -0.052                                   # elevation > ~-3 deg
    # Anchor on the brightest candidate of the SMOOTHED luminance: isolated
    # hot pixels in the sky otherwise hijack the argmax.
    sm = F.avg_pool2d(luminance(g).reshape(1, 1, H, W), 3, 1, 1).reshape(-1)
    peak = int(torch.argmax(torch.where(cand, sm, torch.zeros_like(sm))))
    # Window centroid is symmetric about the peak (no sky cut): for horizon
    # suns the below-horizon half of the glow balances the half above.
    near = (rb.directions @ rb.directions[peak]) > math.cos(math.radians(15.0))
    sel = near & (lum >= 0.05 * lum[peak])
    w = lum[sel] * torch.sqrt((1.0 - y[sel] ** 2).clamp(min=1e-6))
    d = F.normalize((rb.directions[sel] * w[:, None]).sum(0), dim=0)
    ratio = float(sm.max())
    labels[Path(f).name] = {
        "sun_direction": [float(v) for v in d],
        "weight": float(min(ratio / 15.0, 1.0)),
    }
    if (i + 1) % 400 == 0:
        print(f"  {i + 1}/{len(files)}")
out = REPO_ROOT / "data" / "RENI_HDR" / "train" / "sun_labels_pseudo.json"
out.write_text(json.dumps(labels, indent=1))
ws = np.array([v["weight"] for v in labels.values()])
print(f"[labels] {len(labels)} images, mean weight {ws.mean():.2f}, "
      f"full-weight fraction {(ws >= 0.999).mean():.2f} -> {out}")
