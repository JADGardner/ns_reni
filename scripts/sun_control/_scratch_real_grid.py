"""Tone-map a diverse sample of real RENI_HDR EXRs into a PNG grid for
visual comparison against the synthetic generator."""
import glob
import os

import numpy as np
import pyexr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paths = sorted(glob.glob("/workspace/phd/data/RENI_HDR/train/*.exr"))
if not paths:
    paths = sorted(glob.glob("/home/james/data/RENI_HDR/train/*.exr"))
# Evenly spaced diverse sample across the archive.
n = 24
idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
sel = [paths[i] for i in idx]

cols = 4
rows = (n + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows))
for k in range(rows * cols):
    ax = axs.flat[k]
    ax.axis("off")
    if k >= n:
        continue
    img = pyexr.read(sel[k])[..., :3].astype(np.float32)
    img = np.clip(img, 0, None)
    # Resize to canonical 64x128 by simple block-mean if larger.
    q = np.quantile(img, 0.98)
    tone = np.clip(img / max(q, 1e-6), 0, 1) ** (1 / 2.2)
    ax.imshow(tone)
    lum = (img * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
    ax.set_title(f"{os.path.basename(sel[k])}  {img.shape[1]}x{img.shape[0]}",
                 fontsize=7)
fig.tight_layout()
out = "outputs/sun_control/real_sample_grid.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"[saved] {out}")
