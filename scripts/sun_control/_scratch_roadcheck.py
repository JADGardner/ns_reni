"""Road diagnostic: find nuisance seeds where a road is present over grass,
and render them at a decent sun so the road contrast/edges are clearly visible."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.sun_control.synthetic_sky as ss

H, W = 64, 128
dirs = ss.erp_directions(H, W)


def tone(img):
    img = ss.apply_gauge(img)
    return (img / torch.quantile(img.flatten(), 0.98)).clamp(0, 1) ** (1 / 2.2)


# Scan seeds for road-present scenes, note style/ground.
hits = []
for seed in range(400):
    sc = ss._sample_scene(np.random.default_rng(seed))
    if sc["road"]:
        hits.append((seed, sc["road_style"], sc["ground_type"]))
grass_tar = [h for h in hits if h[1] == "tarmac" and h[2].startswith("grass")]
pale = [h for h in hits if h[1] == "pale"]
print(f"road-present seeds: {len(hits)}/400; grass+tarmac={len(grass_tar)}, "
      f"pale-track={len(pale)}")

# Pick 4 grass-tarmac and 4 pale-track examples; render at el=25 az=40.
pick = (grass_tar[:4] + pale[:4])[:8]
fig, axs = plt.subplots(2, 4, figsize=(4 * 4, 2.2 * 2))
for k, (seed, style, gtn) in enumerate(pick):
    p = dict(sun_elevation_deg=25.0, sun_azimuth_deg=40.0, turbidity=3.5,
             sun_intensity=40.0, sun_sharpness=350.0)
    img, _ = ss.render_sky(H, W, dirs=dirs,
                           nuisance_rng=np.random.default_rng(seed), **p)
    ax = axs.flat[k]
    ax.imshow(tone(img).numpy())
    ax.set_title(f"seed={seed} {style} / {gtn}", fontsize=8)
    ax.axis("off")
fig.suptitle("road diagnostic (sun el=25, az=40)", fontsize=10)
fig.tight_layout()
out = "outputs/sun_control/road_check_v7.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"[saved] {out}")
