"""Verify the group-geometry invariant at K=4, measure render time, and render
a sun-sweep strip plus a cast-shadow diagnostic (fixed scene, sun swept)."""
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.sun_control.synthetic_sky as ss

H, W = 64, 128
dirs = ss.erp_directions(H, W)
OUT = "outputs/sun_control"


def tone(img):
    img = ss.apply_gauge(img)
    return (img / torch.quantile(img.flatten(), 0.98)).clamp(0, 1) ** (1 / 2.2)


# ── 1. render-time measurement ──
rng = np.random.default_rng(0)
ps = [ss.sample_params(rng) for _ in range(60)]
seeds = [int(rng.integers(0, 2**31)) for _ in range(60)]
t0 = time.perf_counter()
for p, s in zip(ps, seeds):
    ss.render_sky(H, W, dirs=dirs, nuisance_rng=np.random.default_rng(s), **p)
dt = (time.perf_counter() - t0) / len(ps)
print(f"[time] {dt * 1000:.1f} ms/image at {H}x{W}")


# ── 2. group-geometry invariant at K=4 (incl. a twilight member) ──
suns = [
    dict(sun_elevation_deg=4.0, sun_azimuth_deg=-120.0, turbidity=3.0,
         sun_intensity=120.0, sun_sharpness=500.0),
    dict(sun_elevation_deg=72.0, sun_azimuth_deg=110.0, turbidity=6.0,
         sun_intensity=300.0, sun_sharpness=250.0),
    dict(sun_elevation_deg=30.0, sun_azimuth_deg=15.0, turbidity=4.0,
         sun_intensity=200.0, sun_sharpness=400.0),
    dict(sun_elevation_deg=-9.0, sun_azimuth_deg=-60.0, turbidity=5.0,
         sun_intensity=80.0, sun_sharpness=350.0),        # twilight member
]
all_ok = True
for nseed in [11, 202, 3003, 40004, 500005, 61, 77, 888]:
    outs = [ss.render_sky(H, W, dirs=dirs,
                          nuisance_rng=np.random.default_rng(nseed),
                          return_scene=True, **s) for s in suns]
    m0 = outs[0][2]
    geom_ok = all(
        all(torch.equal(m0[k], outs[j][2][k])
            for k in ("sky_mask", "silhouette", "ground", "road"))
        and torch.allclose(m0["terrain"], outs[j][2]["terrain"])
        for j in range(1, 4))
    shading_differs = all(not torch.allclose(outs[0][0], outs[j][0], atol=1e-4)
                          for j in range(1, 4))
    all_ok &= geom_ok and shading_differs
    print(f"[invariant] seed={nseed:6d} K=4 geometry_identical={geom_ok} "
          f"shading_differs={shading_differs}")
print(f"[invariant] ALL GEOMETRY IDENTICAL ACROSS 4 SUNS, SHADING DIFFERS: {all_ok}")


# ── 3. group-consistency evidence: 4 members + mask XOR vs member a ──
nseed = 3003
outs = [ss.render_sky(H, W, dirs=dirs, nuisance_rng=np.random.default_rng(nseed),
                      return_scene=True, **s) for s in suns]
occ0 = (~outs[0][2]["sky_mask"] | outs[0][2]["silhouette"]).float()
fig, axs = plt.subplots(4, 1, figsize=(7, 8))
titles = ["member a  (el=4, az=-120)", "member b  (el=72, az=110)",
          "member c  (el=30, az=15)", "member d  (el=-9 twilight, az=-60)"]
for j in range(4):
    occj = (~outs[j][2]["sky_mask"] | outs[j][2]["silhouette"]).float()
    xor = float((occ0 - occj).abs().max())
    axs[j].imshow(tone(outs[j][0]).numpy())
    axs[j].set_title(f"{titles[j]}   mask XOR vs a = {xor:.3f}", fontsize=9)
    axs[j].axis("off")
fig.suptitle("K=4 group: SAME nuisance seed, 4 suns -> identical geometry, "
             "different shading", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/group_consistency_v8.png", dpi=130, bbox_inches="tight")
print(f"[saved] {OUT}/group_consistency_v8.png")


# ── 4. sun-sweep strip: fixed scenery, sweep the sun (incl. twilight) ──
nseed = 202
cols = 8
fig, axs = plt.subplots(2, cols, figsize=(3 * cols, 3.2))
for j, az in enumerate(np.linspace(-180, 180, cols, endpoint=False)):
    p = dict(sun_elevation_deg=10.0, sun_azimuth_deg=float(az), turbidity=4.0,
             sun_intensity=120.0, sun_sharpness=400.0)
    img, _ = ss.render_sky(H, W, dirs=dirs,
                           nuisance_rng=np.random.default_rng(nseed), **p)
    axs[0, j].imshow(tone(img).numpy()); axs[0, j].axis("off")
    axs[0, j].set_title(f"az={az:.0f}", fontsize=8)
for j, el in enumerate(np.linspace(-10, 80, cols)):
    p = dict(sun_elevation_deg=float(el), sun_azimuth_deg=30.0, turbidity=4.0,
             sun_intensity=120.0, sun_sharpness=400.0)
    img, _ = ss.render_sky(H, W, dirs=dirs,
                           nuisance_rng=np.random.default_rng(nseed), **p)
    axs[1, j].imshow(tone(img).numpy()); axs[1, j].axis("off")
    axs[1, j].set_title(f"el={el:.0f}", fontsize=8)
fig.suptitle("fixed scenery (seed=202): top = azimuth sweep (el=10), "
             "bottom = elevation sweep incl. twilight", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/sun_sweep_v8.png", dpi=120, bbox_inches="tight")
print(f"[saved] {OUT}/sun_sweep_v8.png")


# ── 5. cast-shadow diagnostic: find a scene with buildings, sweep sun az ──
pick = None
for seed in range(600):
    sc = ss._sample_scene(np.random.default_rng(seed))
    if (len(sc["builds"]) >= 2 and not sc["road"] and not sc["mountain"]
            and sc["ground_type"].startswith("grass")
            and sc["hill_mean"] < 0.03):
        pick = seed
        break
cols = 7
el_diag = 12.0
fig, axs = plt.subplots(1, cols, figsize=(3 * cols, 2.4))
for j, az in enumerate(np.linspace(-150, 150, cols)):
    p = dict(sun_elevation_deg=el_diag, sun_azimuth_deg=float(az), turbidity=2.2,
             sun_intensity=150.0, sun_sharpness=500.0)
    img, _ = ss.render_sky(H, W, dirs=dirs,
                           nuisance_rng=np.random.default_rng(pick), **p)
    axs[j].imshow(tone(img).numpy()); axs[j].axis("off")
    axs[j].set_title(f"sun az={az:.0f}", fontsize=8)
fig.suptitle(f"cast-shadow diagnostic (seed={pick}, el={el_diag:.0f}): shadows "
             "fall on the anti-sun side and track the sun azimuth", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/shadow_diagnostic_v8.png", dpi=130, bbox_inches="tight")
print(f"[saved] {OUT}/shadow_diagnostic_v8.png  (seed={pick})")
