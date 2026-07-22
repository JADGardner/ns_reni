"""Zero-training probe: does any RENI++ latent channel already track the sun?

For every training environment, compute the sun direction as the
luminance-weighted centroid of the top-quantile pixels (fixed gauge), then
measure per-channel alignment between the fitted latent vectors z_ik and the
sun direction across the training set. Reports full-3D cosine alignment,
azimuth-only circular statistics, and norm-intensity correlation.

    PYTHONPATH=.:scripts/figures python scripts/sun_control/probe_sun_channel.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "scripts/figures")
from _common import MODEL_DIRS, REPO_ROOT, equirect_ray_bundle, load_model
from eval_outpaint_compare import EPS
from reni.utils.tonemap import apply_fixed_gauge, luminance

H, W = 64, 128
TOP_Q = 0.9995          # top 0.05% of pixels defines the peak region
DEVICE = "cuda:0"


def sun_direction(path: str, dirs: torch.Tensor) -> tuple[np.ndarray, float]:
    import pyexr

    img = pyexr.read(path).astype("float32")[..., :3]
    finite = np.isfinite(img)
    if not finite.all():
        img[~finite] = img[finite].max()
    img[img <= 0] = img[img > 0].min()
    t = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None],
                      size=(H, W), mode="bilinear")[0].permute(1, 2, 0)
    g = apply_fixed_gauge(t.to(DEVICE))
    lum = luminance(g).reshape(-1)
    # solid-angle weight (sin of polar angle) from the direction grid's y
    thresh = torch.quantile(lum, TOP_Q)
    sel = lum >= thresh
    w = lum[sel]
    d = (dirs[sel] * w[:, None]).sum(0)
    d = F.normalize(d, dim=0)
    peak_ratio = float(lum.max())         # gauge units (p99 lum = 1)
    return d.cpu().numpy(), peak_ratio


def main() -> None:
    _, _, model = load_model(MODEL_DIRS["vnjoint_ortho_2cyc"][100], device=DEVICE)
    bank = model.field.train_mu.detach().cpu().numpy()      # [N, 100, 3]
    n_bank = bank.shape[0]
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / "train" / "*.exr")))
    print(f"[probe] bank {bank.shape}, {len(files)} train EXRs")

    rb = equirect_ray_bundle(DEVICE, idx=0, height=H)
    dirs = rb.directions

    suns, peaks = [], []
    for i, f in enumerate(files):
        d, p = sun_direction(f, dirs)
        suns.append(d); peaks.append(p)
        if (i + 1) % 400 == 0:
            print(f"  [suns] {i + 1}/{len(files)}")
    suns = np.stack(suns)                                   # [M, 3]
    peaks = np.asarray(peaks)

    # Bank-to-file correspondence hypotheses (mirror augmentation doubles N).
    m = len(files)
    hypotheses = {}
    if n_bank == m:
        hypotheses["identity"] = bank
    elif n_bank == 2 * m:
        hypotheses["block_first_half"] = bank[:m]
        hypotheses["block_second_half"] = bank[m:]
        hypotheses["interleaved_even"] = bank[0::2]
        hypotheses["interleaved_odd"] = bank[1::2]
    else:
        print(f"[warn] bank size {n_bank} vs {m} files; using first {m}")
        hypotheses["truncated"] = bank[:m]

    # concentrate on images with a distinct peak (sunny scenes)
    sunny = peaks > np.quantile(peaks, 0.5)
    print(f"[probe] {sunny.sum()} sunny images (peak above median "
          f"{np.quantile(peaks, 0.5):.1f} gauge units)")

    results = {}
    for name, zb in hypotheses.items():
        z = zb[sunny]                                       # [S, 100, 3]
        s = suns[sunny]                                     # [S, 3]
        zdir = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)
        cos3d = np.einsum("skc,sc->sk", zdir, s)            # [S, 100]
        mean_cos = cos3d.mean(0)
        # azimuth-only circular alignment (y is up in this direction grid?
        # derive up axis empirically: the direction with max |mean| component)
        up_axis = int(np.argmax(np.abs(dirs.mean(0).cpu().numpy())))
        keep = [i for i in range(3) if i != up_axis]
        z_az = zdir[..., keep]; s_az = s[:, keep]
        z_az = z_az / (np.linalg.norm(z_az, axis=-1, keepdims=True) + 1e-9)
        s_azn = s_az / (np.linalg.norm(s_az, axis=-1, keepdims=True) + 1e-9)
        cos_az = np.einsum("skc,sc->sk", z_az, s_azn)
        sin_az = (z_az[..., 0] * s_azn[:, None, 1]
                  - z_az[..., 1] * s_azn[:, None, 0])
        # circular resultant length of the azimuth offset per channel
        R = np.sqrt(cos_az.mean(0) ** 2 + sin_az.mean(0) ** 2)
        # norm-vs-peak-intensity correlation
        norms = np.linalg.norm(z, axis=-1)
        pk = peaks[sunny]
        nc = np.array([np.corrcoef(norms[:, k], pk)[0, 1] for k in range(norms.shape[1])])
        order = np.argsort(-R)
        results[name] = {
            "top_by_R": [(int(k), float(R[k]), float(mean_cos[k]), float(nc[k]))
                         for k in order[:5]],
            "best_R": float(R.max()),
            "median_R": float(np.median(R)),
        }
        print(f"\n[{name}] azimuth resultant length: best {R.max():.3f}, "
              f"median {np.median(R):.3f}")
        print("  top channels (k, R_azimuth, mean 3D cos, norm-peak corr):")
        for k in order[:5]:
            print(f"    ch{int(k):3d}  R={R[k]:.3f}  cos3d={mean_cos[k]:+.3f}  "
                  f"normcorr={nc[k]:+.3f}")

    out = REPO_ROOT / "outputs" / "sun_control_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
