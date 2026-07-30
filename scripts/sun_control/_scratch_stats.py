"""Measure fixed-gauge (p99 lum = 1) statistics of real RENI_HDR EXRs and of
the synthetic generator, and print a real vs v7 vs v8 comparison table."""
import glob
import sys

import numpy as np
import pyexr

import scripts.sun_control.synthetic_sky as v8
import scripts.sun_control._v7_snapshot as v7

LUM = np.array([0.2126, 0.7152, 0.0722])


def gauge(img):
    img = np.nan_to_num(img, nan=0.0, posinf=1e4, neginf=0.0)
    img = np.clip(img, 0, None)
    lum = (img * LUM).sum(-1)
    return img * (1.0 / max(np.quantile(lum, 0.99), 1e-8))


def _ratio(a, b):
    return float(a) / max(float(b), 1e-4)


def img_stats(img):
    """Per-image fixed-gauge stats. Sky = top half (elev>0), ground = bottom."""
    H = img.shape[0]
    img = np.nan_to_num(img, nan=0.0, posinf=1e4, neginf=0.0)
    img = np.clip(img, 0, None)
    lum = (img * LUM).sum(-1)
    sky, gnd = img[:H // 2].reshape(-1, 3), img[H // 2:].reshape(-1, 3)
    slum, glum = lum[:H // 2].ravel(), lum[H // 2:].ravel()

    def chroma(a):
        s = a.sum(-1, keepdims=True) + 1e-6
        return (a / s).mean(0)                       # mean [r,g,b] chromaticity

    return dict(
        sky_p50=np.quantile(slum, 0.5), sky_p90=np.quantile(slum, 0.9),
        gnd_p50=np.quantile(glum, 0.5), gnd_p10=np.quantile(glum, 0.1),
        gnd_p90=np.quantile(glum, 0.9),
        gnd_over_sky=_ratio(np.quantile(glum, 0.5), np.quantile(slum, 0.5)),
        peak=_ratio(np.quantile(lum, 0.9999), np.quantile(lum, 0.99)),
        sky_rb=_ratio(sky[:, 0].mean(), sky[:, 2].mean()),
        gnd_chroma_r=chroma(gnd)[0], gnd_chroma_g=chroma(gnd)[1],
        gnd_chroma_b=chroma(gnd)[2],
        wb_rb=_ratio(img[..., 0].mean(), img[..., 2].mean()),
    )


def aggregate(stats_list):
    keys = stats_list[0].keys()
    out = {}
    for k in keys:
        vals = np.array([s[k] for s in stats_list])
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def real_stats(n=400):
    paths = sorted(glob.glob("/workspace/phd/data/RENI_HDR/train/*.exr"))
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    out = []
    for i in idx:
        img = pyexr.read(paths[i])[..., :3].astype(np.float32)
        out.append(img_stats(gauge(img)))
    return aggregate(out)


def synth_stats(mod, n=400, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    dirs = mod.erp_directions(64, 128)
    out = []
    for _ in range(n):
        p = mod.sample_params(rng)
        img, _ = mod.render_sky(64, 128, dirs=dirs, nuisance_rng=rng, **p)
        img = mod.apply_gauge(img).numpy()
        out.append(img_stats(img))
    return aggregate(out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cols = []
    real = real_stats()
    cols.append(("real", real))
    if which in ("all", "v7"):
        cols.append(("v7", synth_stats(v7)))
    if which in ("all", "v8"):
        cols.append(("v8", synth_stats(v8)))

    keys = list(real.keys())
    hdr = f"{'stat':<16}" + "".join(f"{name:>22}" for name, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        row = f"{k:<16}"
        for _, d in cols:
            m, s = d[k]
            row += f"{m:>12.3f}+-{s:<8.3f}"
        print(row)


if __name__ == "__main__":
    main()
