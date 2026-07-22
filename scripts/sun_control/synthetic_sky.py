"""Analytic synthetic-sky generator for sun-channel supervision experiments.

Pure Python/torch. Each sample is a Preetham clear-sky radiance distribution
(luminance + chromaticity as functions of sun position and turbidity), a
Gaussian sun lobe with elevation-dependent colour and intensity, and a simple
Lambertian-ish ground band lit by the sky. Output is linear RGB on the ERP
grid in the fixed gauge (p99 luminance = 1), so samples drop directly into
the RENI_HDR training pipeline, with ground-truth sun directions saved as a
sidecar JSON for latent-channel supervision.

    # visual check
    python scripts/sun_control/synthetic_sky.py --preview 12 \
        --output outputs/sun_control/preview

    # dataset emission (EXRs + sun_labels.json)
    python scripts/sun_control/synthetic_sky.py --emit 2000 \
        --output data/RENI_SUN_SYNTH/train
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

# ── Preetham coefficient tables ─────────────────────────────────────────

def _perez_coeffs(T: float) -> dict[str, np.ndarray]:
    return {
        "Y": np.array([0.1787 * T - 1.4630, -0.3554 * T + 0.4275,
                       -0.0227 * T + 5.3251, 0.1206 * T - 2.5771,
                       -0.0670 * T + 0.3703]),
        "x": np.array([-0.0193 * T - 0.2592, -0.0665 * T + 0.0008,
                       -0.0004 * T + 0.2125, -0.0641 * T - 0.8989,
                       -0.0033 * T + 0.0452]),
        "y": np.array([-0.0167 * T - 0.2608, -0.0950 * T + 0.0092,
                       -0.0079 * T + 0.2102, -0.0441 * T - 1.6537,
                       -0.0109 * T + 0.0529]),
    }


_MX = np.array([
    [0.00166, -0.00375, 0.00209, 0.0],
    [-0.02903, 0.06377, -0.03202, 0.00394],
    [0.11693, -0.21196, 0.06052, 0.25886],
])
_MY = np.array([
    [0.00275, -0.00610, 0.00317, 0.0],
    [-0.04214, 0.08970, -0.04153, 0.00516],
    [0.15346, -0.26756, 0.06670, 0.26688],
])


def _zenith(T: float, theta_s: float) -> tuple[float, float, float]:
    chi = (4.0 / 9.0 - T / 120.0) * (math.pi - 2.0 * theta_s)
    Yz = (4.0453 * T - 4.9710) * math.tan(chi) - 0.2155 * T + 2.4192  # kcd/m2
    Yz = max(Yz, 1e-3)
    tv = np.array([T * T, T, 1.0])
    sv = np.array([theta_s ** 3, theta_s ** 2, theta_s, 1.0])
    xz = float(tv @ _MX @ sv)
    yz = float(tv @ _MY @ sv)
    return Yz, xz, yz


def _perez(theta: torch.Tensor, gamma: torch.Tensor, c: np.ndarray) -> torch.Tensor:
    A, B, C, D, E = [float(v) for v in c]
    cos_t = torch.clamp(torch.cos(theta), min=1e-2)
    return ((1.0 + A * torch.exp(B / cos_t))
            * (1.0 + C * torch.exp(D * gamma) + E * torch.cos(gamma) ** 2))


def _xyY_to_linear_rgb(x: torch.Tensor, y: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, min=1e-4)
    X = x / y * Y
    Z = (1.0 - x - y) / y * Y
    XYZ = torch.stack([X, Y, Z], dim=-1)
    M = torch.tensor([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [0.0557, -0.2040, 1.0570]], dtype=XYZ.dtype)
    rgb = torch.einsum("...c,rc->...r", XYZ, M)
    return torch.clamp(rgb, min=0.0)


# ── ERP grid (rows: up pole -> down pole; azimuth along columns) ─────────

def erp_directions(height: int, width: int) -> torch.Tensor:
    """[H, W, 3] unit directions, y-up, matching the RENI_HDR ERP layout
    (top row = zenith)."""
    v = (torch.arange(height) + 0.5) / height          # 0 top -> 1 bottom
    u = (torch.arange(width) + 0.5) / width
    polar = v * math.pi                                # 0 = up pole
    azim = u * 2.0 * math.pi - math.pi
    pol, az = torch.meshgrid(polar, azim, indexing="ij")
    return torch.stack([
        torch.sin(pol) * torch.sin(az),
        torch.cos(pol),
        torch.sin(pol) * torch.cos(az),
    ], dim=-1)


def sun_colour(elevation_rad: float) -> torch.Tensor:
    """Warm-at-horizon linear RGB sun tint (crude but monotone)."""
    t = min(max(elevation_rad / math.radians(40.0), 0.0), 1.0)
    warm = torch.tensor([1.0, 0.55, 0.25])
    white = torch.tensor([1.0, 0.98, 0.92])
    return warm + (white - warm) * t


def render_sky(
    height: int = 64,
    width: int = 128,
    sun_elevation_deg: float = 25.0,
    sun_azimuth_deg: float = 0.0,
    turbidity: float = 3.0,
    sun_intensity: float = 40.0,
    sun_sharpness: float = 350.0,
    ground_albedo: tuple[float, float, float] = (0.22, 0.18, 0.14),
    dirs: torch.Tensor | None = None,
    nuisance_rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (linear RGB [H, W, 3], unit sun direction [3])."""
    if dirs is None:
        dirs = erp_directions(height, width)
    up = torch.tensor([0.0, 1.0, 0.0])
    el = math.radians(sun_elevation_deg)
    az = math.radians(sun_azimuth_deg)
    sun_d = torch.tensor([
        math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)
    ])

    cos_theta = dirs @ up                              # view zenith cosine
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    cos_gamma = torch.clamp(dirs @ sun_d, -1.0, 1.0)
    gamma = torch.acos(cos_gamma)
    theta_s = math.pi / 2.0 - el

    c = _perez_coeffs(turbidity)
    Yz, xz, yz = _zenith(turbidity, theta_s)
    g0 = torch.tensor(theta_s)

    def channel(cc: np.ndarray, zenith_val: float) -> torch.Tensor:
        num = _perez(theta, gamma, cc)
        den = _perez(torch.tensor(0.0), g0, cc)
        return zenith_val * num / den

    Y = channel(c["Y"], Yz)
    x = channel(c["x"], xz)
    y = channel(c["y"], yz)
    sky = _xyY_to_linear_rgb(x, y, Y)

    # Sun lobe (added in linear; sharpness ~ inverse angular width).
    lobe = torch.exp(sun_sharpness * (cos_gamma - 1.0))
    sky = sky + sun_intensity * Yz * lobe[..., None] * sun_colour(el)

    # Terrain: a band-limited random height profile along azimuth, so the
    # horizon rolls with hills instead of being perfectly flat.
    elev = torch.asin(torch.clamp(dirs @ up, -1.0, 1.0))
    azim = torch.atan2(dirs[..., 0], dirs[..., 2])
    if nuisance_rng is not None:
        r = nuisance_rng
        h_az = torch.zeros_like(azim)
        for k in range(1, 5):
            amp = math.radians(r.uniform(0.5, 4.5) / k)
            phase = r.uniform(0.0, 2.0 * math.pi)
            h_az = h_az + amp * torch.sin(k * azim + phase)
        h_az = h_az + math.radians(r.uniform(0.5, 3.0))   # mean hill height
    else:
        h_az = torch.zeros_like(azim)

    # Ground: albedo-tinted diffuse estimate from mean sky radiance, with a
    # horizon-brightening blend and a sun-facing bounce gradient. Hills use
    # the same shading with a slight atmospheric lift by height.
    sky_mask = elev > h_az
    mean_sky = sky[elev > 0.15].mean(0)
    albedo = torch.tensor(ground_albedo)
    horiz = torch.clamp(1.0 - torch.abs(elev - h_az) / 0.6, 0.0, 1.0) ** 4
    bounce = torch.clamp(dirs @ torch.tensor(
        [sun_d[0], 0.0, sun_d[2]]).to(dirs.dtype), min=0.0)
    atmo = torch.clamp(elev / 0.25, 0.0, 1.0)
    ground = (albedo * mean_sky)[None, None] * (
        0.55 + 0.25 * bounce[..., None] + 0.6 * horiz[..., None])
    ground = ground * (1.0 - 0.35 * atmo[..., None])         + 0.18 * atmo[..., None] * mean_sky
    img = torch.where(sky_mask[..., None], sky, ground)

    # Soft terrain-line transition to avoid a hard seam.
    blend = torch.clamp((elev - h_az) / 0.02, 0.0, 1.0)[..., None]
    img = blend * sky + (1.0 - blend) * torch.where(
        sky_mask[..., None], sky * 0.5 + ground * 0.5, ground)

    # Azimuthal nuisance content decorrelated from the sun, so that in
    # training the supervised channel is the only reliable sun cue:
    # cloud blobs in the sky and a skyline silhouette above the horizon.
    if nuisance_rng is not None:
        r = nuisance_rng
        for _ in range(int(r.integers(2, 6))):        # clouds
            caz = math.radians(r.uniform(-180, 180))
            cel = math.radians(r.uniform(8, 55))
            cd = torch.tensor([math.cos(cel) * math.sin(caz), math.sin(cel),
                               math.cos(cel) * math.cos(caz)])
            cg = torch.clamp(dirs @ cd, -1.0, 1.0)
            width = r.uniform(15.0, 70.0)
            shade = r.uniform(0.55, 1.9)
            cloud = torch.exp(width * (cg - 1.0))
            img = img * (1.0 + (shade - 1.0) * cloud[..., None]
                         * sky_mask[..., None].float())
        n_build = int(r.integers(2, 7))               # skyline
        sky_mean = img[sky_mask].mean(0)
        for _ in range(n_build):
            baz = r.uniform(-math.pi, math.pi)
            half_w = r.uniform(0.06, 0.35)
            h_el = math.radians(r.uniform(3.0, 18.0))
            daz = torch.remainder(azim - baz + math.pi, 2 * math.pi) - math.pi
            in_sil = (daz.abs() < half_w) & (elev < h_az + h_el) & (elev > -0.02)
            tint = torch.tensor([r.uniform(0.02, 0.10)] * 3) * sky_mean
            img = torch.where(in_sil[..., None], tint.expand_as(img), img)

    return img, sun_d


def apply_gauge(img: torch.Tensor) -> torch.Tensor:
    lum = (img * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    scale = 1.0 / torch.quantile(lum.flatten(), 0.99).clamp(min=1e-8)
    return img * scale


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "sun_elevation_deg": float(np.degrees(np.arcsin(rng.uniform(
            np.sin(np.radians(3.0)), np.sin(np.radians(65.0)))))),
        "sun_azimuth_deg": float(rng.uniform(-180.0, 180.0)),
        "turbidity": float(rng.uniform(2.0, 8.0)),
        "sun_intensity": float(np.exp(rng.uniform(np.log(5.0), np.log(120.0)))),
        "sun_sharpness": float(np.exp(rng.uniform(np.log(120.0), np.log(900.0)))),
        "ground_albedo": tuple(float(v) for v in rng.uniform(
            0.05, 0.35, 3) * np.array([1.0, 0.9, 0.75])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--preview", type=int, default=0,
                        help="Render N samples into a PNG grid.")
    parser.add_argument("--emit", type=int, default=0,
                        help="Emit N EXRs plus sun_labels.json.")
    parser.add_argument("--emit-pairs", type=int, default=0,
                        help="Emit N counterfactual pairs (2N EXRs): identical "
                             "nuisance content, independent sun position.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    dirs = erp_directions(args.height, args.width)

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = args.preview
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows))
        for i in range(rows * cols):
            ax = axs.flat[i]
            ax.axis("off")
            if i >= n:
                continue
            p = sample_params(rng)
            img, sun_d = render_sky(args.height, args.width, dirs=dirs,
                                    nuisance_rng=rng, **p)
            img = apply_gauge(img)
            tone = (img / torch.quantile(img.flatten(), 0.98)).clamp(0, 1) ** (1 / 2.2)
            ax.imshow(tone.numpy())
            ax.set_title(f"el={p['sun_elevation_deg']:.0f}°  "
                         f"T={p['turbidity']:.1f}  I={p['sun_intensity']:.0f}",
                         fontsize=8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"{args.output}.png", dpi=130, bbox_inches="tight")
        print(f"[saved] {args.output}.png")

    if args.emit:
        import pyexr

        args.output.mkdir(parents=True, exist_ok=True)
        labels = {}
        for i in range(args.emit):
            p = sample_params(rng)
            img, sun_d = render_sky(args.height, args.width, dirs=dirs,
                                    nuisance_rng=rng, **p)
            img = apply_gauge(img)
            name = f"synth_{i:05d}.exr"
            pyexr.write(str(args.output / name), img.numpy())
            labels[name] = {"sun_direction": [float(v) for v in sun_d],
                            **{k: v for k, v in p.items()}}
            if (i + 1) % 250 == 0:
                print(f"  [emit] {i + 1}/{args.emit}")
        (args.output / "sun_labels.json").write_text(json.dumps(labels, indent=1))
        print(f"[saved] {args.emit} EXRs + sun_labels.json -> {args.output}")

    if args.emit_pairs:
        import pyexr

        args.output.mkdir(parents=True, exist_ok=True)
        labels = {}
        for i in range(args.emit_pairs):
            p = sample_params(rng)
            alt = sample_params(rng)          # independent second sun draw
            nseed = int(rng.integers(0, 2**31))
            for member, (el, az) in (("a", (p["sun_elevation_deg"],
                                            p["sun_azimuth_deg"])),
                                     ("b", (alt["sun_elevation_deg"],
                                            alt["sun_azimuth_deg"]))):
                q = dict(p)
                q["sun_elevation_deg"], q["sun_azimuth_deg"] = el, az
                img, sun_d = render_sky(
                    args.height, args.width, dirs=dirs,
                    nuisance_rng=np.random.default_rng(nseed), **q)
                img = apply_gauge(img)
                name = f"synth_p{i:04d}_{member}.exr"
                pyexr.write(str(args.output / name), img.numpy())
                labels[name] = {"sun_direction": [float(v) for v in sun_d],
                                "pair_id": i, "member": member,
                                **{k: v for k, v in q.items()}}
            if (i + 1) % 250 == 0:
                print(f"  [pairs] {i + 1}/{args.emit_pairs}")
        (args.output / "sun_labels.json").write_text(json.dumps(labels, indent=1))
        print(f"[saved] {args.emit_pairs} pairs + sun_labels.json -> {args.output}")


if __name__ == "__main__":
    main()
