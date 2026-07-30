"""Analytic synthetic-sky generator for sun-channel supervision experiments.

Pure Python/torch. Each sample is a Preetham clear-sky radiance distribution
(luminance + chromaticity as functions of sun position and turbidity), a
Gaussian sun lobe with elevation-dependent colour and intensity, and a
textured ground with band-limited terrain, roads, trees and buildings lit by
the sky and sun. Output is linear RGB on the ERP grid in the fixed gauge
(p99 luminance = 1), so samples drop directly into the RENI_HDR training
pipeline, with ground-truth sun directions saved as a sidecar JSON for
latent-channel supervision.

All scenery geometry is drawn from ``nuisance_rng`` up front (``_sample_scene``)
and is independent of the sun parameters, so counterfactual triplets that share
a nuisance seed have pixel-identical geometry and differ only in shading.

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


def _airmass(zenith_deg: float) -> float:
    """Kasten-Young relative optical airmass."""
    z = min(zenith_deg, 89.9)
    return 1.0 / (math.cos(math.radians(z))
                  + 0.50572 * (96.07995 - z) ** -1.6364)


def sun_transmittance(elevation_rad: float, turbidity: float) -> torch.Tensor:
    """Beer-Lambert direct-sun transmittance (Rayleigh + turbidity-scaled
    aerosol) along the airmass path: the disc dims and reddens as it drops
    toward the horizon, more so in hazy (high-turbidity) skies."""
    m = _airmass(90.0 - math.degrees(elevation_rad))
    tau_ray = torch.tensor([0.050, 0.098, 0.218])       # ~650/550/450 nm
    beta = max(0.046 * turbidity - 0.046, 0.0)          # Angstrom turbidity
    tau_aer = beta * torch.tensor([0.65, 0.55, 0.45]) ** -1.3
    s0 = torch.tensor([1.0, 0.99, 0.96])                # solar tint
    return s0 * torch.exp(-m * (tau_ray + tau_aer))


# ── scenery sampling (ALL nuisance draws happen here, up front and in a
#    fixed order that depends only on the nuisance seed, never on the sun,
#    so triplet members share pixel-identical geometry) ────────────────────

def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


# Base linear reflectance, texture amplitude, glossiness (glitter strength),
# a colour-variation tint, and a sampling weight per ground type. ``green``
# marks types that keep more of their own chroma against the blue sky fill so
# grass reads as grass rather than desaturating to grey.
_GROUND_TYPES = {
    "grass":   dict(alb=(0.110, 0.150, 0.050), tex=0.55, gloss=0.02,
                    var=(0.12, 0.15, 0.05), green=0.35, wt=0.24),
    "grass_lush": dict(alb=(0.060, 0.185, 0.035), tex=0.50, gloss=0.03,
                       var=(0.06, 0.18, 0.04), green=0.55, wt=0.13),
    "grass_dry": dict(alb=(0.170, 0.150, 0.065), tex=0.60, gloss=0.02,
                      var=(0.15, 0.12, 0.05), green=0.25, wt=0.12),
    "dirt":    dict(alb=(0.180, 0.115, 0.072), tex=0.50, gloss=0.03,
                    var=(0.12, 0.07, 0.05), green=0.0, wt=0.15),
    "snow":    dict(alb=(0.72, 0.740, 0.820), tex=0.14, gloss=0.60,
                    var=(0.05, 0.05, 0.07), green=0.0, wt=0.06),
    "asphalt": dict(alb=(0.078, 0.076, 0.082), tex=0.28, gloss=0.28,
                    var=(0.03, 0.03, 0.04), green=0.0, wt=0.09),
    "sand":    dict(alb=(0.310, 0.250, 0.170), tex=0.45, gloss=0.05,
                    var=(0.11, 0.08, 0.05), green=0.0, wt=0.15),
}
_GROUND_NAMES = list(_GROUND_TYPES)
_GROUND_WEIGHTS = np.array([_GROUND_TYPES[n]["wt"] for n in _GROUND_NAMES])
_GROUND_WEIGHTS = _GROUND_WEIGHTS / _GROUND_WEIGHTS.sum()


def _sample_scene(r: np.random.Generator) -> dict:
    """Draw every scenery choice from the nuisance RNG, up front, so the
    scene geometry is a pure function of the nuisance seed."""
    s: dict = {}

    # Terrain silhouette: mean height + a few azimuthal octaves, wider
    # amplitude range than before, with an occasional mountain ridge.
    hill_scale = float(np.exp(r.uniform(np.log(0.6), np.log(3.5))))
    s["hill_mean"] = math.radians(float(r.uniform(-1.5, 4.0)))
    s["octaves"] = [(k,
                     math.radians(float(r.uniform(0.4, 6.5)) * hill_scale / k),
                     float(r.uniform(0.0, 2.0 * math.pi)))
                    for k in range(1, 6)]
    s["mountain"] = bool(r.random() < 0.30)
    s["mtn_az"] = float(r.uniform(-math.pi, math.pi))
    s["mtn_sigma"] = float(r.uniform(0.4, 1.3))
    s["mtn_amp"] = math.radians(float(r.uniform(14.0, 45.0)))
    s["mtn_rough"] = [(int(r.integers(3, 10)),
                       float(r.uniform(0.05, 0.35)),
                       float(r.uniform(0.0, 2.0 * math.pi))) for _ in range(4)]

    # Ground type (grass-weighted), plus a bank of hash-based value-noise
    # octaves with per-image random lattice resolutions, offsets and values,
    # so the ground grain never tiles within an image and no two images share
    # a pattern.
    gt_name = _GROUND_NAMES[int(r.choice(len(_GROUND_NAMES), p=_GROUND_WEIGHTS))]
    s["ground_type"] = gt_name
    base_gu = int(r.integers(5, 11))
    base_gv = int(r.integers(4, 9))
    noise = []
    for o in range(4):
        gu = base_gu * (2 ** o)
        gv = base_gv * (2 ** o)
        grid = r.uniform(-1.0, 1.0, size=(gv + 1, gu)).astype(np.float32)
        noise.append((gu, gv, grid, 0.5 ** o, float(r.uniform(0.0, 1.0))))
    s["noise"] = noise

    # Road converging to a horizon vanishing point. Style is chosen to contrast
    # the ground: dark tarmac over bright/mid ground, a pale track over dark
    # ground. Crisp-edged, wide at the viewer, and shown in ~half of samples.
    s["road"] = bool(r.random() < 0.50)
    s["road_az"] = float(r.uniform(-math.pi, math.pi))
    s["road_w"] = float(r.uniform(1.6, 4.2))
    s["cam_h"] = float(r.uniform(2.4, 5.5))
    g_lum = float(np.dot(_GROUND_TYPES[gt_name]["alb"], (0.2126, 0.7152, 0.0722)))
    coin = float(r.random())
    if g_lum > 0.30:                                 # bright ground -> tarmac
        s["road_style"] = "tarmac"
    elif g_lum < 0.10:                               # dark ground -> pale track
        s["road_style"] = "pale"
    else:
        s["road_style"] = "tarmac" if coin < 0.6 else "pale"

    # Trees: clusters of overlapping lobes -> blobby dark silhouettes.
    n_tree = int(r.integers(0, 5))
    trees = []
    for _ in range(n_tree):
        caz = float(r.uniform(-math.pi, math.pi))
        span = float(r.uniform(0.05, 0.6))
        n_lobe = int(r.integers(3, 10))
        trees.append([(caz + float(r.uniform(-span, span)),
                       float(r.uniform(0.02, 0.16)),
                       math.radians(float(r.uniform(2.0, 15.0))))
                      for _ in range(n_lobe)])
    s["trees"] = trees
    s["tree_dark"] = float(r.uniform(0.04, 0.20))

    # Buildings: rectangular silhouettes, occasionally a taller city block.
    city = bool(r.random() < 0.25)
    n_build = int(r.integers(0, 4)) + (6 if city else 0)
    base_az = float(r.uniform(-math.pi, math.pi))
    builds = []
    for _ in range(n_build):
        baz = (base_az + float(r.uniform(-0.6, 0.6)) if city
               else float(r.uniform(-math.pi, math.pi)))
        builds.append((baz,
                       float(r.uniform(0.05, 0.35)) * (1.6 if city else 1.0),
                       math.radians(float(r.uniform(3.0, 26.0))
                                    * (2.0 if city else 1.0)),
                       float(r.uniform(0.03, 0.14))))
    s["city"] = city
    s["builds"] = builds

    # Clouds.
    n_cloud = int(r.integers(1, 7))
    s["clouds"] = [(math.radians(float(r.uniform(-180.0, 180.0))),
                    math.radians(float(r.uniform(6.0, 60.0))),
                    float(r.uniform(12.0, 75.0)),
                    float(r.uniform(0.5, 1.9))) for _ in range(n_cloud)]

    # Global white-balance jitter (survives the fixed gauge as a colour shift)
    # matched to the real colour-temperature spread, plus a ground-only exposure
    # jitter (a truly uniform exposure would be cancelled by the p99 gauge).
    ct = float(r.uniform(-0.25, 0.60))               # slightly warm-biased
    s["wb"] = (1.0 + 0.16 * ct, 1.0, 1.0 - 0.16 * ct)
    s["ground_exp"] = float(np.exp(r.uniform(np.log(0.55), np.log(1.5))))
    return s


def _default_scene() -> dict:
    """Flat, silhouette-free grass scene for the deterministic fallback."""
    return dict(hill_mean=math.radians(1.0), octaves=[], mountain=False,
                mtn_az=0.0, mtn_sigma=1.0, mtn_amp=0.0, mtn_rough=[],
                ground_type="grass", noise=[], road=False, road_az=0.0,
                road_w=2.0, cam_h=2.0, road_style="tarmac", trees=[],
                tree_dark=0.12, city=False, builds=[], clouds=[],
                wb=(1.0, 1.0, 1.0), ground_exp=1.0)


def _value_noise(u: torch.Tensor, v: torch.Tensor, octaves) -> torch.Tensor:
    """Multi-octave hash-based value noise in roughly [-1, 1]. ``u`` is a
    periodic coordinate in [0, 1) (azimuth), ``v`` a coordinate in [0, 1]
    (perspective distance). Each octave samples a per-image random lattice with
    smoothstep bilinear interpolation and wraps in ``u`` for a seamless seam."""
    out = torch.zeros_like(u)
    norm = 1e-6
    for gu, gv, grid, amp, off in octaves:
        g = torch.from_numpy(grid)                      # (gv + 1, gu)
        pu = ((u + off) % 1.0) * gu
        i0 = torch.floor(pu).long() % gu
        i1 = (i0 + 1) % gu
        fu = pu - torch.floor(pu)
        pv = torch.clamp(v, 0.0, 1.0) * gv
        j0 = torch.clamp(torch.floor(pv).long(), 0, gv - 1)
        j1 = j0 + 1
        fv = pv - j0.to(pv.dtype)
        fu = fu * fu * (3.0 - 2.0 * fu)                 # smoothstep
        fv = fv * fv * (3.0 - 2.0 * fv)
        top = g[j0, i0] + (g[j0, i1] - g[j0, i0]) * fu
        bot = g[j1, i0] + (g[j1, i1] - g[j1, i0]) * fu
        out = out + amp * (top + (bot - top) * fv)
        norm += amp
    return out / norm


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
    return_scene: bool = False,
):
    """Return (linear RGB [H, W, 3], unit sun direction [3]).

    Scenery geometry comes entirely from ``nuisance_rng`` (see
    ``_sample_scene``) and is independent of the sun parameters; only the
    shading of that geometry depends on the sun. With ``return_scene`` the
    geometry masks are returned as a third value for invariant checks."""
    if dirs is None:
        dirs = erp_directions(height, width)
    up = torch.tensor([0.0, 1.0, 0.0])
    el = math.radians(sun_elevation_deg)
    az = math.radians(sun_azimuth_deg)
    sun_d = torch.tensor([
        math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)
    ])

    # Twilight band: sun below the horizon but strictly above -12 deg, kept well
    # clear of the parked -45 deg label so nothing there gains meaning. No disc
    # appears, the sky dims with depth below the horizon, and a warm glow hugs
    # the horizon at the sun azimuth. The Perez sky is evaluated with the sun
    # pinned just above the horizon for numerical stability.
    twilight = el < 0.0
    tw = min(max(-sun_elevation_deg / 12.0, 0.0), 1.0)   # 0 at horizon .. 1 at -12
    el_eff = max(el, math.radians(0.5))
    sky_sun_d = torch.tensor([
        math.cos(el_eff) * math.sin(az), math.sin(el_eff),
        math.cos(el_eff) * math.cos(az)])

    cos_theta = dirs @ up                              # view zenith cosine
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    cos_gamma = torch.clamp(dirs @ sky_sun_d, -1.0, 1.0)
    gamma = torch.acos(cos_gamma)
    theta_s = math.pi / 2.0 - el_eff

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

    elev = torch.asin(torch.clamp(dirs @ up, -1.0, 1.0))
    azim = torch.atan2(dirs[..., 0], dirs[..., 2])
    sun_col = sun_transmittance(el_eff, turbidity)

    # Sun lobe, gated off below the horizon so twilight shows no disc.
    disc_gate = min(max(sun_elevation_deg / 2.0, 0.0), 1.0)
    lobe = torch.exp(sun_sharpness * (cos_gamma - 1.0))
    sky = sky + disc_gate * sun_intensity * Yz * lobe[..., None] * sun_col

    if twilight:
        along = torch.clamp(dirs @ torch.tensor(
            [sky_sun_d[0], 0.0, sky_sun_d[2]]), -1.0, 1.0)
        warm = sun_col / sun_col.mean().clamp(min=1e-4)
        sky = sky * (1.0 - 0.90 * tw)                    # dim with depth
        sky = sky * (0.40 + 0.60 * torch.clamp(along, 0.0, 1.0))[..., None]
        glow = torch.exp(-(elev / 0.13) ** 2) * torch.clamp(along, 0.0, 1.0) ** 3
        sky = sky + (0.8 - 0.5 * tw) * Yz * warm * glow[..., None]

    scene = _sample_scene(nuisance_rng) if nuisance_rng is not None \
        else _default_scene()

    # ── terrain silhouette (sun-independent geometry) ──
    h_terr = torch.full_like(azim, scene["hill_mean"])
    for k, amp, phase in scene["octaves"]:
        h_terr = h_terr + amp * torch.sin(k * azim + phase)
    if scene["mountain"]:
        daz = _wrap(azim - scene["mtn_az"])
        ridge = scene["mtn_amp"] * torch.exp(
            -(daz ** 2) / (2.0 * scene["mtn_sigma"] ** 2))
        rough = torch.zeros_like(azim)
        for k, amp, phase in scene["mtn_rough"]:
            rough = rough + amp * torch.sin(k * azim + phase)
        h_terr = h_terr + ridge * (1.0 + rough)
    sky_mask = elev > h_terr

    # ── ground shading (sun-dependent lighting of sun-independent geometry) ──
    mean_sky = sky[elev > 0.15].mean(0)                # bluish sky ambient
    gt = _GROUND_TYPES[scene["ground_type"]]
    alb_scale = float(np.dot(ground_albedo, (0.2126, 0.7152, 0.0722))) / 0.18
    alb = torch.tensor(gt["alb"]) * max(alb_scale, 0.4)
    # Perspective-compressed value-noise grain: u periodic in azimuth, v a
    # viewer-distance coordinate so texture tightens toward the horizon.
    u_coord = (azim + math.pi) / (2.0 * math.pi)
    dist_g = 1.7 / torch.tan(torch.clamp(-elev, min=0.02))
    v_coord = 1.0 / (1.0 + dist_g / 6.0)
    tex = _value_noise(u_coord, v_coord, scene["noise"])
    tex_mult = torch.clamp(1.0 + gt["tex"] * tex, 0.2, 2.0)

    sun_up = max(math.sin(el), 0.0)                    # sun-above-horizon factor
    sun_az_d = torch.tensor([sun_d[0], 0.0, sun_d[2]])
    sun_az_d = sun_az_d / sun_az_d.norm().clamp(min=1e-4)
    bounce = torch.clamp(dirs @ sun_az_d, min=0.0)     # hotspot toward sun az
    warm = sun_col / sun_col.mean().clamp(min=1e-4)    # normalised warm tint
    low = 1.0 - sun_up                                 # low sun -> warmer ground

    # Ambient sky fill, desaturated toward grey for green grounds so grass keeps
    # its own chroma, then warmed slightly to match the measured warm cast of
    # real ground (chromaticity r > g > b on average).
    amb = mean_sky * (1.0 - gt["green"]) + mean_sky.mean() * gt["green"]
    amb = amb * torch.tensor([1.08, 1.00, 0.82])

    # Irradiance calibrated to the real RENI_HDR ground luminance percentiles
    # (fixed-gauge median ~0.07): a modest sky-fill floor keeps the foreground
    # off pure black while the sun term carries the green of sunlit grass.
    E = (0.90 * amb[None, None]
         + sun_up * sun_col[None, None] * (0.60 + 0.55 * bounce[..., None]))
    E = E * (1.0 + 0.60 * low * bounce[..., None] * (warm[None, None] - 1.0))
    ground = alb[None, None] * E * tex_mult[..., None]
    ground = ground + (torch.tensor(gt["var"])
                       * torch.clamp(tex, min=0.0)[..., None]) * 0.12 * amb
    haze = torch.exp(-torch.clamp(h_terr - elev, min=0.0) / 0.25)[..., None]
    ground = ground + 0.08 * haze * mean_sky[None, None]

    # Forward-scatter / glitter streak below the sun on glossy grounds: a tight
    # specular lobe at the mirror-sun direction plus a vertical streak along the
    # sun azimuth, both scaled by sun height and the ground glossiness.
    sun_mir = torch.tensor([sun_d[0], -sun_d[1], sun_d[2]])
    spec = torch.clamp(dirs @ sun_mir, min=0.0)
    daz_sun = _wrap(azim - az)
    az_band = torch.exp(-(daz_sun ** 2) / (2.0 * 0.10))
    depth = torch.clamp(-elev, min=0.0) / (abs(el) + 0.08)
    streak = az_band * torch.exp(-((depth - 1.0) ** 2) / (2.0 * 0.45 ** 2))
    glit = gt["gloss"] * (spec ** 60 + 0.35 * streak) * sun_up
    ground = ground + glit[..., None] * sun_col[None, None] * 1.1
    ground = ground * scene["ground_exp"]              # per-image ground exposure

    # ── sky/ground composite with a soft terrain edge ──
    edge = torch.clamp((elev - h_terr) / 0.015, 0.0, 1.0)[..., None]
    img = edge * sky + (1.0 - edge) * ground

    # ── road converging to a horizon vanishing point (sun-lit) ──
    # Flat ground seen from height h: a pixel at depression |e| hits the ground
    # at distance h/tan|e|, so a constant-width road along azimuth a0 shows where
    # the lateral offset d*sin(a-a0) is under the half-width, thinning to a
    # vanishing point at the horizon. Edges are crisp (small lateral feather),
    # the surface strongly contrasts the ground (dark tarmac or pale track), and
    # only the last fraction of a degree at the horizon fades, so it reads bold.
    road_m = torch.zeros_like(sky_mask)
    if scene["road"]:
        depr = torch.clamp(-elev, min=1e-3)
        dist = scene["cam_h"] / torch.tan(depr)
        lat = (dist * torch.sin(_wrap(azim - scene["road_az"]))).abs()
        road_m = (lat < scene["road_w"]) & (elev < -0.01)
        alpha = torch.clamp((scene["road_w"] - lat) / 0.6, 0.0, 1.0)   # crisp edge
        alpha = alpha * torch.clamp(-elev / 0.05, 0.0, 1.0)            # horizon fade
        rtex = (1.0 + 0.12 * tex)[..., None]                           # faint grain
        if scene["road_style"] == "tarmac":
            road_rad = torch.tensor([0.075, 0.075, 0.085])[None, None] * E * rtex
            road_rad = road_rad + glit[..., None] * sun_col[None, None] * 1.1
        else:                                            # pale dirt / gravel track
            road_rad = torch.tensor([0.42, 0.38, 0.31])[None, None] * E * rtex
        alpha = alpha[..., None]
        img = torch.where(road_m[..., None],
                          alpha * road_rad + (1.0 - alpha) * img, img)

    # ── stylised cast shadows (sun-dependent shading of nuisance geometry) ──
    # Each silhouette object darkens a soft ground wedge on the anti-sun side,
    # length ~ cot(elevation) so shadows stretch at low sun, leaning toward the
    # anti-sun azimuth with depth. Object positions/sizes come from the scene
    # only; shadows vanish in twilight and soften under high turbidity.
    if sun_up > 0.02 and not twilight:
        casters = []
        for lobes in scene["trees"]:
            azs = np.array([lo[0] for lo in lobes])
            d = np.arctan2(np.sin(azs - azs[0]), np.cos(azs - azs[0]))
            casters.append((float(azs[0] + d.mean()),
                            float(max(d.max() - d.min(), 0.0) / 2.0 + 0.04),
                            float(max(lo[2] for lo in lobes))))
        for baz, half_w, b_h, _dark in scene["builds"]:
            casters.append((float(baz), float(half_w), float(b_h)))

        cot = min(1.0 / max(math.tan(el), 0.08), 10.0)
        base_len = min(max(0.05 * cot, 0.03), 0.55)
        soft_turb = min(max(1.15 - 0.09 * turbidity, 0.35), 1.0)
        dep = torch.clamp(-elev, min=0.0)
        shade_acc = torch.zeros_like(azim)
        for baz, hw, height in casters:
            slen = base_len * min(1.0 + 4.0 * height, 3.0)
            dd = (az + math.pi) - baz
            lean = max(-0.6, min(0.6, math.atan2(math.sin(dd), math.cos(dd))))
            center = baz + lean * torch.clamp(dep / slen, 0.0, 1.0)
            across = torch.clamp(1.0 - _wrap(azim - center).abs() / (hw + 0.10),
                                 0.0, 1.0)
            along = torch.clamp(1.0 - dep / slen, 0.0, 1.0)
            shade_acc = torch.maximum(shade_acc, across * along)
        shade = 0.58 * soft_turb * shade_acc * (~sky_mask).float()
        img = img * (1.0 - shade)[..., None]

    # ── clouds (nuisance geometry, sky-only) ──
    for caz, cel, cwidth, cshade in scene["clouds"]:
        cd = torch.tensor([math.cos(cel) * math.sin(caz), math.sin(cel),
                           math.cos(cel) * math.cos(caz)])
        cg = torch.clamp(dirs @ cd, -1.0, 1.0)
        cloud = torch.exp(cwidth * (cg - 1.0))
        img = img * (1.0 + (cshade - 1.0) * cloud[..., None]
                     * sky_mask[..., None].float())

    # ── tree silhouettes: blobby dark lobes above the terrain line ──
    silh = torch.zeros_like(sky_mask)
    sky_lum = mean_sky.mean().clamp(min=1e-4)
    tree_col = sky_lum * scene["tree_dark"] * torch.tensor([0.55, 0.70, 0.40])
    rim = torch.clamp(bounce, 0.0, 1.0)[..., None] * sun_up   # sun-lit rim
    for lobes in scene["trees"]:
        top = torch.zeros_like(azim)
        for laz, sig, amp in lobes:
            top = torch.maximum(top, amp * torch.exp(
                -(_wrap(azim - laz) ** 2) / (2.0 * sig ** 2)))
        tmask = sky_mask & (elev < h_terr + top) & (top > math.radians(0.3))
        tcol = tree_col[None, None] * (0.6 + 0.7 * rim)
        img = torch.where(tmask[..., None], tcol.expand_as(img), img)
        silh = silh | tmask

    # ── building silhouettes: rectangles with a sun-lit face ──
    face = torch.clamp(dirs @ sun_az_d, min=0.0)[..., None] * sun_up
    for baz, half_w, b_h, dark in scene["builds"]:
        bmask = sky_mask & (_wrap(azim - baz).abs() < half_w) & (elev < h_terr + b_h)
        bcol = (torch.tensor([dark, dark, dark * 1.05])[None, None] * mean_sky) \
            * (0.7 + 0.9 * face)
        img = torch.where(bmask[..., None], bcol.expand_as(img), img)
        silh = silh | bmask

    # Global white-balance jitter (survives the fixed gauge as a colour shift).
    img = img * torch.tensor(scene["wb"])
    img = torch.clamp(img, min=0.0)
    if return_scene:
        return img, sun_d, {"sky_mask": sky_mask, "silhouette": silh,
                            "ground": ~sky_mask, "road": road_m,
                            "terrain": h_terr}
    return img, sun_d


def apply_gauge(img: torch.Tensor) -> torch.Tensor:
    lum = (img * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    scale = 1.0 / torch.quantile(lum.flatten(), 0.99).clamp(min=1e-8)
    return img * scale


def _el_az_to_dir(el_deg: float, az_deg: float) -> torch.Tensor:
    el, az = math.radians(el_deg), math.radians(az_deg)
    return torch.tensor([math.cos(el) * math.sin(az), math.sin(el),
                         math.cos(el) * math.cos(az)])


def _dir_to_el_az(d: torch.Tensor) -> tuple[float, float]:
    return (math.degrees(math.asin(max(-1.0, min(1.0, float(d[1]))))),
            math.degrees(math.atan2(float(d[0]), float(d[2]))))


def _slerp(d1: torch.Tensor, d2: torch.Tensor, t: float) -> torch.Tensor:
    ang = math.acos(max(-1.0, min(1.0, float((d1 * d2).sum()))))
    if ang < 1e-4:
        v = (1.0 - t) * d1 + t * d2
    else:
        v = (math.sin((1.0 - t) * ang) * d1
             + math.sin(t * ang) * d2) / math.sin(ang)
    return v / v.norm()


def sample_params(rng: np.random.Generator) -> dict:
    # ~15% of samples fall in the twilight band [-12, 0); the rest are daytime.
    # Elevation never goes below -12 deg, staying clear of the parked -45 label.
    if rng.random() < 0.15:
        el = float(rng.uniform(-12.0, -0.5))
    else:
        el = float(rng.uniform(2.0, 85.0))
    # Sun intensity/sharpness calibrated so the fixed-gauge sun-peak ratio
    # matches real RENI_HDR (bright, compact discs well above the sky p99).
    return {
        "sun_elevation_deg": el,
        "sun_azimuth_deg": float(rng.uniform(-180.0, 180.0)),
        "turbidity": float(rng.uniform(1.8, 6.5)),
        "sun_intensity": float(np.exp(rng.uniform(np.log(30.0), np.log(500.0)))),
        "sun_sharpness": float(np.exp(rng.uniform(np.log(200.0), np.log(1600.0)))),
        "ground_albedo": tuple(float(v) for v in rng.uniform(
            0.05, 0.35, 3) * np.array([1.0, 0.9, 0.75])),
    }


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",")
                   if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--preview", type=int, default=0,
                        help="Render N samples into a PNG grid.")
    parser.add_argument("--emit", type=int, default=0,
                        help="Emit N EXRs plus sun_labels.json.")
    parser.add_argument("--emit-pairs", type=int, default=0,
                        help="Emit N counterfactual triplets (3N EXRs): "
                             "identical nuisance content; members a/b have "
                             "independent suns, c the slerp midpoint.")
    parser.add_argument(
        "--emit-lattice", type=int, default=0,
        help="Emit N V8 nuisance scenes over a dense, shared sun lattice. "
             "Every member of a scene has identical sampled content and "
             "differs only through sun-dependent illumination.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="For --emit-lattice, append to an existing dataset and merge "
             "its sun_labels.json. New lattice filenames and group IDs use "
             "a separate namespace.",
    )
    parser.add_argument("--lattice-azimuths", type=int, default=12)
    parser.add_argument(
        "--lattice-elevations", type=_parse_float_list,
        default=(10.0, 35.0, 60.0, 80.0),
        help="Comma-separated daytime elevations in degrees.",
    )
    parser.add_argument("--lattice-twilight-azimuths", type=int, default=4)
    parser.add_argument(
        "--lattice-twilight-elevations", type=_parse_float_list,
        default=(-3.0, -7.0, -11.0),
        help="Comma-separated twilight elevations in degrees.",
    )
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
            d_a = _el_az_to_dir(p["sun_elevation_deg"], p["sun_azimuth_deg"])
            d_b = _el_az_to_dir(alt["sun_elevation_deg"],
                                alt["sun_azimuth_deg"])
            # Quad group: a (t=0), c (t=1/3), d (t=2/3), b (t=1). Interpolated
            # elevations are clamped to [-12, 85] so a slerp arc between two
            # twilight suns can never dip toward the parked -45 label.
            def _slerp_elaz(t):
                el, az = _dir_to_el_az(_slerp(d_a, d_b, t))
                return (min(max(el, -12.0), 85.0), az)
            members = (("a", (p["sun_elevation_deg"], p["sun_azimuth_deg"])),
                       ("b", (alt["sun_elevation_deg"], alt["sun_azimuth_deg"])),
                       ("c", _slerp_elaz(1.0 / 3.0)),
                       ("d", _slerp_elaz(2.0 / 3.0)))
            for member, (el, az) in members:
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
        print(f"[saved] {args.emit_pairs} quads + sun_labels.json -> {args.output}")

    if args.emit_lattice:
        import pyexr

        if args.lattice_azimuths < 1 or args.lattice_twilight_azimuths < 1:
            raise ValueError("lattice azimuth counts must be positive")
        if any(not 0.0 < el <= 85.0 for el in args.lattice_elevations):
            raise ValueError("daytime lattice elevations must be in (0, 85]")
        if any(not -12.0 <= el < 0.0
               for el in args.lattice_twilight_elevations):
            raise ValueError(
                "twilight lattice elevations must be in [-12, 0)")

        args.output.mkdir(parents=True, exist_ok=True)
        labels_path = args.output / "sun_labels.json"
        labels = (
            json.loads(labels_path.read_text())
            if args.append and labels_path.exists()
            else {}
        )
        initial_label_count = len(labels)
        daylight_az = np.linspace(
            -180.0, 180.0, args.lattice_azimuths, endpoint=False)
        twilight_az = np.linspace(
            -180.0, 180.0, args.lattice_twilight_azimuths,
            endpoint=False)
        members_per_group = (
            len(args.lattice_elevations) * len(daylight_az)
            + len(args.lattice_twilight_elevations) * len(twilight_az)
        )

        for group_id in range(args.emit_lattice):
            # This is exactly the V8 parameter and nuisance pipeline. The
            # sampled sun coordinates are replaced by the lattice below;
            # turbidity, disc properties, ground material, scenery and
            # photometric jitter remain fixed throughout the group.
            base = sample_params(rng)
            nuisance_seed = int(rng.integers(0, 2**31))
            # A per-scene phase prevents the decoder from seeing only a
            # finite global lookup table while retaining uniform coverage
            # within every individual counterfactual group.
            phase = float(rng.uniform(
                0.0, 360.0 / max(args.lattice_azimuths, 1)))

            commands = []
            for elevation in args.lattice_elevations:
                for azimuth in daylight_az:
                    commands.append(
                        ("daylight", float(elevation),
                         float((azimuth + phase + 180.0) % 360.0 - 180.0)))
            for elevation in args.lattice_twilight_elevations:
                for azimuth in twilight_az:
                    commands.append(
                        ("twilight", float(elevation),
                         float((azimuth + phase + 180.0) % 360.0 - 180.0)))

            for member_index, (kind, elevation, azimuth) in enumerate(commands):
                params = dict(base)
                params["sun_elevation_deg"] = elevation
                params["sun_azimuth_deg"] = azimuth
                name = (
                    f"synth_g{group_id:04d}_m{member_index:03d}.exr")
                image_path = args.output / name
                if name in labels:
                    raise FileExistsError(
                        f"refusing to replace labelled lattice member {name}")
                if image_path.exists():
                    if not args.append:
                        raise FileExistsError(
                            f"refusing to overwrite lattice member {name}")
                    # Recovery is deterministic: with the same seed, an
                    # interrupted append can rebuild the sidecar for EXRs
                    # already written before the final atomic bookkeeping
                    # step, without touching their bytes.
                    sun_d = _el_az_to_dir(elevation, azimuth)
                else:
                    img, sun_d = render_sky(
                        args.height, args.width, dirs=dirs,
                        nuisance_rng=np.random.default_rng(nuisance_seed),
                        **params,
                    )
                    img = apply_gauge(img)
                    pyexr.write(str(image_path), img.numpy())
                labels[name] = {
                    "sun_direction": [float(v) for v in sun_d],
                    "group_id": f"v8_lattice_{group_id:04d}",
                    "member": f"m{member_index:03d}",
                    "lattice_kind": kind,
                    "nuisance_seed": nuisance_seed,
                    "azimuth_phase_deg": phase,
                    **{key: value for key, value in params.items()},
                }
            if (group_id + 1) % 25 == 0:
                print(f"  [lattice] {group_id + 1}/{args.emit_lattice}")

        labels_path.write_text(json.dumps(labels, indent=1))
        emitted = len(labels) - initial_label_count
        print(
            f"[saved] {args.emit_lattice} V8 lattice groups x "
            f"{members_per_group} members = {emitted} new EXRs "
            f"({len(labels)} total) "
            f"+ sun_labels.json -> {args.output}")


if __name__ == "__main__":
    main()
