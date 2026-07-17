"""Programmatic rotation-equivariance error experiment for RENI++ (Ch 2).

Replaces the hand-made ``tab:rotation_error_comparison`` (from
``publication/non_convexity.ipynb``) with a controlled 2x2 experiment that
separates two explanations for the published table's large errors at small
angles (5/20/45 deg) and tiny errors at 90/180/270:

* (a) James's hypothesis: random ray sampling during the latent refit breaks
  exact recovery, and full-image (deterministic) batches should recover the
  rotated latent exactly by equivariance (zero-init is a fixed point of the
  rotation action, so a full-gradient trajectory started at zero commutes with
  the rotation R -- the fit to the rotated target traces R applied to the fit
  to the unrotated target, landing on R(z0) at every step).

* (b) Refinement: 90-degree multiples are exact integer column shifts of the
  equirectangular image (lossless), while other angles require resampling the
  equirect image to build the rotated target -- resampling blurs/quantises the
  target and moves the optimum itself away from R(z0). The target breaks
  equivariance, not the optimiser.

The experiment crosses two factors and produces the 2x2:

    target mode in {rotated-directions (lossless), resampled-image}
      x  ray regime in {full-image, random}

For each test image we fit a fresh reference latent ``z0`` from zero against the
unrotated target under each ray regime (this is the angle-0 fit, exactly as the
original notebook took ``model_latents[0]`` as its reference). For each angle we
build the rotated target in each mode, refit a fresh latent from zero, and
measure how close it lands to ``R(z0) = z0 @ R`` (the field applies the latent
rotation as ``latent_codes @ rotation`` in ``get_outputs``; the current
implementation's invariance axis is z, so R = rot_z(theta)).

* rotated-directions mode keeps the image pixels and rotates the ray directions
  by R -- the represented signal is the rotated envmap, losslessly (no
  resampling). This is the corrected/headline protocol.
* resampled-image mode keeps canonical directions and builds the rotated target
  by rolling the equirect image columns by W*theta/360 with bilinear
  interpolation (exact ``np.roll`` at integer/90-degree shifts, blurred
  otherwise) -- faithfully reproducing what the dataparser's
  ``apply_eval_rotation`` and the original table did.

Metrics per (angle, mode, regime), mean +/- std over images:
* relative latent error ||z_fit - R(z0)|| / ||R(z0)|| (direct; the strict
  exact-recovery measure -- this is the headline).
* Procrustes relative error ||z0 @ R_est - z_fit|| / ||z_fit|| with
  R_est = special_procrustes(lstsq(z0, z_fit)) over full SO(3) -- the exact
  definition the published table used ('Relative Error').
* rotation-matrix error ||R_gt - R_est||_F and the geodesic angle between them,
  where R_gt = rot_z(theta) ('Ground Truth Rotation Matrix Error').

The fit mirrors the model's eval-latent recipe read from the checkpoint config
(Adam lr=0.1, cosine-ramped exponential decay, the two-bracket bracket + blended
losses with their trained coefficients). The full-image batch equals one 64x128
image (8192 rays) exactly, matching ``eval_num_rays_per_batch``.

A third factor -- the optimiser -- is crossed in to isolate the mechanism. Plain
gradient descent (SGD, optionally with momentum) is rotation-equivariant: for an
orthogonal R the gradient of the rotated problem obeys grad L_R(z0 @ R) =
grad L(z0) @ R, so a trajectory started at zero satisfies z_fit(t) = z0(t) @ R at
every step, recovering R(z0) to machine precision regardless of convergence or
angle. Adam is NOT: its per-coordinate second moment (element-wise 1/sqrt(v))
only commutes with rotations that are signed axis permutations -- exactly the
90/180/270-degree rotations about the z-axis -- so under Adam the fit recovers
R(z0) exactly at those angles but drifts into a different basin at 5/20/45. This
is the actual explanation for the published table's shape, distinct from both
(a) and (b).

Run from the ns_reni repo root (inside the phd research container):

    RENI_PAPER_MODELS=/home/james/model-storage/reni_paper_models \\
    PYTHONPATH=. python scripts/figures/make_rotation_error_table.py

Outputs publication/tables/rotation_error.{csv,tex}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from _common import (
    MODEL_DIRS,
    REPO_ROOT,
    _clean_yaml,
    equirect_ray_bundle,
    init_fit_latent,
    load_model,
    resolve_run_dir,
    rotation_fn,
    seed_all,
)
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.tonemap import two_bracket_to_linear

DEFAULT_MODEL = MODEL_DIRS["two_bracket_w3_1cyc_testfit"][100]
DEFAULT_ANGLES = (5, 20, 45, 90, 180, 270)
DEFAULT_IMAGE_INDICES = (0, 2, 5, 8, 11, 14, 17, 20)
OUT_DIR = REPO_ROOT / "publication" / "tables"

MODES = ("rotated_directions", "resampled_image")
REGIMES = ("full_image", "random")
OPTIMIZERS = ("adam", "gd")


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _two_bracket_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    m_ldr: float,
    m_log: float,
    coeffs: Dict[str, float],
) -> torch.Tensor:
    """Eval-fit loss for a two-bracket model, mirroring RENIModel.get_loss_dict.

    3*MSE(ldr bracket) + 1*MSE(log bracket) + 1*MSE(log1p blended-HDR).
    Coefficients come from the checkpoint config (loss_coefficients).
    """
    mse = torch.nn.functional.mse_loss
    ldr = mse(pred[..., :3], target[..., :3])
    log = mse(pred[..., 3:6], target[..., 3:6])
    pred_lin = two_bracket_to_linear(pred, m_ldr=m_ldr, m_log=m_log)
    tgt_lin = two_bracket_to_linear(target, m_ldr=m_ldr, m_log=m_log)
    blended = mse(torch.log1p(pred_lin), torch.log1p(tgt_lin))
    return (
        coeffs["ldr_bracket_mse_loss"] * ldr
        + coeffs["log_bracket_mse_loss"] * log
        + coeffs["blended_recon_loss"] * blended
    )


def _lr_at(step: int, steps: int, lr_init: float, lr_final: float) -> float:
    """Cosine-ramped exponential decay (nerfstudio ExponentialDecayScheduler,
    ramp='cosine', warmup_steps=0): interpolate log-lr on a cosine schedule."""
    if steps <= 1:
        return lr_init
    t = min(step / (steps - 1), 1.0)
    ramp = (1.0 - math.cos(math.pi * t)) / 2.0  # 0 -> 1
    log_lr = math.log(lr_init) + ramp * (math.log(lr_final) - math.log(lr_init))
    return math.exp(log_lr)


def fit_latent(
    model,
    directions: torch.Tensor,
    target: torch.Tensor,
    latent_dim: int,
    device: str,
    steps: int,
    lr_init: float,
    lr_final: float,
    coeffs: Dict[str, float],
    regime: str,
    batch: int,
    seed: int,
    optimizer: str = "adam",
    gd_lr: float = 3.0,
    gd_momentum: float = 0.9,
    return_stats: bool = False,
    z_init: torch.Tensor = None,
):
    """Fit a single [latent_dim, 3] latent from ``z_init`` against ``target``.

    ``directions`` [N, 3] are paired row-for-row with ``target`` [N, 6]. The
    decoder is held fixed; only the latent is optimised. The default init is
    the prior origin, a fixed point of the rotation action (0 @ R = 0), so in
    the full-image regime a plain-gradient trajectory commutes with R exactly.
    Frame-normalised variants cannot start at exactly zero (gradient
    singularity), so callers pass a shared noise init and CO-ROTATE it for
    the rotated fits (z_init @ R), which preserves the commutation argument;
    for zero inits that reduces to today's behaviour identically.

    optimizer='adam' mirrors the eval-fit recipe (Adam, cosine-ramped decay).
    optimizer='gd' is momentum SGD, which is rotation-equivariant.
    """
    n = directions.shape[0]
    if z_init is None:
        z = init_fit_latent(model, device, dtype=directions.dtype)
    else:
        z = z_init.detach().clone().to(device=device, dtype=directions.dtype)
        z.requires_grad_(True)
    if optimizer == "adam":
        optim = torch.optim.Adam([z], lr=lr_init, eps=1e-15, weight_decay=0.0)
    elif optimizer == "gd":
        optim = torch.optim.SGD([z], lr=gd_lr, momentum=gd_momentum)
    else:
        raise ValueError(optimizer)
    cam_indices = torch.zeros(n, 1, dtype=torch.long, device=device)
    origins = torch.zeros_like(directions)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    last_loss = float("nan")
    for step in range(steps):
        if optimizer == "adam":
            for group in optim.param_groups:
                group["lr"] = _lr_at(step, steps, lr_init, lr_final)
        if regime == "full_image":
            sel = slice(None)
        else:
            sel = torch.randint(0, n, (batch,), generator=gen, device=device)
        d_sel = directions[sel]
        ray_samples = model.create_ray_samples(
            origins[sel], d_sel, cam_indices[sel]
        )
        pred = model.field.forward(
            ray_samples, latent_codes=z.expand(d_sel.shape[0], -1, -1)
        )[RENIFieldHeadNames.RGB]
        loss = _two_bracket_loss(
            pred, target[sel], model.tonemap_m_ldr, model.tonemap_m_log, coeffs
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        last_loss = float(loss.detach())

    zc = z.detach()[0]
    if return_stats:
        return zc, {"final_loss": last_loss, "z_norm": float(torch.norm(zc))}
    return zc


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def resample_columns(target_hw6: torch.Tensor, shift_cols: float) -> torch.Tensor:
    """Bilinear equirect rotation about the polar axis = subpixel column roll.

    ``output[:, c] = input[:, (c + shift_cols) mod W]`` (matching np.roll by
    -round(shift_cols)); bilinear between neighbouring columns for fractional
    shifts, exact copy at integer shifts (90-degree multiples on this grid)."""
    H, W, C = target_hw6.shape
    cols = torch.arange(W, device=target_hw6.device, dtype=torch.float32)
    src = cols + shift_cols
    base = torch.floor(src)
    frac = (src - base).view(1, W, 1)
    c0 = (base.long() % W)
    c1 = ((base.long() + 1) % W)
    return target_hw6[:, c0, :] * (1.0 - frac) + target_hw6[:, c1, :] * frac


def build_condition_target(
    mode: str,
    directions: torch.Tensor,
    target_flat: torch.Tensor,
    target_hw6: torch.Tensor,
    R: torch.Tensor,
    dir_conv: str,
    roll_sign: float,
    angle_deg: float,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (directions, target_flat) for a given rotated condition."""
    if mode == "rotated_directions":
        Rd = R if dir_conv == "R" else R.T
        return directions @ Rd, target_flat
    # resampled_image: roll the equirect columns, keep canonical directions.
    shift = roll_sign * width * angle_deg / 360.0
    rolled = resample_columns(target_hw6, shift)
    return directions, rolled.reshape(-1, target_hw6.shape[-1])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def special_procrustes(M: torch.Tensor) -> torch.Tensor:
    """Nearest rotation matrix (SO(3)) to a 3x3 matrix M via SVD."""
    U, _, Vh = torch.linalg.svd(M.double())
    d = torch.sign(torch.det(U @ Vh))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64, device=M.device))
    return (U @ D @ Vh).to(M.dtype)


def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    cos = (torch.trace(Ra.double().T @ Rb.double()) - 1.0) / 2.0
    return math.degrees(math.acos(float(cos.clamp(-1.0, 1.0))))


def condition_metrics(z0: torch.Tensor, zfit: torch.Tensor, R: torch.Tensor) -> Dict[str, float]:
    z0R = z0 @ R
    rel_direct = (torch.norm(zfit - z0R) / torch.norm(z0R)).item()
    M = torch.linalg.lstsq(z0.double(), zfit.double()).solution
    R_est = special_procrustes(M)  # double
    Rd = R.double()
    rel_proc = (torch.norm(z0.double() @ R_est - zfit.double()) / torch.norm(zfit.double())).item()
    rot_fro = torch.norm(Rd - R_est, p="fro").item()
    rot_geo = geodesic_deg(Rd, R_est)
    return {
        "rel_err_direct": rel_direct,
        "rel_err_procrustes": rel_proc,
        "rot_mat_err_fro": rot_fro,
        "rot_geodesic_deg": rot_geo,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    std = t.std(unbiased=True).item() if len(values) > 1 else 0.0
    return t.mean().item(), std


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(
    model, directions, target_flat, target_hw6, z0_full, R90, latent_dim, device,
    steps, lr_init, lr_final, coeffs, width,
) -> Tuple[str, float]:
    """Pick the direction-rotation convention (R vs R^T) and column-roll sign
    that recover R(z0)=z0@R90 at a 90-degree (exact) shift in the full regime.

    The correct choice gives a direct latent error near machine precision; the
    wrong choice lands on z0@R270 and is O(1), so the argmin is unambiguous."""
    z0R = z0_full @ R90

    def err(zfit):
        return (torch.norm(zfit - z0R) / torch.norm(z0R)).item()

    dir_errs = {}
    for conv in ("R", "RT"):
        Rd = R90 if conv == "R" else R90.T
        zfit = fit_latent(model, directions @ Rd, target_flat, latent_dim, device,
                          steps, lr_init, lr_final, coeffs, "full_image", 0, 0)
        dir_errs[conv] = err(zfit)
    dir_conv = min(dir_errs, key=dir_errs.get)

    roll_errs = {}
    for sign in (1.0, -1.0):
        shift = sign * width * 90.0 / 360.0
        rolled = resample_columns(target_hw6, shift).reshape(-1, target_hw6.shape[-1])
        zfit = fit_latent(model, directions, rolled, latent_dim, device,
                          steps, lr_init, lr_final, coeffs, "full_image", 0, 0)
        roll_errs[sign] = err(zfit)
    roll_sign = min(roll_errs, key=roll_errs.get)

    print(f"[calibrate] dir convention errors {dir_errs} -> {dir_conv}")
    print(f"[calibrate] roll sign errors {roll_errs} -> {roll_sign:+.0f}")
    return dir_conv, roll_sign


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image_indices", type=int, nargs="+", default=list(DEFAULT_IMAGE_INDICES))
    parser.add_argument("--angles", type=int, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--random-batch", type=int, default=8192,
                        help="Rays per step in the random regime (matches eval_num_rays_per_batch).")
    parser.add_argument("--lr-init", type=float, default=0.1)
    parser.add_argument("--lr-final", type=float, default=1e-7)
    parser.add_argument("--gd-lr", type=float, default=3.0,
                        help="Learning rate for the momentum-SGD (rotation-equivariant) control.")
    parser.add_argument("--gd-momentum", type=float, default=0.9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                        help="float64 casts the decoder + fit to double; the rotated "
                             "and unrotated computation graphs then agree to ~1e-12, so "
                             "the equivariant (GD) recovery reaches machine precision.")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "rotation_error")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[load] {args.model_dir}")
    _, datamanager, model = load_model(Path(args.model_dir), device=args.device)
    assert getattr(model, "two_bracket", False), "This script targets a two-bracket model."
    latent_dim = model.field.latent_dim

    # load_model rebuilds from RENIField.config defaults and does not copy the
    # trained loss_coefficients; read them from the saved config so the fit loss
    # matches the eval-fit recipe (3x LDR-bracket weight for the w3 model).
    saved = _clean_yaml((resolve_run_dir(Path(args.model_dir)) / "config.yml").read_text())
    saved_coeffs = saved["pipeline"]["model"].get("loss_coefficients", {})
    coeffs = {
        "ldr_bracket_mse_loss": float(saved_coeffs.get("ldr_bracket_mse_loss", 3.0)),
        "log_bracket_mse_loss": float(saved_coeffs.get("log_bracket_mse_loss", 1.0)),
        "blended_recon_loss": float(saved_coeffs.get("blended_recon_loss", 1.0)),
    }
    print(f"[config] latent_dim={latent_dim} loss_coeffs={coeffs} "
          f"m_ldr={model.tonemap_m_ldr} m_log={model.tonemap_m_log} dtype={args.dtype}")

    if dtype == torch.float64:
        model.field.double()

    rot = rotation_fn(model)  # rot_z for the current implementation
    R_of = lambda deg: rot(torch.deg2rad(torch.tensor(float(deg)))).to(args.device).to(dtype)

    # Stored eval latents, for a sanity comparison against the fresh angle-0 fit.
    stored_eval = model.field.eval_mu.detach()

    # results[(angle, mode, regime, optimizer)][metric] = list over images
    results: Dict[Tuple[int, str, str, str], Dict[str, List[float]]] = {}
    stored_agreement: List[float] = []
    ref_stats: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    # No-rotation self-consistency floor: refit the SAME unrotated target and
    # measure ||z0b - z0|| / ||z0||. This is the rotation-free analogue of the
    # recovery error (GPU nondeterminism for full-image, sampling noise for
    # random) -- the floor the rotated fits should match if recovery is exact.
    floor: Dict[Tuple[str, str], List[float]] = {}

    def do_fit(directions, target, regime, opt, seed, **kw):
        return fit_latent(
            model, directions, target, latent_dim, args.device, args.steps,
            args.lr_init, args.lr_final, coeffs, regime, args.random_batch,
            seed=seed, optimizer=opt, gd_lr=args.gd_lr, gd_momentum=args.gd_momentum,
            **kw)

    for img_i in args.image_indices:
        batch = datamanager.eval_dataset[img_i]
        target_hw6 = batch["image"].to(args.device).to(dtype)
        H, W = target_hw6.shape[0], target_hw6.shape[1]
        target_flat = target_hw6.reshape(-1, target_hw6.shape[-1])
        ray_bundle = equirect_ray_bundle(args.device, idx=img_i, height=H)
        directions = ray_bundle.directions.to(args.device).to(dtype)
        assert directions.shape[0] == target_flat.shape[0], (
            f"ray/target mismatch: {directions.shape} vs {target_flat.shape}")

        # Shared per-image init: zeros for standard models; a fixed noise
        # draw for frame-normalised variants. Rotated fits co-rotate it.
        z_base = init_fit_latent(model, args.device, dtype=dtype,
                                 requires_grad=False)

        # Reference latents: fresh angle-0 fit, one per (regime, optimizer).
        z0: Dict[Tuple[str, str], torch.Tensor] = {}
        for regime in REGIMES:
            for opt in OPTIMIZERS:
                z0[(regime, opt)], st = do_fit(
                    directions, target_flat, regime, opt, args.seed,
                    return_stats=True, z_init=z_base)
                ref_stats.setdefault((regime, opt), []).append(st)
                # Second unrotated fit -> no-rotation floor (same seed as the
                # rotated conditions use +angle offset; random draws differ,
                # full-image is deterministic so this isolates nondeterminism).
                z0b = do_fit(directions, target_flat, regime, opt, args.seed + 1,
                             z_init=z_base)
                floor.setdefault((regime, opt), []).append(
                    (torch.norm(z0b - z0[(regime, opt)]) / torch.norm(z0[(regime, opt)])).item())
        if img_i < stored_eval.shape[0]:
            zs = stored_eval[img_i]
            stored_agreement.append(
                (torch.norm(z0[("full_image", "adam")] - zs) / torch.norm(zs)).item())

        # Calibrate direction convention + roll sign once (cheap, reused).
        if img_i == args.image_indices[0]:
            dir_conv, roll_sign = calibrate(
                model, directions, target_flat, target_hw6, z0[("full_image", "adam")],
                R_of(90), latent_dim, args.device, args.steps, args.lr_init,
                args.lr_final, coeffs, W)

        for angle in args.angles:
            R = R_of(angle)
            for mode in MODES:
                d_cond, t_cond = build_condition_target(
                    mode, directions, target_flat, target_hw6, R, dir_conv,
                    roll_sign, angle, W)
                for regime in REGIMES:
                    for opt in OPTIMIZERS:
                        zfit = do_fit(d_cond, t_cond, regime, opt,
                                      args.seed + angle + 1,
                                      z_init=z_base @ R)
                        m = condition_metrics(z0[(regime, opt)], zfit, R)
                        key = (angle, mode, regime, opt)
                        slot = results.setdefault(key, {k: [] for k in m})
                        for k, v in m.items():
                            slot[k].append(v)
        print(f"[image {img_i}] done ({H}x{W})")

    floor_mean = {}
    for (regime, opt), stats in ref_stats.items():
        loss = sum(s["final_loss"] for s in stats) / len(stats)
        zn = sum(s["z_norm"] for s in stats) / len(stats)
        fl = sum(floor[(regime, opt)]) / len(floor[(regime, opt)])
        floor_mean[(regime, opt)] = fl
        print(f"[ref-fit] {regime}/{opt}: mean final loss {loss:.5f}, mean |z| {zn:.3f}, "
              f"no-rotation floor {fl:.4f}")

    _write_outputs(args, results, dir_conv, roll_sign, stored_agreement, floor_mean)


def _write_outputs(args, results, dir_conv, roll_sign, stored_agreement, floor_mean) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ("rel_err_direct", "rel_err_procrustes", "rot_mat_err_fro", "rot_geodesic_deg")

    # --- CSV: the full grid (mean/std per angle x mode x regime x optimizer).
    csv_path = args.output.with_suffix(".csv")
    header = ["angle", "mode", "regime", "optimizer", "n_images"]
    for m in metrics:
        header += [f"{m}_mean", f"{m}_std"]
    rows = [header]
    agg: Dict[Tuple[int, str, str, str], Dict[str, Tuple[float, float]]] = {}
    order = lambda k: (k[0], MODES.index(k[1]), REGIMES.index(k[2]), OPTIMIZERS.index(k[3]))
    for key in sorted(results, key=order):
        angle, mode, regime, opt = key
        slot = results[key]
        agg[key] = {}
        row = [angle, mode, regime, opt, len(next(iter(slot.values())))]
        for m in metrics:
            mean, std = mean_std(slot[m])
            agg[key][m] = (mean, std)
            row += [f"{mean:.6g}", f"{std:.6g}"]
        rows.append(row)
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[saved] {csv_path}")

    # --- JSON sidecar with provenance.
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({
        "model_dir": str(args.model_dir),
        "image_indices": args.image_indices,
        "angles": args.angles,
        "steps": args.steps,
        "random_batch": args.random_batch,
        "lr_init": args.lr_init,
        "lr_final": args.lr_final,
        "gd_lr": args.gd_lr,
        "gd_momentum": args.gd_momentum,
        "seed": args.seed,
        "dir_convention": dir_conv,
        "roll_sign": roll_sign,
        "no_rotation_floor": {f"{r}|{o}": v for (r, o), v in floor_mean.items()},
        "stored_eval_vs_angle0_full_adam_relerr_mean": (
            sum(stored_agreement) / len(stored_agreement) if stored_agreement else None),
        "metrics": {
            f"{a}|{mo}|{re}|{op}": {m: agg[(a, mo, re, op)][m] for m in metrics}
            for (a, mo, re, op) in agg
        },
    }, indent=2) + "\n")
    print(f"[saved] {json_path}")

    # --- LaTeX headline: the corrected rotated-directions protocol, full-image
    # regime, contrasting the eval-fit optimiser (Adam) against plain gradient
    # descent (rotation-equivariant). Relative Error uses the published
    # Procrustes definition; Rot. Matrix Error is ||rot_z(theta) - R_est||_F.
    regime = "full_image"
    lines = [
        "% Generated by scripts/figures/make_rotation_error_table.py -- do not edit by hand",
        "% Corrected rotation-equivariance experiment (replaces tab:rotation_error_comparison).",
        "% Rotated-directions (lossless) target; full-image batches. Adam is the eval-fit",
        "% optimiser; gradient descent is rotation-equivariant and recovers R(z0) exactly.",
        "\\begin{tabular}{@{}c cc cc@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Adam (eval-fit optimiser)} "
        "& \\multicolumn{2}{c}{Gradient descent} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "\\makecell{Rotation\\\\(deg)} & Relative Error & \\makecell{Rot.\\ Matrix\\\\Error} "
        "& Relative Error & \\makecell{Rot.\\ Matrix\\\\Error} \\\\",
        "\\midrule",
    ]
    for angle in args.angles:
        cells = []
        for opt in OPTIMIZERS:
            re_m, re_s = agg[(angle, "rotated_directions", regime, opt)]["rel_err_procrustes"]
            rm_m, rm_s = agg[(angle, "rotated_directions", regime, opt)]["rot_mat_err_fro"]
            cells.append(f"${re_m:.3f} \\pm {re_s:.3f}$")
            cells.append(f"${rm_m:.3f} \\pm {rm_s:.3f}$")
        lines.append(f"{angle} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path = args.output.with_suffix(".tex")
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] {tex_path}")

    # --- Console summary: decisive direct exact-recovery numbers, all factors.
    print("\nDirect relative latent error ||z_fit - z0@R|| / ||z0@R||  (mean over images)")
    cols = [(mo, re, op) for mo in MODES for re in REGIMES for op in OPTIMIZERS]
    hdr = f"{'angle':>6} " + " ".join(
        f"{mo[:4]+'/'+re[:4]+'/'+op:>17}" for (mo, re, op) in cols)
    print(hdr)
    print("-" * len(hdr))
    for angle in args.angles:
        cells = [f"{agg[(angle, mo, re, op)]['rel_err_direct'][0]:.4f}" for (mo, re, op) in cols]
        print(f"{angle:>6} " + " ".join(f"{c:>17}" for c in cells))
    print("\nNo-rotation self-consistency floor (rotation-free recovery error):")
    for (regime, opt) in sorted(floor_mean):
        print(f"    {regime}/{opt}: {floor_mean[(regime, opt)]:.4f}")
    if stored_agreement:
        mean_agree = sum(stored_agreement) / len(stored_agreement)
        print(f"[sanity] fresh angle-0 full-image Adam fit vs stored eval latents: "
              f"mean rel err {mean_agree:.4f}")


if __name__ == "__main__":
    main()
