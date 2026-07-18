"""Frustum completion for SH with a Gaussian prior over coefficients.

Builds the classical-statistics baseline between "no prior" (exact SH least
squares, which explodes outside the observed view) and the learnt RENI++
prior. Pipeline, all in the fixed gauge (GT p99 luminance = 1), log domain,
64x128:

1. Fit ninth-order SH coefficients to every training environment in closed
   form, with exact azimuth augmentation (integer ERP column rolls are exact
   rotations about the gravity axis, and the least-squares solution of the
   rolled image equals the rotated solution).
2. Estimate a Gaussian over the flattened [num_coeffs*3] coefficients
   (shrinkage-regularised covariance) plus the scalar residual variance.
3. MAP-fit the visible frustum pixels in closed form,
       (B^T B kron I_3 / sigma^2 + lambda Sigma^-1) c
           = B^T y / sigma^2 + lambda Sigma^-1 mu,
   with lambda chosen on the 10 validation environments by hidden-region
   LDR PSNR under the same protocol.
4. Score the 21 test environments through the evaluate_baseline scoring
   path (GT-derived exposure, radiance cap) and emit the standard CSV.

    PYTHONPATH=.:scripts/figures python scripts/figures/eval_outpaint_sh_prior.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from _common import MODEL_DIRS, REPO_ROOT, equirect_ray_bundle, load_model, seed_all
from eval_outpaint_compare import (
    EPS,
    METRICS,
    REGIONS,
    _frustum_mask,
    region_metrics,
    region_metrics_masked,
)
from reni.illumination_fields.sh_illumination_field import shEvaluate
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import apply_fixed_gauge

OUT_DIR = REPO_ROOT / "outputs" / "evaluations"
PRIOR_PATH = REPO_ROOT / "outputs" / "reni" / "sh_gaussian_prior_9th.pt"
H, W = 64, 128
N_ROLLS = 8
LAMBDAS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0,
           30.0, 100.0, 300.0, 1e3, float("inf"))


def load_gauge_log(path: str, device: str) -> torch.Tensor:
    """EXR -> [H, W, 3] log of fixed-gauge linear radiance."""
    import numpy as np
    import pyexr

    img = pyexr.read(path).astype("float32")[..., :3]
    finite = np.isfinite(img)
    if not finite.all():
        img[~finite] = img[finite].max()
    img[img <= 0] = img[img > 0].min()
    img = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None],
                        size=(H, W), mode="bilinear")[0].permute(1, 2, 0)
    return torch.log(apply_fixed_gauge(img.to(device)) + EPS)


def design_matrix(ray_bundle, order: int, device: str) -> torch.Tensor:
    d = ray_bundle.directions
    return shEvaluate(torch.acos(d[:, 2]), torch.atan2(d[:, 0], d[:, 1]),
                      order, device=device)  # [H*W, n_coeffs]


def fit_training_prior(basis: torch.Tensor, device: str):
    """Closed-form SH fits to every training env at N_ROLLS azimuths ->
    Gaussian moments over flattened coefficients + residual variance."""
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / "train" / "*.exr")))
    print(f"[prior] fitting {len(files)} training envs x {N_ROLLS} rolls")
    pinv = torch.linalg.pinv(basis.double())  # [n_coeffs, H*W]
    n_coeffs = basis.shape[1]
    samples: List[torch.Tensor] = []
    sq_residual_sum, residual_count = 0.0, 0
    for i, f in enumerate(files):
        y = load_gauge_log(f, device).double()  # [H, W, 3]
        for k in range(N_ROLLS):
            y_roll = torch.roll(y, shifts=k * (W // N_ROLLS), dims=1).reshape(-1, 3)
            c = pinv @ y_roll  # [n_coeffs, 3]
            samples.append(c.reshape(-1).cpu())
            if k == 0:
                res = basis.double() @ c - y_roll
                sq_residual_sum += float((res ** 2).sum())
                residual_count += res.numel()
        if (i + 1) % 200 == 0:
            print(f"  [prior] {i + 1}/{len(files)}")
    coeffs = torch.stack(samples)  # [N*N_ROLLS, n_coeffs*3]
    keep = torch.isfinite(coeffs).all(dim=1)
    if int((~keep).sum()):
        print(f"[prior] dropping {int((~keep).sum())} non-finite samples")
    coeffs = coeffs[keep]
    mu = coeffs.mean(dim=0)
    centred = coeffs - mu
    sigma = centred.T @ centred / (coeffs.shape[0] - 1)
    sigma += 1e-3 * sigma.diagonal().mean() * torch.eye(sigma.shape[0])
    sigma_var = sq_residual_sum / residual_count
    print(f"[prior] {coeffs.shape[0]} samples, dim {coeffs.shape[1]}, "
          f"residual var {sigma_var:.4f}")
    return {"mu": mu, "sigma": sigma, "noise_var": sigma_var,
            "n_coeffs": n_coeffs, "num_train": len(files), "n_rolls": N_ROLLS}


def map_solve(basis_vis: torch.Tensor, y_vis: torch.Tensor, prior: Dict,
              lam: float) -> torch.Tensor:
    """Closed-form MAP coefficients [n_coeffs, 3] for one image."""
    n = prior["n_coeffs"]
    if lam == float("inf"):
        # Pure prior-mean limit: ignore the observation entirely.
        return prior["mu_dev"].reshape(n, 3)
    btb = basis_vis.T @ basis_vis  # [n, n]
    bty = basis_vis.T @ y_vis      # [n, 3]
    lik = torch.kron(btb, torch.eye(3, dtype=btb.dtype, device=btb.device))
    a = lik / prior["noise_var"]
    rhs = (bty.reshape(-1) / prior["noise_var"])
    if lam > 0:
        sigma_inv = prior["sigma_inv"]
        a = a + lam * sigma_inv
        rhs = rhs + lam * (sigma_inv @ prior["mu_dev"])
    c = torch.linalg.solve(a, rhs)
    return c.reshape(n, 3)


def evaluate_split(split: str, basis: torch.Tensor, mask: torch.Tensor,
                   prior: Dict, lam: float, model, device: str):
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / split / "*.exr")))
    visible = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)
    basis_vis = basis.double()[visible]
    per_image: List[Dict[str, Dict[str, float]]] = []
    for f in files:
        y = load_gauge_log(f, device)
        gt_lin = torch.exp(y) - EPS
        c = map_solve(basis_vis, y.reshape(-1, 3).double()[visible], prior, lam)
        pred_lin = torch.exp(basis.double() @ c).float().reshape(H, W, 3)
        pred_lin = torch.nan_to_num(pred_lin, nan=0.0, posinf=1e6,
                                    neginf=0.0).clamp(0.0, 1e6)
        # Table convention (matches evaluate_run): each image tone-mapped
        # with its own quantile exposure.
        gt_ldr = linear_to_sRGB(gt_lin, use_quantile=True)
        pred_ldr = linear_to_sRGB(pred_lin, use_quantile=True)
        # Probe: GT-derived exposure for the prediction (recorded, not primary).
        pred_ldr_gtq = linear_to_sRGB(
            pred_lin, q=torch.quantile(gt_lin.flatten(), 0.98))
        predq_hidden = region_metrics_masked(
            pred_lin, gt_lin, pred_ldr_gtq, gt_ldr, 1.0 - mask)["psnr_ldr"]
        per_image.append({
            "full": region_metrics(model, pred_lin, pred_lin, gt_lin,
                                   pred_ldr, gt_ldr, "full", "none"),
            "visible": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr, mask),
            "hidden": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr,
                                            1.0 - mask),
            "gtq_hidden_psnr_ldr": predq_hidden,
        })
    averaged = {
        region: {m: float(torch.nanmean(torch.tensor(
            [img[region][m] for img in per_image]))) for m in METRICS}
        for region in REGIONS
    }
    predq = float(torch.tensor(
        [img["gtq_hidden_psnr_ldr"] for img in per_image]).nanmean())
    print(f"    [{split}] hidden ldr={averaged['hidden']['psnr_ldr']:6.2f} "
          f"(gt-exposure convention {predq:6.2f})")
    return averaged, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order", type=int, default=9)
    parser.add_argument("--fov-h", type=float, default=90.0)
    parser.add_argument("--fov-v", type=float, default=60.0)
    parser.add_argument("--refit-prior", action="store_true",
                        help="Refit the training prior even if cached.")
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "outpaint_frustum90x60_sh_prior")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)
    device = args.device

    _, _, model = load_model(MODEL_DIRS["sh"]["9th"], device=device)
    model.eval()
    ray_bundle = equirect_ray_bundle(device, idx=0, height=H)
    basis = design_matrix(ray_bundle, args.order, device)
    mask = _frustum_mask(H, W, args.fov_h, args.fov_v)

    if PRIOR_PATH.exists() and not args.refit_prior:
        prior = torch.load(PRIOR_PATH)
        print(f"[prior] loaded cache {PRIOR_PATH}")
    else:
        prior = fit_training_prior(basis, device)
        PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prior, PRIOR_PATH)
        print(f"[prior] saved {PRIOR_PATH}")
    prior["mu"] = prior["mu"].to(device)
    prior["sigma"] = prior["sigma"].to(device)
    prior["sigma_inv"] = torch.linalg.inv(prior["sigma"])
    prior["mu_dev"] = prior["mu"]

    print("[tune] lambda sweep on val (hidden-region LDR PSNR)")
    best_lam, best_val = None, -1e9
    sweep = {}
    for lam in LAMBDAS:
        val_metrics, _ = evaluate_split("val", basis, mask, prior, lam, model, device)
        hidden_ldr = val_metrics["hidden"]["psnr_ldr"]
        sweep[lam] = {r: {k: val_metrics[r][k] for k in ("psnr_ldr", "psnr_hdr")}
                      for r in REGIONS}
        print(f"  lambda={lam:<7g} hidden ldr={hidden_ldr:6.2f} "
              f"hdr={val_metrics['hidden']['psnr_hdr']:8.2f} "
              f"visible ldr={val_metrics['visible']['psnr_ldr']:6.2f}")
        if hidden_ldr > best_val:
            best_lam, best_val = lam, hidden_ldr
    print(f"[tune] chosen lambda={best_lam}")

    test_metrics, per_image = evaluate_split("test", basis, mask, prior,
                                             best_lam, model, device)
    print(f"[test] hidden ldr={test_metrics['hidden']['psnr_ldr']:6.2f} "
          f"hdr={test_metrics['hidden']['psnr_hdr']:8.2f} "
          f"visible ldr={test_metrics['visible']['psnr_ldr']:6.2f} "
          f"full ldr={test_metrics['full']['psnr_ldr']:6.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "sh_prior",
        "order": args.order,
        "lambda": best_lam,
        "lambda_sweep_val": {str(k): v for k, v in sweep.items()},
        "prior_cache": str(PRIOR_PATH),
        "num_train": prior.get("num_train"),
        "n_rolls": prior.get("n_rolls"),
        "noise_var": prior.get("noise_var"),
        "metrics": test_metrics,
        "per_image": per_image,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [("model", "region", "num_eval_images", *METRICS)]
    for region in REGIONS:
        rows.append(("sh_prior", region, len(per_image),
                     *(f"{test_metrics[region][m]:.6f}" for m in METRICS)))
    with args.output.with_suffix(".csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[saved] {args.output}.json / .csv")


if __name__ == "__main__":
    main()
