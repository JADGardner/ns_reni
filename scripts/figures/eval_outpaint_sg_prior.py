"""Frustum completion for SG with a Gaussian prior over raw parameters.

SG companion to eval_outpaint_sh_prior.py. All in the fixed gauge, log
domain, 64x128:

1. Batched fits of the raw SG parameters (log amplitudes, tanh-space angle
   offsets, log sharpness; the native positive parameterisation) to every
   training environment, 2,400 Adam steps at lr 1e-2 in image chunks.
2. Exact azimuth augmentation by cyclic permutation of lobe-grid columns
   (the phi grid is uniform, so permuting columns rotates the rendered
   function exactly).
3. Gaussian over the flattened raw parameters (shrinkage covariance).
4. MAP frustum fits: the standard 10*mse + cosine visible-pixel loss plus
   lambda * mean Mahalanobis penalty, batched over images, lambda chosen on
   the validation split by hidden-region LDR PSNR.
5. Test scoring through the evaluate_baseline path (GT-derived exposure,
   radiance cap); emits the standard CSV.

    PYTHONPATH=.:scripts/figures python scripts/figures/eval_outpaint_sg_prior.py
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
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import apply_fixed_gauge

OUT_DIR = REPO_ROOT / "outputs" / "evaluations"
PRIOR_PATH = REPO_ROOT / "outputs" / "reni" / "sg_gaussian_prior_300.pt"
H, W = 64, 128
FIT_STEPS = 2400
FIT_LR = 1e-2
TRAIN_CHUNK = 256
LAMBDAS = (0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def load_gauge_log(path: str, device: str) -> torch.Tensor:
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


class BatchedSG:
    """Batched log-domain SG renderer in the field's raw parameterisation."""

    def __init__(self, field, dirs: torch.Tensor):
        self.L = field.sg_num
        self.sg_row, self.sg_col = field.sg_row, field.sg_col
        self.theta_c = field.theta_center_train[:1].transpose(1, 2).to(dirs.device)
        self.phi_c = field.phi_center_train[:1].transpose(1, 2).to(dirs.device)
        self.theta_range = float(field.theta_range)
        self.phi_range = float(field.phi_range)
        self.theta_dir = torch.acos(dirs[:, 2])[None, :, None]   # [1, P, 1]
        self.phi_dir = torch.atan2(dirs[:, 0], dirs[:, 1])[None, :, None]
        self.init_lamb = float(torch.log(torch.tensor(3.141592653589793 / self.sg_row)))

    def init_u(self, batch: int, device: str) -> torch.Tensor:
        u = torch.zeros(batch, self.L, 6, device=device)
        u[..., 5] = self.init_lamb
        return u

    def render_log(self, u: torch.Tensor) -> torch.Tensor:
        """u [B, L, 6] -> log radiance [B, P, 3] (renderSG conventions)."""
        w = torch.exp(u[..., 0:3])                                   # [B, L, 3]
        th = (self.theta_range * torch.tanh(u[..., 3])[:, None, :]
              + self.theta_c)                                        # [B, 1|P, L]
        ph = self.phi_range * torch.tanh(u[..., 4])[:, None, :] + self.phi_c
        lam = torch.exp(u[..., 5])[:, None, :]                       # [B, 1, L]
        cos_angle = (torch.sin(th) * torch.sin(self.theta_dir)
                     * torch.cos(ph - self.phi_dir)
                     + torch.cos(th) * torch.cos(self.theta_dir))    # [B, P, L]
        expo = torch.exp(lam * (cos_angle - 1.0))                    # [B, P, L]
        rgb = torch.einsum("bpl,blc->bpc", expo, w)
        return torch.log(rgb + EPS)

    def column_permutations(self, u: torch.Tensor) -> torch.Tensor:
        """[N, L, 6] -> [N*sg_col, L, 6]: exact azimuth rotations by cyclic
        permutation of the lobe-grid columns (row-major [sg_row, sg_col])."""
        n = u.shape[0]
        grid = u.reshape(n, self.sg_row, self.sg_col, 6)
        rolls = [torch.roll(grid, shifts=k, dims=2) for k in range(self.sg_col)]
        return torch.cat(rolls, dim=0).reshape(n * self.sg_col, self.L, 6)


def fit_batch(renderer: BatchedSG, targets: torch.Tensor, visible, device: str,
              steps: int, prior=None, lam: float = 0.0) -> torch.Tensor:
    """Fit raw params to targets [B, P_sel, 3] on the selected pixels."""
    u = renderer.init_u(targets.shape[0], device).requires_grad_(True)
    optimiser = torch.optim.Adam([u], lr=FIT_LR)
    for _ in range(steps):
        optimiser.zero_grad()
        out = renderer.render_log(u)
        if visible is not None:
            out = out[:, visible]
        mse = ((out - targets) ** 2).mean()
        cosine = 1 - torch.nn.functional.cosine_similarity(
            out, targets, dim=-1).mean()
        loss = 10.0 * mse + cosine
        if prior is not None and lam > 0:
            dev = u.reshape(u.shape[0], -1).double() - prior["mu"][None]
            maha = torch.einsum("bi,ij,bj->b", dev, prior["sigma_inv"], dev)
            loss = loss + lam * (maha.mean() / dev.shape[1]).float()
        loss.backward()
        optimiser.step()
    return u.detach()


def fit_training_prior(renderer: BatchedSG, device: str) -> Dict:
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / "train" / "*.exr")))
    print(f"[prior] fitting {len(files)} training envs in chunks of {TRAIN_CHUNK}")
    fitted: List[torch.Tensor] = []
    sq_res, n_res = 0.0, 0
    for start in range(0, len(files), TRAIN_CHUNK):
        chunk_files = files[start:start + TRAIN_CHUNK]
        targets = torch.stack(
            [load_gauge_log(f, device).reshape(-1, 3) for f in chunk_files])
        u = fit_batch(renderer, targets, None, device, FIT_STEPS)
        with torch.no_grad():
            res = renderer.render_log(u) - targets
            sq_res += float((res ** 2).sum())
            n_res += res.numel()
        fitted.append(u.cpu())
        print(f"  [prior] {min(start + TRAIN_CHUNK, len(files))}/{len(files)}")
    u_all = torch.cat(fitted)
    samples = renderer.column_permutations(u_all).reshape(
        u_all.shape[0] * renderer.sg_col, -1).double()
    keep = torch.isfinite(samples).all(dim=1)
    if int((~keep).sum()):
        print(f"[prior] dropping {int((~keep).sum())} non-finite samples")
    samples = samples[keep]
    mu = samples.mean(dim=0)
    centred = samples - mu
    sigma = centred.T @ centred / (samples.shape[0] - 1)
    sigma += 1e-3 * sigma.diagonal().mean() * torch.eye(sigma.shape[0])
    noise_var = sq_res / n_res
    print(f"[prior] {samples.shape[0]} samples, dim {samples.shape[1]}, "
          f"residual var {noise_var:.4f}")
    return {"mu": mu, "sigma": sigma, "noise_var": noise_var,
            "num_train": len(files), "sg_col": renderer.sg_col}


def evaluate_split(split: str, renderer: BatchedSG, mask: torch.Tensor,
                   prior, lam: float, model, device: str):
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / split / "*.exr")))
    visible = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)
    gt_logs = torch.stack([load_gauge_log(f, device) for f in files])  # [B, H, W, 3]
    targets = gt_logs.reshape(len(files), -1, 3)[:, visible]
    u = fit_batch(renderer, targets, visible, device, FIT_STEPS, prior, lam)
    with torch.no_grad():
        pred_lin_all = torch.exp(renderer.render_log(u)).reshape(len(files), H, W, 3)
    per_image: List[Dict[str, Dict[str, float]]] = []
    for i in range(len(files)):
        gt_lin = torch.exp(gt_logs[i]) - EPS
        pred_lin = torch.nan_to_num(pred_lin_all[i], nan=0.0, posinf=1e6,
                                    neginf=0.0).clamp(0.0, 1e6)
        # Table convention (matches evaluate_run): own-quantile exposures.
        gt_ldr = linear_to_sRGB(gt_lin, use_quantile=True)
        pred_ldr = linear_to_sRGB(pred_lin, use_quantile=True)
        per_image.append({
            "full": region_metrics(model, pred_lin, pred_lin, gt_lin,
                                   pred_ldr, gt_ldr, "full", "none"),
            "visible": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr, mask),
            "hidden": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr,
                                            1.0 - mask),
        })
    averaged = {
        region: {m: float(torch.nanmean(torch.tensor(
            [img[region][m] for img in per_image]))) for m in METRICS}
        for region in REGIONS
    }
    return averaged, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fov-h", type=float, default=90.0)
    parser.add_argument("--fov-v", type=float, default=60.0)
    parser.add_argument("--refit-prior", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "outpaint_frustum90x60_sg_prior")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)
    device = args.device

    _, _, model = load_model(MODEL_DIRS["sg"][300], device=device)
    model.eval()
    ray_bundle = equirect_ray_bundle(device, idx=0, height=H)
    renderer = BatchedSG(model.field, ray_bundle.directions)
    mask = _frustum_mask(H, W, args.fov_h, args.fov_v)

    if PRIOR_PATH.exists() and not args.refit_prior:
        prior = torch.load(PRIOR_PATH)
        print(f"[prior] loaded cache {PRIOR_PATH}")
    else:
        prior = fit_training_prior(renderer, device)
        PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prior, PRIOR_PATH)
        print(f"[prior] saved {PRIOR_PATH}")
    prior["mu"] = prior["mu"].to(device)
    prior["sigma_inv"] = torch.linalg.inv(prior["sigma"]).to(device)

    print("[tune] lambda sweep on val (hidden-region LDR PSNR)")
    best_lam, best_val, sweep = None, -1e9, {}
    for lam in LAMBDAS:
        val_metrics, _ = evaluate_split("val", renderer, mask, prior, lam,
                                        model, device)
        hidden_ldr = val_metrics["hidden"]["psnr_ldr"]
        sweep[lam] = {r: {k: val_metrics[r][k] for k in ("psnr_ldr", "psnr_hdr")}
                      for r in REGIONS}
        print(f"  lambda={lam:<7g} hidden ldr={hidden_ldr:6.2f} "
              f"hdr={val_metrics['hidden']['psnr_hdr']:8.2f} "
              f"visible ldr={val_metrics['visible']['psnr_ldr']:6.2f}")
        if hidden_ldr > best_val:
            best_lam, best_val = lam, hidden_ldr
    print(f"[tune] chosen lambda={best_lam}")

    test_metrics, per_image = evaluate_split("test", renderer, mask, prior,
                                             best_lam, model, device)
    print(f"[test] hidden ldr={test_metrics['hidden']['psnr_ldr']:6.2f} "
          f"hdr={test_metrics['hidden']['psnr_hdr']:8.2f} "
          f"visible ldr={test_metrics['visible']['psnr_ldr']:6.2f} "
          f"full ldr={test_metrics['full']['psnr_ldr']:6.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "sg_prior",
        "lambda": best_lam,
        "lambda_sweep_val": {str(k): v for k, v in sweep.items()},
        "prior_cache": str(PRIOR_PATH),
        "num_train": prior.get("num_train"),
        "noise_var": prior.get("noise_var"),
        "fit_steps": FIT_STEPS,
        "metrics": test_metrics,
        "per_image": per_image,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [("model", "region", "num_eval_images", *METRICS)]
    for region in REGIONS:
        rows.append(("sg_prior", region, len(per_image),
                     *(f"{test_metrics[region][m]:.6f}" for m in METRICS)))
    with args.output.with_suffix(".csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[saved] {args.output}.json / .csv")


if __name__ == "__main__":
    main()
