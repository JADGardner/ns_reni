"""Frustum completion from LDR observations (the online-photo setting).

The main completion benchmark fits log-HDR pixels. In most applications the
observation is a tone-mapped, clipped LDR image, so this benchmark fits every
model to a clipped sRGB rendering of the visible frustum and scores the
decoded prediction against the true HDR. Protocol, all in the fixed gauge
(GT p99 luminance = 1), 64x128:

- Target: linear_to_sRGB(gauge linear, q = GT p98, clamp) on the visible
  pixels; the exposure is known and shared, and roughly the top 2% of pixels
  (sun and bright sky) clip.
- Fit: each model's prediction is decoded to linear radiance, tone-mapped
  with the same exposure, clamped, and fitted with the standard
  10*mse + cosine loss. The clamp gives censoring semantics: a clipped pixel
  only asks the prediction to reach the clip level.
- Rows: SH/SG Gaussian priors (iterative MAP, lambda re-tuned on val),
  original RENI, published RENI++ (joint log-exposure scalar, since LDR
  fixes no absolute scale for a scale-free model), and the two-bracket
  channelwise and joint-GS models (native fixed gauge).
- Scoring: identical to the main benchmark (HDR vs gauge GT; LDR under the
  own-quantile convention; full-sphere LPIPS and peak-direction error).

    PYTHONPATH=.:scripts/figures python scripts/figures/eval_outpaint_ldrfit.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

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
from eval_outpaint_sg_prior import BatchedSG
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import apply_fixed_gauge, two_bracket_to_linear

OUT_DIR = REPO_ROOT / "outputs" / "evaluations"
SH_PRIOR_PATH = REPO_ROOT / "outputs" / "reni" / "sh_gaussian_prior_9th.pt"
SG_PRIOR_PATH = REPO_ROOT / "outputs" / "reni" / "sg_gaussian_prior_300.pt"
H, W = 64, 128
FIT_STEPS = 600
SG_FIT_STEPS = 2400
FIT_LR = 1e-2
SH_LAMBDAS = (1.0, 3.0, 10.0, 30.0)
SG_LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0)

ROWS = ("sh", "sh_prior", "sg", "sg_prior", "reni_old", "reni_pp",
        "logdom_1cyc", "logdom_std_100k", "logdom_2cyc", "w3_2cyc",
        "vnjoint_2cyc", "vnjoint_ortho_2cyc")


def load_gauge_lin(path: str, device: str) -> torch.Tensor:
    import numpy as np
    import pyexr

    img = pyexr.read(path).astype("float32")[..., :3]
    finite = np.isfinite(img)
    if not finite.all():
        img[~finite] = img[finite].max()
    img[img <= 0] = img[img > 0].min()
    img = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None],
                        size=(H, W), mode="bilinear")[0].permute(1, 2, 0)
    return apply_fixed_gauge(img.to(device))


def ldr_target(gauge_lin: torch.Tensor):
    """Known-exposure clipped sRGB observation + the exposure quantile."""
    q = torch.quantile(gauge_lin.flatten(), 0.98)
    return linear_to_sRGB(gauge_lin, q=q, clamp=True), q


def ldr_loss(pred_lin_vis: torch.Tensor, target_vis: torch.Tensor,
             q: torch.Tensor) -> torch.Tensor:
    pred_srgb = linear_to_sRGB(pred_lin_vis.clamp_min(0.0), q=q, clamp=True)
    mse = torch.nn.functional.mse_loss(pred_srgb, target_vis)
    cosine = 1 - torch.nn.functional.cosine_similarity(
        pred_srgb, target_vis, dim=-1).mean()
    return 10.0 * mse + cosine


def score_image(model, pred_lin: torch.Tensor, gt_lin: torch.Tensor,
                mask: torch.Tensor) -> Dict:
    pred_lin = torch.nan_to_num(pred_lin, nan=0.0, posinf=1e6,
                                neginf=0.0).clamp(0.0, 1e6)
    gt_ldr = linear_to_sRGB(gt_lin, use_quantile=True)
    pred_ldr = linear_to_sRGB(pred_lin, use_quantile=True)
    metrics = {
        "full": region_metrics(model, pred_lin, pred_lin, gt_lin,
                               pred_ldr, gt_ldr, "full", "none"),
        "visible": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr, mask),
        "hidden": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr,
                                        1.0 - mask),
    }
    # Visible-region HDR is the new question here (LDR-to-HDR lifting);
    # region_metrics_masked already reports psnr_hdr for it.
    return metrics


def fit_learnt(model, gauge_lin, target, q, visible, ray_bundle, device,
               fit_exposure: bool):
    """Latent fit through the LDR loss; optionally a joint log-exposure."""
    field = model.field
    z = torch.zeros(1, field.latent_dim, 3, device=device)
    if (field.config.invariant_function in ("VNCanonical", "VNJoint")
            and getattr(field.config, "canonical_frame_orthonormalise", False)):
        z = 1e-2 * torch.randn(1, field.latent_dim, 3, device=device)
    z = z.requires_grad_(True)
    params = [z]
    log_s = torch.zeros((), device=device, requires_grad=True)
    if fit_exposure:
        params.append(log_s)
    optimiser = torch.optim.Adam(params, lr=FIT_LR)
    fit_samples = model.create_ray_samples(
        ray_bundle.origins[visible], ray_bundle.directions[visible],
        ray_bundle.camera_indices[visible])
    target_vis = target.reshape(-1, 3)[visible]
    for _ in range(FIT_STEPS):
        optimiser.zero_grad()
        latents = z.repeat(fit_samples.shape[0], 1, 1)
        out = field.forward(fit_samples, rotation=None,
                            latent_codes=latents)[RENIFieldHeadNames.RGB]
        if model.two_bracket:
            pred_lin = two_bracket_to_linear(
                out, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
        else:
            pred_lin = field.unnormalise(out)
        if fit_exposure:
            pred_lin = pred_lin * torch.exp(log_s)
        ldr_loss(pred_lin, target_vis, q).backward()
        optimiser.step()
    return z.detach(), float(log_s.detach())


@torch.no_grad()
def decode_learnt(model, z, log_s, ray_bundle, device):
    field = model.field
    chunks = []
    for start in range(0, ray_bundle.origins.shape[0], 65536):
        end = start + 65536
        samples = model.create_ray_samples(
            ray_bundle.origins[start:end], ray_bundle.directions[start:end],
            ray_bundle.camera_indices[start:end])
        out = field.forward(samples, rotation=None,
                            latent_codes=z.repeat(samples.shape[0], 1, 1))[RENIFieldHeadNames.RGB]
        if model.two_bracket:
            lin = two_bracket_to_linear(
                out, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
        else:
            lin = field.unnormalise(out)
        chunks.append(lin)
    pred = torch.cat(chunks, dim=0).reshape(H, W, 3)
    import math
    return pred * math.exp(log_s)


LEARNT_DIRS = {
    "reni_old": ("key", "reni_old"),
    "reni_pp": ("key", "reni_pp"),
    "w3_2cyc": ("key", "two_bracket_w3_2cyc"),
    "vnjoint_ortho_2cyc": ("key", "vnjoint_ortho_2cyc"),
    # Reset-study shims (paths as in the prioroff eval's runs dict).
    "logdom_1cyc": ("path", "/workspace/phd/outputs/reni/logdomain_2cyc_step50000"),
    "logdom_std_100k": ("path", "/workspace/phd/outputs/reni/logdomain_std_step100000"),
    "logdom_2cyc": ("path", "/workspace/phd/outputs/reni/logdomain_2cyc_step100000"),
    "vnjoint_2cyc": ("path", "/workspace/phd/outputs/reni/vnjoint_ldrw3_step100000"),
}


def eval_learnt(key: str, files, mask, visible, ray_bundle, model_metrics_host,
                device):
    kind, spec = LEARNT_DIRS[key]
    model_dir = MODEL_DIRS[spec][100] if kind == "key" else Path(spec)
    _, _, model = load_model(model_dir, device=device)
    model.eval()
    fit_exposure = not getattr(model, "two_bracket", False)
    per_image = []
    for i, f in enumerate(files):
        gauge_lin = load_gauge_lin(f, device)
        target, q = ldr_target(gauge_lin)
        z, log_s = fit_learnt(model, gauge_lin, target, q, visible, ray_bundle,
                              device, fit_exposure)
        pred_lin = decode_learnt(model, z, log_s, ray_bundle, device)
        per_image.append(score_image(model_metrics_host, pred_lin, gauge_lin, mask))
        print(f"  [{key}] idx={i:02d} hidden hdr="
              f"{per_image[-1]['hidden']['psnr_hdr']:6.2f} "
              f"visible hdr={per_image[-1]['visible']['psnr_hdr']:6.2f}")
    del model
    torch.cuda.empty_cache()
    return per_image


def eval_sh_prior(files, mask, visible, ray_bundle, model_metrics_host, device,
                  lam: float):
    from eval_outpaint_sh_prior import design_matrix

    prior = torch.load(SH_PRIOR_PATH)
    mu = prior["mu"].to(device).float()
    sigma_inv = torch.linalg.inv(prior["sigma"]).to(device).float()
    basis = design_matrix(ray_bundle, 9, device)
    basis_vis = basis[visible]
    per_image = []
    for f in files:
        gauge_lin = load_gauge_lin(f, device)
        target, q = ldr_target(gauge_lin)
        target_vis = target.reshape(-1, 3)[visible]
        init = mu if lam > 0 else torch.zeros_like(mu)
        c = init.reshape(-1, 3).clone().requires_grad_(True)
        optimiser = torch.optim.Adam([c], lr=FIT_LR)
        for _ in range(FIT_STEPS):
            optimiser.zero_grad()
            pred_lin_vis = torch.exp(basis_vis @ c)
            loss = ldr_loss(pred_lin_vis, target_vis, q)
            if lam > 0:
                dev = c.reshape(-1) - mu
                loss = loss + lam * (dev @ sigma_inv @ dev) / dev.shape[0]
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            pred_lin = torch.exp(basis @ c).reshape(H, W, 3)
        per_image.append(score_image(model_metrics_host, pred_lin, gauge_lin, mask))
    return per_image


def eval_sg_prior(files, mask, visible, ray_bundle, model_metrics_host, device,
                  lam: float, sg_field):
    prior = torch.load(SG_PRIOR_PATH)
    mu = prior["mu"].to(device)
    sigma_inv = torch.linalg.inv(prior["sigma"]).to(device)
    renderer = BatchedSG(sg_field, ray_bundle.directions)
    gauge = torch.stack([load_gauge_lin(f, device) for f in files])
    targets, qs = [], []
    for i in range(len(files)):
        t, q = ldr_target(gauge[i])
        targets.append(t.reshape(-1, 3)[visible])
        qs.append(q)
    targets = torch.stack(targets)
    u = renderer.init_u(len(files), device).requires_grad_(True)
    optimiser = torch.optim.Adam([u], lr=FIT_LR)
    for _ in range(SG_FIT_STEPS):
        optimiser.zero_grad()
        out = renderer.render_log(u)[:, visible]
        loss = 0.0
        pred_lin = torch.exp(out)
        for i in range(len(files)):
            loss = loss + ldr_loss(pred_lin[i], targets[i], qs[i])
        loss = loss / len(files)
        if lam > 0:
            dev = u.reshape(len(files), -1).double() - mu[None]
            maha = torch.einsum("bi,ij,bj->b", dev, sigma_inv, dev)
            loss = loss + lam * (maha.mean() / dev.shape[1]).float()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        pred_all = torch.exp(renderer.render_log(u.detach())).reshape(
            len(files), H, W, 3)
    return [score_image(model_metrics_host, pred_all[i], gauge[i], mask)
            for i in range(len(files))]


def average(per_image):
    return {region: {m: float(torch.nanmean(torch.tensor(
        [img[region][m] for img in per_image]))) for m in METRICS}
        for region in REGIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fov-h", type=float, default=90.0)
    parser.add_argument("--fov-v", type=float, default=60.0)
    parser.add_argument("--rows", nargs="+", default=list(ROWS))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "outpaint_frustum90x60_ldrfit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)
    device = args.device

    # Metric modules host (ssim/lpips); any RENIModel serves.
    _, _, metrics_host = load_model(MODEL_DIRS["sh"]["9th"], device=device)
    metrics_host.eval()
    ray_bundle = equirect_ray_bundle(device, idx=0, height=H)
    mask = _frustum_mask(H, W, args.fov_h, args.fov_v)
    visible = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)

    def files_for(split):
        fs = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / split / "*.exr")))
        return fs[:args.max_images] if args.max_images else fs

    results, chosen = {}, {}
    for key in args.rows:
        print(f"[eval] {key}")
        if key == "sh":
            per_image = eval_sh_prior(files_for("test"), mask, visible,
                                      ray_bundle, metrics_host, device, 0.0)
        elif key == "sg":
            _, _, sg_model = load_model(MODEL_DIRS["sg"][300], device=device)
            per_image = eval_sg_prior(files_for("test"), mask, visible,
                                      ray_bundle, metrics_host, device, 0.0,
                                      sg_model.field)
            del sg_model
        elif key == "sh_prior":
            best = (None, -1e9)
            for lam in SH_LAMBDAS:
                v = average(eval_sh_prior(files_for("val"), mask, visible,
                                          ray_bundle, metrics_host, device, lam))
                print(f"  [tune sh_prior] lambda={lam:g} "
                      f"hidden hdr={v['hidden']['psnr_hdr']:6.2f} "
                      f"ldr={v['hidden']['psnr_ldr']:6.2f}")
                if v["hidden"]["psnr_ldr"] > best[1]:
                    best = (lam, v["hidden"]["psnr_ldr"])
            chosen[key] = best[0]
            per_image = eval_sh_prior(files_for("test"), mask, visible,
                                      ray_bundle, metrics_host, device, best[0])
        elif key == "sg_prior":
            _, _, sg_model = load_model(MODEL_DIRS["sg"][300], device=device)
            best = (None, -1e9)
            for lam in SG_LAMBDAS:
                v = average(eval_sg_prior(files_for("val"), mask, visible,
                                          ray_bundle, metrics_host, device, lam,
                                          sg_model.field))
                print(f"  [tune sg_prior] lambda={lam:g} "
                      f"hidden hdr={v['hidden']['psnr_hdr']:6.2f} "
                      f"ldr={v['hidden']['psnr_ldr']:6.2f}")
                if v["hidden"]["psnr_ldr"] > best[1]:
                    best = (lam, v["hidden"]["psnr_ldr"])
            chosen[key] = best[0]
            per_image = eval_sg_prior(files_for("test"), mask, visible,
                                      ray_bundle, metrics_host, device, best[0],
                                      sg_model.field)
            del sg_model
        else:
            per_image = eval_learnt(key, files_for("test"), mask, visible,
                                    ray_bundle, metrics_host, device)
        results[key] = {"metrics": average(per_image), "per_image": per_image,
                        "num_eval_images": len(per_image)}
        m = results[key]["metrics"]
        print(f"[done] {key}: hidden hdr={m['hidden']['psnr_hdr']:6.2f} "
              f"ldr={m['hidden']['psnr_ldr']:6.2f} "
              f"visible hdr={m['visible']['psnr_hdr']:6.2f} "
              f"lpips={m['full']['lpips_ldr']:.3f} "
              f"peak={m['full']['peak_angle_error_deg']:5.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"protocol": "ldr_fit_known_exposure_p98_clip",
         "chosen_lambdas": chosen, "models": results}, indent=2) + "\n")
    rows = [("model", "region", "num_eval_images", *METRICS)]
    for key, r in results.items():
        for region in REGIONS:
            rows.append((key, region, r["num_eval_images"],
                         *(f"{r['metrics'][region][m]:.6f}" for m in METRICS)))
    with args.output.with_suffix(".csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[saved] {args.output}.json / .csv")


if __name__ == "__main__":
    main()
