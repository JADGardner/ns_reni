"""Masked outpainting benchmark: fit eval latents on the VISIBLE half only.

Companion to eval_two_bracket_compare.py. Where that script refits each test
latent on the full envmap and measures full-image fidelity, this one fits the
latent on only the visible (right) half of each equirect test image with the
frozen decoder -- exactly the completion setting of scripts/figures/
fig_outpainting.py, but as a quantitative benchmark -- and then decodes the
full envmap and scores three regions separately: the full image, the visible
half, and the HIDDEN (left) half.

Scientific question: latent-reset training refits better under FULL supervision
(pre-reset checkpoints win). The hypothesis here is that under PARTIAL
supervision the ordering reverses on the hidden half, because that region is
completed by the learned prior/manifold rather than fitted to data.

Fit recipe follows fig_outpainting.py (~600 steps, Adam lr 1e-2, log-HDR domain
loss on the visible pixels, optional 1e-4 z^2 latent prior). The metric families
and decode/exposure conventions match eval_two_bracket_compare.py: standard
metrics (psnr/ssim/lpips in HDR and LDR) plus the peak-aware HDR metrics from
reni/utils/hdr_metrics.py. Scale-invariant (3-channel) models get the least-
squares log-domain scale the standard eval uses and median-ratio peak alignment;
fixed-gauge (two-bracket) models are scored raw with no peak alignment.

Run from the phd repo root via docker compose (never a subdirectory):

    docker compose run --rm research bash -c "cd /workspace/phd/code/ns_reni && \
        PYTHONPATH=.:scripts/figures python scripts/figures/eval_outpaint_compare.py \
        --run reni_pp=/home/james/model-storage/reni_paper_models/reni_plus_plus_models/latent_dim_100 \
        --run latent_reset_4cyc=outputs/reni_latent_reset_4_rerun \
        --prior off"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from _common import REPO_ROOT, equirect_ray_bundle, seed_all
from eval_latent_reset_compare import (
    _build_test_config,
    _eval_split_name,
    _latest_checkpoint,
    _load_decoder_state,
    resolve_run_dir,
)
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.hdr_metrics import compute_hdr_peak_metrics
from reni.utils.tonemap import two_bracket_to_linear

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluations"

REGIONS = ("full", "visible", "hidden")
STANDARD_METRICS = (
    "psnr_hdr",
    "ssim_hdr",
    "lpips_hdr",
    "psnr_ldr",
    "ssim_ldr",
    "lpips_ldr",
)
PEAK_METRICS = (
    "log_rmse_hdr",
    "lum_weighted_rmse_hdr",
    "peak_intensity_rel_error",
    "peak_angle_error_deg",
    "peak_argmax_angle_error_deg",
)
METRICS = STANDARD_METRICS + PEAK_METRICS
HIGHER_BETTER = {
    "psnr_hdr": True,
    "ssim_hdr": True,
    "lpips_hdr": False,
    "psnr_ldr": True,
    "ssim_ldr": True,
    "lpips_ldr": False,
    "log_rmse_hdr": False,
    "lum_weighted_rmse_hdr": False,
    "peak_intensity_rel_error": False,
    "peak_angle_error_deg": False,
    "peak_argmax_angle_error_deg": False,
}
# Directional peak metrics re-parameterise azimuth on a half-width crop, so they
# are only strictly meaningful on the full region (kept for completeness).
CROP_UNSAFE_METRICS = ("peak_angle_error_deg", "peak_argmax_angle_error_deg")

FIT_STEPS = 600
FIT_LR = 1e-2
PRIOR_WEIGHT = 1e-4
EPS = 1e-8


def parse_runs(values) -> Dict[str, Path]:
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run expects LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        runs[label] = Path(path)
    return runs


def _visible_mask(height: int, width: int) -> torch.Tensor:
    """[H, W] mask, 1 = visible. Right half visible, left half hidden."""
    mask = torch.zeros(height, width)
    mask[:, width // 2 :] = 1.0
    return mask


def _frustum_mask(height: int, width: int, fov_h_deg: float,
                  fov_v_deg: float) -> torch.Tensor:
    """[H, W] mask, 1 = visible. Central pinhole frustum: a camera looking at
    the ERP centre (lon = 0, lat = 0) with the given horizontal/vertical FoV.
    Pinhole constraints on the image plane: |x/z| <= tan(fov_h/2) and
    |y/z| <= tan(fov_v/2), which on the ERP grid become |tan(lon)| <= tan(a)
    and |tan(lat)| <= tan(b) cos(lon)."""
    lon = ((torch.arange(width) + 0.5) / width * 2.0 - 1.0) * math.pi
    lat = (0.5 - (torch.arange(height) + 0.5) / height) * math.pi
    lon = lon.unsqueeze(0).expand(height, width)
    lat = lat.unsqueeze(1).expand(height, width)
    tan_a = math.tan(math.radians(fov_h_deg) / 2.0)
    tan_b = math.tan(math.radians(fov_v_deg) / 2.0)
    front = lon.abs() < math.pi / 2.0
    in_h = lon.tan().abs() <= tan_a
    in_v = lat.tan().abs() <= tan_b * lon.cos()
    return (front & in_h & in_v).float()


def _region_slice(region: str, width: int) -> slice:
    if region == "full":
        return slice(0, width)
    if region == "visible":
        return slice(width // 2, width)
    if region == "hidden":
        return slice(0, width // 2)
    raise ValueError(f"Unknown region {region!r}")


def fit_latent_on_visible(
    model,
    gt_raw: torch.Tensor,
    mask: torch.Tensor,
    ray_bundle,
    device: str,
    prior_weight: float,
    steps: int = FIT_STEPS,
    lr: float = FIT_LR,
) -> torch.Tensor:
    """Fit z on the visible pixels with the frozen decoder (fig_outpainting recipe).

    gt_raw: [H, W, C] in the model's target domain (3ch log-HDR or 6ch two-bracket).
    mask:   [H, W], 1 = visible. Returns the fitted latent [1, latent_dim, 3].
    """
    channels = gt_raw.shape[-1]
    target = gt_raw.reshape(-1, channels).to(device)
    visible = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)
    if visible.numel() == 0:
        raise ValueError("Visible mask contains no pixels")

    z = torch.zeros(1, model.field.latent_dim, 3, device=device)
    if (
        model.field.config.invariant_function in ("VNCanonical", "VNJoint")
        and getattr(model.field.config, "canonical_frame_orthonormalise", False)
    ):
        # Z=0 is a gradient singularity of the orthonormalised frame
        # (Gram-Schmidt Jacobian ~1/||q||); start a hair off it.
        z = 1e-2 * torch.randn(1, model.field.latent_dim, 3, device=device)
    z = z.requires_grad_(True)
    optimiser = torch.optim.Adam([z], lr=lr)

    fit_samples = model.create_ray_samples(
        ray_bundle.origins[visible],
        ray_bundle.directions[visible],
        ray_bundle.camera_indices[visible],
    )
    tgt = target[visible]
    if model.two_bracket:
        gt_log = torch.log(
            two_bracket_to_linear(tgt, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log).clamp_min(0.0)
            + EPS
        )

    for _ in range(steps):
        optimiser.zero_grad()
        latents = z.repeat(fit_samples.shape[0], 1, 1)
        out = model.field.forward(fit_samples, rotation=None, latent_codes=latents)[RENIFieldHeadNames.RGB]
        if model.two_bracket:
            # Fit in log-HDR of the BLENDED linear reconstruction, not on the
            # raw brackets (matches fig_outpainting's log-HDR domain loss).
            pred_lin = two_bracket_to_linear(out, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
            pred_log = torch.log(pred_lin.clamp_min(0.0) + EPS)
            mse = torch.nn.functional.mse_loss(pred_log, gt_log)
            cosine = 1 - torch.nn.functional.cosine_similarity(pred_log, gt_log, dim=-1).mean()
        else:
            # 3-channel log-domain models: raw output is already log-HDR.
            mse = torch.nn.functional.mse_loss(out, tgt)
            cosine = 1 - torch.nn.functional.cosine_similarity(out, tgt, dim=-1).mean()
        prior = z.pow(2).mean()
        loss = 10.0 * mse + cosine + prior_weight * prior
        loss.backward()
        optimiser.step()

    return z.detach()


@torch.no_grad()
def decode_full_envmap(model, z: torch.Tensor, ray_bundle, height: int, width: int,
                       channels: int, chunk: int = 65536) -> torch.Tensor:
    """Decode the fitted latent over the full equirect grid -> [H, W, C] raw output."""
    chunks: List[torch.Tensor] = []
    for start in range(0, ray_bundle.origins.shape[0], chunk):
        end = start + chunk
        samples = model.create_ray_samples(
            ray_bundle.origins[start:end],
            ray_bundle.directions[start:end],
            ray_bundle.camera_indices[start:end],
        )
        latents = z.repeat(samples.shape[0], 1, 1)
        out = model.field.forward(samples, rotation=None, latent_codes=latents)[RENIFieldHeadNames.RGB]
        chunks.append(out)
    return torch.cat(chunks, dim=0).reshape(height, width, channels)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """PSNR with data_range=1.0 over all elements (matches torchmetrics config)."""
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def _lpips_normed(model, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """HDR LPIPS: per-image min/max normalise each crop to [0,1] (as RENIModel does)."""
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        return torch.tensor(float("nan"))
    pred_min, pred_max = pred.min(), pred.max()
    gt_min, gt_max = gt.min(), gt.max()
    if pred_max - pred_min <= 1e-8 or gt_max - gt_min <= 1e-8:
        return torch.tensor(float("nan"))
    pred_norm = ((pred - pred_min) / (pred_max - pred_min)).unsqueeze(0).permute(0, 3, 1, 2)
    gt_norm = ((gt - gt_min) / (gt_max - gt_min)).unsqueeze(0).permute(0, 3, 1, 2)
    return model.lpips(pred_norm, gt_norm)


def _chw(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0).permute(0, 3, 1, 2)


def region_metrics(
    model,
    pred_lin: torch.Tensor,
    pred_lin_unaligned: torch.Tensor,
    gt_lin: torch.Tensor,
    pred_ldr: torch.Tensor,
    gt_ldr: torch.Tensor,
    region: str,
    peak_alignment: str,
) -> Dict[str, float]:
    """All metric families on a single region (contiguous column crop)."""
    width = gt_lin.shape[1]
    sl = _region_slice(region, width)
    pl, plu, gl = pred_lin[:, sl], pred_lin_unaligned[:, sl], gt_lin[:, sl]
    pldr, gldr = pred_ldr[:, sl], gt_ldr[:, sl]

    out: Dict[str, float] = {}
    # Pixel metrics: masking a contiguous column range == cropping.
    out["psnr_hdr"] = float(_psnr(pl, gl))
    out["psnr_ldr"] = float(_psnr(pldr, gldr))
    # SSIM/LPIPS on the crop.
    out["ssim_hdr"] = float(model.ssim(preds=_chw(pl), target=_chw(gl)))
    out["ssim_ldr"] = float(model.ssim(preds=_chw(pldr), target=_chw(gldr)))
    out["lpips_hdr"] = float(_lpips_normed(model, pl, gl))
    out["lpips_ldr"] = float(model.lpips(_chw(pldr), _chw(gldr)))
    # Peak-aware HDR metrics on the (unaligned) crop; the metric fn applies its
    # own robust alignment. Directional peak metrics are only strictly valid on
    # the full region (half crops re-parameterise azimuth) but are kept.
    peak = compute_hdr_peak_metrics(plu, gl, alignment=peak_alignment)
    for key in PEAK_METRICS:
        out[key] = float(peak[key])
    return out


def region_metrics_masked(
    pred_lin: torch.Tensor,
    gt_lin: torch.Tensor,
    pred_ldr: torch.Tensor,
    gt_ldr: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """PSNR-only metrics on an arbitrary pixel mask (frustum regions).

    SSIM/LPIPS and the directional peak metrics need contiguous images, so
    for masked regions they are reported as NaN; nanmean averaging and the
    summary printer already handle that."""
    sel = mask.reshape(-1).bool().to(pred_lin.device)
    out = {m: float("nan") for m in METRICS}
    out["psnr_hdr"] = float(_psnr(pred_lin.reshape(-1, 3)[sel],
                                  gt_lin.reshape(-1, 3)[sel]))
    out["psnr_ldr"] = float(_psnr(pred_ldr.reshape(-1, 3)[sel],
                                  gt_ldr.reshape(-1, 3)[sel]))
    return out


def evaluate_baseline(
    family: str,
    key,
    device: str,
    seed: int,
    max_images: Optional[int],
    mask_mode: str,
    fov_h: float,
    fov_v: float,
    ldr_exposure: str = "gtq",
) -> Dict[str, Any]:
    """Frustum completion for the per-image baselines (SH, SG, original RENI).

    Parameters are fitted from their native init to the VISIBLE pixels only,
    with the same step/lr budget as the latent fits, using a log-HDR loss on
    linear radiance in the fixed gauge (ground-truth p99 luminance = 1); the
    prediction is scored raw in that gauge. SH/SG carry no completion prior,
    so nothing constrains the hidden region."""
    import glob as _glob

    import pyexr
    import torch.nn.functional as F
    from _common import MODEL_DIRS, load_model
    from reni.utils.tonemap import apply_fixed_gauge

    seed_all(seed)
    _, _, model = load_model(MODEL_DIRS[family][key], device=device)
    model.eval()
    field = model.field
    files = sorted(_glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / "test" / "*.exr")))
    if max_images is not None:
        files = files[:max_images]
    height, width = 64, 128
    ray_bundle = equirect_ray_bundle(device, idx=0, height=height)
    if mask_mode == "frustum":
        mask = _frustum_mask(height, width, fov_h, fov_v)
    else:
        mask = _visible_mask(height, width)
    visible = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)
    fit_samples = model.create_ray_samples(
        ray_bundle.origins[visible],
        ray_bundle.directions[visible],
        ray_bundle.camera_indices[visible],
    )

    is_sh = hasattr(field, "num_sh_coeffs")
    is_sg = hasattr(field, "sg_num")
    if is_sh:
        from reni.illumination_fields.sh_illumination_field import shEvaluate
        d_vis = ray_bundle.directions[visible]
        sh_basis_vis = shEvaluate(
            torch.acos(d_vis[:, 2]), torch.atan2(d_vis[:, 0], d_vis[:, 1]),
            field.spherical_harmonic_order, device=device)  # [n_vis, n_coeffs]
    if is_sg:
        sg_theta_c = field.theta_center_train[:1].to(device)  # [1, sg_num, 1]
        sg_phi_c = field.phi_center_train[:1].to(device)

    def sg_latents(u: torch.Tensor) -> torch.Tensor:
        """Raw SG parameters -> the latent_codes convention of renderSG,
        mirroring deparameterize (exp amplitudes/sharpness, grid-centred
        tanh angles); exp keeps the log-domain mixture positive."""
        w_u, t_u, p_u, l_u = torch.split(u, [3, 1, 1, 1], dim=2)
        return torch.cat([
            torch.exp(w_u),
            field.theta_range * torch.tanh(t_u) + sg_theta_c,
            field.phi_range * torch.tanh(p_u) + sg_phi_c,
            torch.exp(l_u),
        ], dim=2)

    def to_raw_domain(lin: torch.Tensor) -> torch.Tensor:
        """Fixed-gauge linear radiance -> the field's raw output domain
        (inverse of field.unnormalise)."""
        x = lin
        if field.log_domain:
            x = torch.log(x + EPS)
        if not field.min_max.dtype == torch.bool:
            min_val, max_val = field.min_max
            x = 2.0 * (x - min_val) / (max_val - min_val) - 1.0
        return x

    print(f"[eval] baseline {family}/{key}: images={len(files)}, "
          f"mask={mask_mode}, fit={FIT_STEPS}x{FIT_LR}, domain=log fixed-gauge linear")
    per_image: List[Dict[str, Dict[str, float]]] = []
    for idx, path in enumerate(files):
        img = pyexr.read(path).astype("float32")[..., :3]
        img[img <= 0] = img[img > 0].min()
        img = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None],
                            size=(height, width), mode="bilinear")[0].permute(1, 2, 0)
        gt_lin = apply_fixed_gauge(img.to(device))
        # Fit in the field's raw domain (log-HDR for these models, matching
        # the "3-channel log-domain" branch of fit_latent_on_visible).
        gt_raw_vis = to_raw_domain(gt_lin.reshape(-1, 3)[visible])

        if is_sh:
            # Exact minimum-norm least squares on the visible pixels: the
            # most charitable fit the basis allows (CPU gelsd for min-norm).
            sol = torch.linalg.lstsq(
                sh_basis_vis.cpu().double(), gt_raw_vis.cpu().double(),
                driver="gelsd").solution
            z = sol.float().to(device)[None]  # [1, n_coeffs, 3]
        else:
            if is_sg:
                # Raw params as in setup_param: zero amplitudes/angle offsets,
                # log sharpness at the grid scale; optimised past the latent
                # budget since the raw mixture converges more slowly.
                u = field.setup_param(
                    1, field.sg_num, field.sg_row, field.config.channel_dim
                ).data.view(1, field.sg_num, 6).to(device).requires_grad_(True)
                steps = 4 * FIT_STEPS
            else:
                u = torch.zeros(1, field.latent_dim, 3, device=device,
                                requires_grad=True)
                steps = FIT_STEPS
            optimiser = torch.optim.Adam([u], lr=FIT_LR)
            for _ in range(steps):
                optimiser.zero_grad()
                z_step = sg_latents(u) if is_sg else u
                latents = z_step.repeat(fit_samples.shape[0], 1, 1)
                out = field.forward(fit_samples, rotation=None,
                                    latent_codes=latents)[RENIFieldHeadNames.RGB]
                mse = torch.nn.functional.mse_loss(out, gt_raw_vis)
                cosine = 1 - torch.nn.functional.cosine_similarity(
                    out, gt_raw_vis, dim=-1).mean()
                (10.0 * mse + cosine).backward()
                optimiser.step()
            z = (sg_latents(u) if is_sg else u).detach()
        with torch.no_grad():
            chunks: List[torch.Tensor] = []
            for start in range(0, ray_bundle.origins.shape[0], 65536):
                end = start + 65536
                samples = model.create_ray_samples(
                    ray_bundle.origins[start:end],
                    ray_bundle.directions[start:end],
                    ray_bundle.camera_indices[start:end],
                )
                out = field.forward(samples, rotation=None,
                                    latent_codes=z.repeat(samples.shape[0], 1, 1))[RENIFieldHeadNames.RGB]
                chunks.append(field.unnormalise(out))
            pred_lin = torch.cat(chunks, dim=0).reshape(height, width, 3)
            # Unconstrained basis extrapolation overflows exp outside the fit
            # region (SH especially); cap at 1e6 gauge units (GT p99 = 1) so
            # the metrics stay finite while remaining abysmal.
            pred_lin = torch.nan_to_num(
                pred_lin, nan=0.0, posinf=1e6, neginf=0.0).clamp(0.0, 1e6)

        # gtq: one ground-truth-derived exposure for both images, for rows
        # whose extrapolation diverges (a prediction-derived quantile is
        # meaningless there). predq: own-quantile exposures, the convention
        # of the learnt rows (evaluate_run).
        gt_ldr = linear_to_sRGB(gt_lin, use_quantile=True)
        if ldr_exposure == "gtq":
            q_gt = torch.quantile(gt_lin.flatten(), 0.98)
            gt_ldr = linear_to_sRGB(gt_lin, q=q_gt)
            pred_ldr = linear_to_sRGB(pred_lin, q=q_gt)
        else:
            pred_ldr = linear_to_sRGB(pred_lin, use_quantile=True)
        image_metrics = {
            "full": region_metrics(model, pred_lin, pred_lin, gt_lin,
                                   pred_ldr, gt_ldr, "full", "none"),
            "visible": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr, mask),
            "hidden": region_metrics_masked(pred_lin, gt_lin, pred_ldr, gt_ldr, 1.0 - mask),
        }
        per_image.append(image_metrics)
        print(f"  [{family}/{key}] idx={idx:02d} "
              f"hidden psnr_hdr={image_metrics['hidden']['psnr_hdr']:6.2f} "
              f"psnr_ldr={image_metrics['hidden']['psnr_ldr']:6.2f} | "
              f"visible psnr_ldr={image_metrics['visible']['psnr_ldr']:6.2f}")

    averaged: Dict[str, Dict[str, float]] = {}
    for region in REGIONS:
        averaged[region] = {}
        for metric in METRICS:
            values = torch.tensor([img[region][metric] for img in per_image])
            averaged[region][metric] = float(torch.nanmean(values))
    result = {
        "label": f"{family}_{key}",
        "family": family,
        "model_key": str(key),
        "num_eval_images": len(files),
        "two_bracket": False,
        "peak_alignment": "none",
        "fit_domain": "log_fixed_gauge_linear",
        "metrics": averaged,
        "per_image": per_image,
    }
    del model
    torch.cuda.empty_cache()
    return result


def evaluate_run(
    label: str,
    run_path: Path,
    data_root: Path,
    device: str,
    latent_steps: Optional[int],
    prior_weight: float,
    seed: int,
    max_images: Optional[int],
    mask_mode: str = "halves",
    fov_h: float = 90.0,
    fov_v: float = 60.0,
) -> Dict[str, Any]:
    seed_all(seed)
    run_dir = resolve_run_dir(run_path)
    checkpoint = _latest_checkpoint(run_dir)
    print(f"[load] {label}: {checkpoint}")

    model_config = _build_test_config(run_dir, data_root, latent_steps)
    pipeline = model_config.pipeline.setup(
        device=device, test_mode="test", world_size=1, local_rank=0, grad_scaler=None,
    )
    load_stats = _load_decoder_state(pipeline, checkpoint, device)
    model = pipeline.model
    model.to(device)
    model.eval()

    two_bracket = bool(getattr(model, "two_bracket", False))
    scale_inv_eval = model.config.loss_inclusions["scale_inv_loss"] in [True, "eval", "both"]
    peak_alignment = "median_ratio" if scale_inv_eval else "none"
    convert_to_ldr = bool(model.metadata["convert_to_ldr"])

    split = _eval_split_name(pipeline.datamanager)
    num_eval = len(pipeline.datamanager.eval_dataset)
    if max_images is not None:
        num_eval = min(num_eval, max_images)
    print(f"[eval] {label}: split={split}, images={num_eval}, two_bracket={two_bracket}, "
          f"scale_inv={scale_inv_eval}, prior_weight={prior_weight}, load={load_stats}")

    per_image: List[Dict[str, Dict[str, float]]] = []
    for idx in range(num_eval):
        batch = pipeline.datamanager.eval_dataset[idx]
        gt_raw = batch["image"]
        if gt_raw.dim() == 4:
            gt_raw = gt_raw[0]
        height, width, channels = gt_raw.shape
        gt_raw = gt_raw.to(device)

        if mask_mode == "frustum":
            mask = _frustum_mask(height, width, fov_h, fov_v)
        else:
            mask = _visible_mask(height, width)
        ray_bundle = equirect_ray_bundle(device, idx=0, height=height)

        z = fit_latent_on_visible(model, gt_raw, mask, ray_bundle, device, prior_weight)
        pred_raw = decode_full_envmap(model, z, ray_bundle, height, width, channels)

        # Decode/exposure path identical to RENIModel.get_image_metrics_and_images.
        if scale_inv_eval:
            scale = (gt_raw * pred_raw).sum() / (pred_raw * pred_raw).sum()
            pred_raw_aligned = scale * pred_raw
        else:
            pred_raw_aligned = pred_raw
        gt_lin = model._to_linear_hdr(gt_raw)
        pred_lin = model._to_linear_hdr(pred_raw_aligned)
        pred_lin_unaligned = model._to_linear_hdr(pred_raw)

        if convert_to_ldr:
            gt_ldr, pred_ldr = gt_lin, pred_lin
        else:
            gt_ldr = linear_to_sRGB(gt_lin, use_quantile=True)
            pred_ldr = linear_to_sRGB(pred_lin, use_quantile=True)

        if mask_mode == "frustum":
            image_metrics = {
                "full": region_metrics(
                    model, pred_lin, pred_lin_unaligned, gt_lin, pred_ldr,
                    gt_ldr, "full", peak_alignment,
                ),
                "visible": region_metrics_masked(
                    pred_lin, gt_lin, pred_ldr, gt_ldr, mask),
                "hidden": region_metrics_masked(
                    pred_lin, gt_lin, pred_ldr, gt_ldr, 1.0 - mask),
            }
            # Exposure-convention probe: LDR with a ground-truth-derived
            # quantile for the prediction too (the baseline rows' convention).
            if not convert_to_ldr:
                q_gt = torch.quantile(gt_lin.flatten(), 0.98)
                pred_ldr_gtq = linear_to_sRGB(pred_lin, q=q_gt)
                image_metrics["gtq_ldr"] = {
                    "visible": region_metrics_masked(
                        pred_lin, gt_lin, pred_ldr_gtq, gt_ldr, mask)["psnr_ldr"],
                    "hidden": region_metrics_masked(
                        pred_lin, gt_lin, pred_ldr_gtq, gt_ldr, 1.0 - mask)["psnr_ldr"],
                }
        else:
            image_metrics = {
                region: region_metrics(
                    model, pred_lin, pred_lin_unaligned, gt_lin, pred_ldr, gt_ldr, region, peak_alignment,
                )
                for region in REGIONS
            }
        per_image.append(image_metrics)
        print(f"  [{label}] idx={idx:02d} "
              f"hidden psnr_hdr={image_metrics['hidden']['psnr_hdr']:6.2f} "
              f"psnr_ldr={image_metrics['hidden']['psnr_ldr']:6.2f} "
              f"lpips_hdr={image_metrics['hidden']['lpips_hdr']:.3f} | "
              f"full psnr_ldr={image_metrics['full']['psnr_ldr']:6.2f}")

    # Average per region/metric over images (nan-aware, matching the eval spirit).
    averaged: Dict[str, Dict[str, float]] = {}
    for region in REGIONS:
        averaged[region] = {}
        for metric in METRICS:
            values = torch.tensor([img[region][metric] for img in per_image])
            averaged[region][metric] = float(torch.nanmean(values))

    result = {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "split": split,
        "num_eval_images": num_eval,
        "two_bracket": two_bracket,
        "scale_inv_eval": scale_inv_eval,
        "peak_alignment": peak_alignment,
        "convert_to_ldr": convert_to_ldr,
        "prior_weight": prior_weight,
        "load_stats": load_stats,
        "metrics": averaged,
        "per_image": per_image,
    }
    del pipeline, model
    torch.cuda.empty_cache()
    return result


def _write_outputs(output_stem: Path, results: Dict[str, Any]) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    rows = [("model", "region", "num_eval_images", *METRICS)]
    for key, result in results["models"].items():
        for region in REGIONS:
            metrics = result["metrics"][region]
            rows.append(
                (key, region, result["num_eval_images"],
                 *(f"{metrics[m]:.6f}" for m in METRICS))
            )
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")


def _print_summary(results: Dict[str, Any]) -> None:
    print(f"\nOutpainting summary (mask={results['mask']}; prior_weight="
          f"{results['prior_weight']})")
    name_width = max(12, *(len(k) for k in results["models"]))
    cols = [
        ("hidden.psnr_hdr", "hid_psnr_hdr"),
        ("hidden.psnr_ldr", "hid_psnr_ldr"),
        ("hidden.lpips_hdr", "hid_lpips_hdr"),
        ("visible.psnr_ldr", "vis_psnr_ldr"),
        ("full.psnr_ldr", "full_psnr_ldr"),
        ("full.psnr_hdr", "full_psnr_hdr"),
    ]
    header = f"{'model':{name_width}s} " + " ".join(f"{c[1]:>14s}" for c in cols)
    print(header)
    print("-" * len(header))
    for key, result in results["models"].items():
        cells = []
        for region_metric, _ in cols:
            region, metric = region_metric.split(".")
            value = result["metrics"][region][metric]
            cells.append(f"{value:14.4f}" if not math.isnan(value) else f"{'nan':>14s}")
        print(f"{key:{name_width}s} " + " ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                        help="Run to evaluate; repeatable.")
    parser.add_argument("--ldr-exposure", choices=["gtq", "predq"], default="gtq",
                        help="Baseline LDR exposure: ground-truth-derived (gtq) "
                             "or own-quantile as in the learnt rows (predq).")
    parser.add_argument("--baseline", action="append", default=[], metavar="FAMILY=KEY",
                        help="Per-image baseline from _common.MODEL_DIRS (e.g. "
                             "sh=9th, sg=300, reni_old=100); repeatable.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "RENI_HDR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prior", choices=["on", "off"], default="off",
                        help="Include the 1e-4 z^2 latent prior term in the fit loss.")
    parser.add_argument("--mask", choices=["halves", "frustum"], default="halves",
                        help="Visible region: right half (default) or a central "
                             "pinhole frustum looking at the ERP centre.")
    parser.add_argument("--fov-h", type=float, default=90.0,
                        help="Frustum horizontal FoV in degrees (frustum mask only).")
    parser.add_argument("--fov-v", type=float, default=60.0,
                        help="Frustum vertical FoV in degrees (frustum mask only).")
    parser.add_argument("--latent-steps", type=int, default=None,
                        help="Unused placeholder for parity with sibling scripts; the "
                             "fit uses the fixed fig_outpainting recipe (600 steps).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of eval images (smoke test).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output stem for JSON/CSV; default derived from --prior.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    prior_weight = PRIOR_WEIGHT if args.prior == "on" else 0.0
    if args.output is not None:
        output_stem = args.output
    elif args.mask == "frustum":
        output_stem = DEFAULT_OUTPUT_DIR / (
            f"outpaint_frustum{int(args.fov_h)}x{int(args.fov_v)}_prior{args.prior}")
    else:
        output_stem = DEFAULT_OUTPUT_DIR / f"outpaint_compare_prior{args.prior}"

    if not args.run and not args.baseline:
        parser.error("provide at least one --run or --baseline")
    runs = parse_runs(args.run)
    models = {
        label: evaluate_run(label, path, args.data, args.device, args.latent_steps,
                            prior_weight, args.seed, args.max_images,
                            mask_mode=args.mask, fov_h=args.fov_h, fov_v=args.fov_v)
        for label, path in runs.items()
    }
    for value in args.baseline:
        family, _, key = value.partition("=")
        key = int(key) if key.isdigit() else key
        models[f"{family}"] = evaluate_baseline(
            family, key, args.device, args.seed, args.max_images,
            mask_mode=args.mask, fov_h=args.fov_h, fov_v=args.fov_v,
            ldr_exposure=args.ldr_exposure)

    if args.mask == "frustum":
        mask_desc = (f"central_frustum_{args.fov_h:g}x{args.fov_v:g}deg_visible"
                     "_rest_hidden")
    else:
        mask_desc = "vertical_split_right_visible_left_hidden"
    results = {
        "runs": {label: str(path) for label, path in runs.items()},
        "data": str(args.data),
        "device": args.device,
        "seed": args.seed,
        "prior": args.prior,
        "prior_weight": prior_weight,
        "fit_steps": FIT_STEPS,
        "fit_lr": FIT_LR,
        "mask": mask_desc,
        "regions": list(REGIONS),
        "models": models,
    }

    _write_outputs(output_stem, results)
    _print_summary(results)


if __name__ == "__main__":
    main()
