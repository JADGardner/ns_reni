"""Common validation-only refit for the isolated log-loss correction pilot.

All decoders use the same corrected fitting loss and seed. Scores remove a
single scalar log-exposure offset per image, never multiply log radiances.
These diagnostics use an explicit new protocol, not the historical table's
normalisation; their absolute values must not be pasted into that table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _common import seed_all
from eval_latent_reset_compare import _build_test_config, _latest_checkpoint, _load_decoder_state
from reni.utils.colourspace import linear_to_sRGB


def aligned_scores(pred_log, target_log):
    """ERP solid-angle weighted metrics and a shared-exposure display pair."""
    height, width, _ = target_log.shape
    weights = torch.sin((torch.arange(height, device=target_log.device, dtype=target_log.dtype) + .5) * torch.pi / height)
    weights = weights[:, None, None].expand(height, width, 3)
    weights = weights / weights.sum()
    offset = ((target_log - pred_log) * weights).sum()
    pred_log = pred_log + offset
    log_rmse = (((pred_log - target_log).square() * weights).sum()).sqrt()
    pred, target = pred_log.exp(), target_log.exp()
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite linear HDR reconstruction")
    white = torch.quantile(target.flatten(), .98).clamp_min(1e-8)
    relative_rmse = ((weights * (pred - target).square()).sum() / (weights * target.square()).sum()).sqrt()
    pred_ldr = linear_to_sRGB(pred, q=white)
    target_ldr = linear_to_sRGB(target, q=white)
    mse = (weights * (pred_ldr - target_ldr).square()).sum()
    colour_error = 1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    score = {
        "log_rmse": float(log_rmse),
        "relative_hdr_rmse": float(relative_rmse),
        "srgb_psnr_shared_gt_q98": float(-10 * torch.log10(mse.clamp_min(1e-15))),
        "linear_rgb_cosine_error": float((colour_error * weights.sum(-1)).sum()),
        "log_exposure_offset": float(offset),
    }
    return score, torch.cat([target_ldr, pred_ldr], dim=1), pred_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Exact directory containing config.yml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    config = _build_test_config(args.run, args.data, args.latent_steps)
    config.pipeline.test_mode = "val"
    dp = config.pipeline.datamanager.dataparser
    dp.train_subset_size = 1  # decoder loading discards train latent banks
    dp.val_subset_size = None
    dp.custom_val_folder = None
    model_config = config.pipeline.model
    model_config.log_loss_variant = "both"
    model_config.loss_inclusions = dict(model_config.loss_inclusions)
    for key in model_config.loss_inclusions:
        model_config.loss_inclusions[key] = key in ("scale_inv_loss", "cosine_similarity_loss")
    model_config.loss_coefficients = {"scale_inv_loss": 1., "cosine_similarity_loss": 1.}
    model_config.luminance_weighted_loss = False
    if not dp.convert_to_log_domain or dp.min_max_normalize is not None or dp.tonemap_targets:
        raise ValueError("Pilot evaluation accepts raw-log models only")
    pipeline = config.pipeline.setup(device="cuda:0", test_mode="val", world_size=1, local_rank=0, grad_scaler=None)
    checkpoint = _latest_checkpoint(args.run)
    load_stats = _load_decoder_state(pipeline, checkpoint, "cuda:0")
    pipeline.eval()
    seed_all(args.seed)
    pipeline.model.fit_eval_latents(pipeline.datamanager)
    rows = []
    with torch.no_grad():
        for sample in pipeline.datamanager.fixed_indices_eval_dataloader:
            image_idx, rays, batch = pipeline._eval_image_to_ray_bundle(sample)
            rays = pipeline._flatten_eval_ray_bundle(rays)
            rays.camera_indices.fill_(image_idx)
            target = batch["image"].to("cuda:0")
            prediction = pipeline.model(rays)["rgb"].reshape_as(target)
            if target.ndim != 3:
                raise ValueError("Expected an ERP image with shape [H, W, 3]")
            score, pair, aligned_log = aligned_scores(prediction, target)
            score["image_idx"] = image_idx
            score["filename"] = str(pipeline.datamanager.eval_dataset.image_filenames[image_idx])
            rows.append(score)
            Image.fromarray((pair.cpu().numpy() * 255).round().astype(np.uint8)).save(args.output / f"{image_idx:03d}_gt_prediction.png")
            np.savez_compressed(args.output / f"{image_idx:03d}.npz", target_log=target.cpu().numpy(), prediction_log=prediction.cpu().numpy(), aligned_prediction_log=aligned_log.cpu().numpy())
    report = {
        "run": str(args.run), "checkpoint": str(checkpoint), "split": "val",
        "seed": args.seed, "latent_steps": args.latent_steps, "load_stats": load_stats,
        "fitting_loss": "per-image scalar-centred log MSE + linear-RGB cosine; coefficients 1",
        "scoring": "ERP solid-angle weights; per-image scalar additive log alignment; shared GT q98 for sRGB",
        "per_image": rows,
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in ("filename", "image_idx", "log_exposure_offset")},
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["mean"], indent=2))


if __name__ == "__main__":
    main()
