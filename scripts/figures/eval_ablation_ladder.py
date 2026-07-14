#!/usr/bin/env python3
"""Evaluate RENI architecture-ablation ladders in one radiometric gauge.

Each decoder is loaded without its saved latent banks, test latents are refit
from zero, and predictions are decoded to linear HDR. Scale-invariant models
receive one robust per-image exposure alignment. The prediction and target are
then placed in a shared gauge where the target's 99th-percentile luminance is
one. HDR PSNR is measured directly in that linear space. LDR PSNR uses one
target-derived 98th-percentile exposure for both images before sRGB conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

from _common import REPO_ROOT, seed_all
from eval_latent_reset_compare import (
    _build_test_config,
    _eval_split_name,
    _latest_checkpoint,
    _load_decoder_state,
    resolve_run_dir,
)
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import luminance


DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "evaluations" / "ablation_fixed_gauge"


def _parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run expects LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        runs[label] = Path(path)
    return runs


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _common_gauge(
    pred_linear: torch.Tensor,
    gt_linear: torch.Tensor,
    align_exposure: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    pred = pred_linear.clamp_min(0.0)
    gt = gt_linear.clamp_min(0.0)

    exposure = torch.ones((), device=pred.device, dtype=pred.dtype)
    if align_exposure:
        pred_lum = luminance(pred).flatten()
        gt_lum = luminance(gt).flatten()
        valid = (pred_lum > 1e-8) & (gt_lum > 1e-8)
        if valid.any():
            exposure = torch.median(gt_lum[valid] / pred_lum[valid])
            pred = pred * exposure

    gt_q99 = torch.quantile(luminance(gt).flatten(), 0.99).clamp_min(1e-8)
    gauge = gt_q99.reciprocal()
    return pred * gauge, gt * gauge, float(exposure.detach().cpu())


@torch.no_grad()
def _measure_images(pipeline: Any, max_images: int | None) -> dict[str, Any]:
    model = pipeline.model
    height = model.metadata["image_height"]
    width = model.metadata["image_width"]
    scale_inv = model.config.loss_inclusions.get("scale_inv_loss") in [True, "eval", "both"]

    hdr_values: list[float] = []
    ldr_values: list[float] = []
    exposures: list[float] = []

    for idx, eval_image_output in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        if max_images is not None and idx >= max_images:
            break
        image_idx, ray_bundle, batch = pipeline._eval_image_to_ray_bundle(eval_image_output)
        ray_bundle = pipeline._flatten_eval_ray_bundle(ray_bundle)
        ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * image_idx

        outputs = model(ray_bundle)
        pred_native = outputs["rgb"].reshape(height, width, -1)
        gt_native = batch["image"].to(pred_native.device).reshape(height, width, -1)
        pred_linear = model._to_linear_hdr(pred_native)
        gt_linear = model._to_linear_hdr(gt_native)
        pred, gt, exposure = _common_gauge(pred_linear, gt_linear, scale_inv)

        q98 = torch.quantile(luminance(gt).flatten(), 0.98).clamp_min(1e-8)
        pred_ldr = linear_to_sRGB(pred, q=q98)
        gt_ldr = linear_to_sRGB(gt, q=q98)

        hdr_values.append(float(_psnr(pred, gt).cpu()))
        ldr_values.append(float(_psnr(pred_ldr, gt_ldr).cpu()))
        exposures.append(exposure)

    if not hdr_values:
        raise RuntimeError("No evaluation images were measured.")
    return {
        "num_eval_images": len(hdr_values),
        "psnr_ldr": sum(ldr_values) / len(ldr_values),
        "psnr_hdr_fixed_gauge": sum(hdr_values) / len(hdr_values),
        "exposure_alignment": "median_ratio" if scale_inv else "none",
        "median_exposure_scale": float(torch.tensor(exposures).median()),
    }


def _evaluate_run(
    label: str,
    path: Path,
    data: Path,
    device: str,
    latent_steps: int | None,
    max_images: int | None,
    checkpoint_step: int | None,
    seed: int,
    preserve_eval_latents: bool,
) -> dict[str, Any]:
    seed_all(seed)
    run_dir = resolve_run_dir(path)
    if checkpoint_step is None:
        checkpoint = _latest_checkpoint(run_dir)
    else:
        checkpoint = run_dir / "nerfstudio_models" / f"step-{checkpoint_step:09d}.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Requested step {checkpoint_step} for {label}, but {checkpoint} does not exist."
            )
    print(f"[load] {label}: {checkpoint}")

    config = _build_test_config(run_dir, data, latent_steps)
    pipeline = config.pipeline.setup(
        device=device,
        test_mode="test",
        world_size=1,
        local_rank=0,
        grad_scaler=None,
    )
    load_stats = _load_decoder_state(
        pipeline,
        checkpoint,
        device,
        preserve_eval_latents=preserve_eval_latents,
    )
    pipeline.model.to(device)
    pipeline.model.eval()
    if not preserve_eval_latents:
        pipeline._optimise_evaluation_latents(step=1)
    metrics = _measure_images(pipeline, max_images)

    result = {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "split": _eval_split_name(pipeline.datamanager),
        "latent_refit": not preserve_eval_latents,
        "load_stats": load_stats,
        "metrics": metrics,
    }
    del pipeline
    torch.cuda.empty_cache()
    return result


def _write_outputs(output: Path, results: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")

    rows = [[
        "model",
        "split",
        "num_eval_images",
        "latent_refit",
        "exposure_alignment",
        "psnr_ldr",
        "psnr_hdr_fixed_gauge",
    ]]
    for label, result in results["models"].items():
        metrics = result["metrics"]
        rows.append([
            label,
            result["split"],
            metrics["num_eval_images"],
            result["latent_refit"],
            metrics["exposure_alignment"],
            f"{metrics['psnr_ldr']:.6f}",
            f"{metrics['psnr_hdr_fixed_gauge']:.6f}",
        ])
    with output.with_suffix(".csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "RENI_HDR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-steps", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Evaluate this exact saved training step for every run instead of each latest checkpoint.",
    )
    parser.add_argument(
        "--preserve-eval-latents",
        action="append",
        default=[],
        metavar="LABEL",
        help="Use this run's saved test latent bank instead of refitting it from zero.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    seed_all(args.seed)
    runs = _parse_runs(args.run)
    unknown_preserved = set(args.preserve_eval_latents) - set(runs)
    if unknown_preserved:
        raise ValueError(f"Unknown --preserve-eval-latents labels: {sorted(unknown_preserved)}")
    models = {
        label: _evaluate_run(
            label,
            path,
            args.data,
            args.device,
            args.latent_steps,
            args.max_images,
            args.checkpoint_step,
            args.seed,
            label in args.preserve_eval_latents,
        )
        for label, path in runs.items()
    }
    results = {
        "protocol": {
            "test_latents": (
                "refit from zero except labels explicitly using archived fitted eval latents"
            ),
            "archived_eval_latent_labels": list(args.preserve_eval_latents),
            "hdr_domain": "decoded linear radiance",
            "gauge": "per-image GT 99th-percentile luminance = 1",
            "scale_invariant_alignment": "median luminance ratio",
            "ldr_exposure": "shared per-image GT 98th-percentile luminance",
            "aggregation": "mean per-image PSNR",
        },
        "runs": {label: str(path) for label, path in runs.items()},
        "data": str(args.data),
        "device": args.device,
        "latent_steps_override": args.latent_steps,
        "checkpoint_step": args.checkpoint_step,
        "models": models,
    }
    _write_outputs(args.output, results)

    print("\nmodel                              LDR PSNR  HDR PSNR")
    for label, result in models.items():
        metrics = result["metrics"]
        print(f"{label:34s} {metrics['psnr_ldr']:8.2f}  {metrics['psnr_hdr_fixed_gauge']:8.2f}")


if __name__ == "__main__":
    main()
