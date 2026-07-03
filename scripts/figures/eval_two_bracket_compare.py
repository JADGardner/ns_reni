"""Compare G7h runs (baseline / two-bracket / luminance control) with refit test latents.

Extends the eval_latent_reset_compare refit protocol with peak-aware HDR
metrics (log-HDR RMSE, luminance-weighted HDR RMSE, top-0.1% peak intensity
relative error, peak angular errors). Each run is rebuilt in test mode from its
saved config, checkpoint decoder weights are loaded (latent banks dropped), and
test latents are refit from zero before measuring; peak metrics are computed in
a shared fixed gauge (GT p99 luminance = 1) with median-ratio exposure
alignment for scale-invariant models, so runs trained in different target
spaces are comparable.

Run from the ns_reni repo root:

    PYTHONPATH=.:scripts/figures python scripts/figures/eval_two_bracket_compare.py \
        --run baseline=outputs/reni_latent_reset_4_rerun \
        --run two_bracket=outputs/reni_latent_reset_d100_two_bracket \
        --run lum_control=outputs/reni_latent_reset_d100_luminance_control \
        --data /path/to/RENI_HDR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import torch

from _common import REPO_ROOT, seed_all
from eval_latent_reset_compare import (
    _build_test_config,
    _eval_split_name,
    _latest_checkpoint,
    _load_decoder_state,
    resolve_run_dir,
)

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "evaluations" / "two_bracket_compare"

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


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    cleaned = {}
    for key in METRICS:
        value = metrics.get(key, float("nan"))
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        cleaned[key] = float(value)
    return cleaned


def evaluate_run(
    label: str,
    run_path: Path,
    data_root: Path,
    device: str,
    latent_steps: int | None,
    seed: int,
    force_align: bool = False,
) -> Dict[str, Any]:
    seed_all(seed)
    run_dir = resolve_run_dir(run_path)
    checkpoint = _latest_checkpoint(run_dir)
    print(f"[load] {label}: {checkpoint}")

    model_config = _build_test_config(run_dir, data_root, latent_steps)
    model_config.pipeline.model.compute_hdr_peak_metrics = True
    if force_align:
        model_config.pipeline.model.force_eval_exposure_alignment = True
    pipeline = model_config.pipeline.setup(
        device=device,
        test_mode="test",
        world_size=1,
        local_rank=0,
        grad_scaler=None,
    )
    load_stats = _load_decoder_state(pipeline, checkpoint, device)
    pipeline.model.to(device)
    pipeline.model.eval()

    split = _eval_split_name(pipeline.datamanager)
    num_eval = len(pipeline.datamanager.eval_dataset)
    print(f"[eval] {label}: split={split}, images={num_eval}, load={load_stats}")
    metrics = pipeline.get_average_eval_image_metrics(step=1, optimise_latents=True)

    result = {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "split": split,
        "num_eval_images": num_eval,
        "latent_refit": True,
        "force_eval_exposure_alignment": force_align,
        "two_bracket": bool(getattr(pipeline.model, "two_bracket", False)),
        "luminance_weighted_loss": bool(model_config.pipeline.model.luminance_weighted_loss),
        "load_stats": load_stats,
        "metrics": _clean_metrics(metrics),
    }
    del pipeline
    torch.cuda.empty_cache()
    return result


def _write_outputs(output_stem: Path, results: Dict[str, Any]) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    rows = [("model", "split", "num_eval_images", "latent_refit", *METRICS)]
    for key, result in results["models"].items():
        metrics = result["metrics"]
        rows.append(
            (
                key,
                result["split"],
                result["num_eval_images"],
                result["latent_refit"],
                *(f"{metrics[m]:.6f}" for m in METRICS),
            )
        )
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")


def _print_summary(results: Dict[str, Any]) -> None:
    print("\nSummary (refit test latents)")
    name_width = max(12, *(len(k) for k in results["models"]))
    header = f"{'model':{name_width}s} " + " ".join(f"{m:>14s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for key, result in results["models"].items():
        metrics = result["metrics"]
        cells = " ".join(
            f"{metrics[m]:14.4f}" if not math.isnan(metrics[m]) else f"{'nan':>14s}" for m in METRICS
        )
        print(f"{key:{name_width}s} {cells}")
    print("\n(arrows) " + " ".join(f"{m}:{'^' if HIGHER_BETTER[m] else 'v'}" for m in METRICS))


def parse_runs(values) -> Dict[str, Path]:
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run expects LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        runs[label] = Path(path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                        help="Run to evaluate; repeatable.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "RENI_HDR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-steps", type=int, default=None,
                        help="Override eval latent optimisation steps; default uses the model config.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output stem for JSON/CSV files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-align", nargs="*", default=[],
                        help="Run labels that get a per-image median-ratio exposure "
                             "alignment before metrics (parity diagnostic for "
                             "fixed-gauge models vs the scale-invariant free scale).")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    runs = parse_runs(args.run)
    models = {
        label: evaluate_run(label, path, args.data, args.device, args.latent_steps, args.seed,
                            force_align=label in args.force_align)
        for label, path in runs.items()
    }

    results = {
        "runs": {label: str(path) for label, path in runs.items()},
        "data": str(args.data),
        "device": args.device,
        "latent_steps_override": args.latent_steps,
        "seed": args.seed,
        "models": models,
    }

    _write_outputs(args.output, results)
    _print_summary(results)


if __name__ == "__main__":
    main()
