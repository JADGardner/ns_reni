"""Evaluate the latent-reset checkpoint against the RENI++ paper checkpoint.

The latent-reset run saved validation eval latents during training, so this
script rebuilds a test-mode pipeline, drops checkpoint eval latents, and refits
test latents from zero before measuring. It also reports the RENI++ D=100
checkpoint eval bank as a diagnostic row; the refit rows are the fair comparison.

Run from the ns_reni repo root:

    PYTHONPATH=. python scripts/figures/eval_latent_reset_compare.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from _common import MODEL_DIRS, RENIField, REPO_ROOT, _clean_yaml, load_model, resolve_run_dir, seed_all


DEFAULT_CANDIDATE = (
    REPO_ROOT
    / "outputs"
    / "reni_latent_reset_4_rerun"
    / "reni"
    / "2026-07-01_4cycles_rerun"
)
DEFAULT_BASELINE = MODEL_DIRS["reni_pp"][100]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "evaluations" / "latent_reset_test_compare"
PAPER_TABLE_CACHE = REPO_ROOT / "publication" / "tables" / "comparison.metrics.json"

METRICS = (
    "psnr_hdr",
    "ssim_hdr",
    "lpips_hdr",
    "psnr_ldr",
    "ssim_ldr",
    "lpips_ldr",
)
HIGHER_BETTER = {
    "psnr_hdr": True,
    "ssim_hdr": True,
    "lpips_hdr": False,
    "psnr_ldr": True,
    "ssim_ldr": True,
    "lpips_ldr": False,
}
LATENT_BANK_KEYS = {
    "field.train_mu",
    "field.train_logvar",
    "field.eval_mu",
    "field.eval_logvar",
    "field.train_scale",
    "field.eval_scale",
}


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "nerfstudio_models"
    checkpoints = sorted(
        ckpt_dir.glob("step-*.ckpt"),
        key=lambda p: int(re.search(r"step-(\d+)\.ckpt", p.name).group(1)),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return checkpoints[-1]


def _path_from_yaml(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, list):
        return Path(*[str(part) for part in value])
    return Path(value)


def _copy_dataparser_settings(dataparser_config: Any, saved_dp: Dict[str, Any], data_root: Path) -> None:
    dataparser_config.data = data_root
    for key in (
        "resize_image_width",
        "convert_to_ldr",
        "convert_to_log_domain",
        "min_max_normalize",
        "augment_with_mirror",
        "augment_with_rotation",
        "apply_eval_rotation",
        "train_subset_size",
        "val_subset_size",
        "custom_val_folder",
        "fixed_gauge_normalisation",
        "fixed_gauge_percentile",
        "fixed_gauge_target",
        "tonemap_targets",
        "tonemap_m_ldr",
        "tonemap_m_log",
    ):
        if key in saved_dp and hasattr(dataparser_config, key):
            value = saved_dp[key]
            # Removing PyYAML's python-specific tuple tag makes archived
            # fixed log ranges load as lists. RENIDataset distinguishes a
            # fixed (min, max) tuple from its string-computed modes.
            if key == "min_max_normalize" and isinstance(value, list):
                value = tuple(value)
            setattr(dataparser_config, key, value)

    if "fit_val_in_ldr" in saved_dp and hasattr(dataparser_config, "val_in_ldr"):
        dataparser_config.val_in_ldr = saved_dp["fit_val_in_ldr"]
    if "val_in_ldr" in saved_dp and hasattr(dataparser_config, "val_in_ldr"):
        dataparser_config.val_in_ldr = saved_dp["val_in_ldr"]

    for key in ("train_mask_path", "eval_mask_path"):
        if key in saved_dp and hasattr(dataparser_config, key):
            setattr(dataparser_config, key, _path_from_yaml(saved_dp[key]))

    if hasattr(dataparser_config, "use_validation_as_train"):
        dataparser_config.use_validation_as_train = False
    if hasattr(dataparser_config, "use_test_as_train"):
        dataparser_config.use_test_as_train = False


def _copy_field_settings(field_config: Any, saved_field: Dict[str, Any]) -> None:
    for key in (
        "fixed_decoder",
        "conditioning",
        "invariant_function",
        "equivariance",
        "axis_of_invariance",
        "positional_encoding",
        "encoded_input",
        "latent_dim",
        "hidden_features",
        "hidden_layers",
        "mapping_layers",
        "mapping_features",
        "num_attention_heads",
        "num_attention_layers",
        "output_activation",
        "out_features",
        "last_layer_linear",
        "trainable_scale",
        "old_implementation",
        "view_train_latents",
    ):
        if key in saved_field and hasattr(field_config, key):
            setattr(field_config, key, saved_field[key])


def _copy_loss_settings(model_config: Any, saved_model: Dict[str, Any]) -> None:
    if "loss_inclusions" in saved_model:
        current = dict(model_config.loss_inclusions)
        for key, value in saved_model["loss_inclusions"].items():
            if key in current:
                current[key] = value
        model_config.loss_inclusions = current

    if "loss_coefficients" in saved_model and hasattr(model_config, "loss_coefficients"):
        current = dict(model_config.loss_coefficients)
        current.update(saved_model["loss_coefficients"])
        model_config.loss_coefficients = current

    for key in (
        "luminance_weighted_loss",
        "luminance_weight_power",
        "luminance_weight_cap",
        "blended_recon_domain",
    ):
        if key in saved_model and hasattr(model_config, key):
            setattr(model_config, key, saved_model[key])


def _set_eval_latent_steps(model_config: Any, latent_steps: Optional[int]) -> None:
    if latent_steps is None:
        return
    scheduler = model_config.eval_latent_optimizer["eval_latents"]["scheduler"]
    if not hasattr(scheduler, "max_steps"):
        raise TypeError(f"Unsupported eval latent scheduler config: {type(scheduler)}")
    scheduler.max_steps = latent_steps


def _build_test_config(run_dir: Path, data_root: Path, latent_steps: Optional[int]) -> Any:
    config = _clean_yaml((run_dir / "config.yml").read_text())
    model_config = copy.deepcopy(RENIField.config)
    model_config.pipeline.test_mode = "test"

    datamanager = model_config.pipeline.datamanager
    if hasattr(datamanager, "data"):
        datamanager.data = data_root
    saved_dp = config["pipeline"]["datamanager"]["dataparser"]
    _copy_dataparser_settings(datamanager.dataparser, saved_dp, data_root)

    saved_model = config["pipeline"]["model"]
    _copy_field_settings(model_config.pipeline.model.field, saved_model["field"])
    _copy_loss_settings(model_config.pipeline.model, saved_model)
    _set_eval_latent_steps(model_config.pipeline.model, latent_steps)
    return model_config


def _model_state_from_checkpoint(checkpoint: Path, device: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = {
        key[len("_model.") :]: value
        for key, value in ckpt["pipeline"].items()
        if key.startswith("_model.")
    }
    if model_state and all(key.startswith("module.") for key in model_state):
        model_state = {key[len("module.") :]: value for key, value in model_state.items()}
    return model_state


def _load_decoder_state(
    pipeline: Any,
    checkpoint: Path,
    device: str,
    preserve_eval_latents: bool = False,
) -> Dict[str, int]:
    checkpoint_state = _model_state_from_checkpoint(checkpoint, device)
    target_state = pipeline.model.state_dict()
    filtered = {}
    skipped = {"latent_bank": 0, "missing_key": 0, "shape": 0}

    for key, value in checkpoint_state.items():
        keep_eval_latent = preserve_eval_latents and key in {
            "field.eval_mu",
            "field.eval_logvar",
            "field.eval_scale",
        }
        if key in LATENT_BANK_KEYS and not keep_eval_latent:
            skipped["latent_bank"] += 1
            continue
        if key not in target_state:
            skipped["missing_key"] += 1
            continue
        if target_state[key].shape != value.shape:
            skipped["shape"] += 1
            continue
        filtered[key] = value

    missing, unexpected = pipeline.model.load_state_dict(filtered, strict=False)
    real_missing = [
        key
        for key in missing
        if key not in LATENT_BANK_KEYS
        and "lpips" not in key
        and "device_indicator" not in key
    ]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint}: "
            f"missing={real_missing}, unexpected={list(unexpected)}"
        )

    skipped["loaded"] = len(filtered)
    return skipped


def _clean_metrics(metrics: Dict[str, Any], keys: Iterable[str] = METRICS) -> Dict[str, float]:
    cleaned = {}
    for key in keys:
        value = metrics[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        cleaned[key] = float(value)
    return cleaned


def _eval_split_name(datamanager: Any) -> str:
    filenames = getattr(datamanager.eval_dataset._dataparser_outputs, "image_filenames", [])
    split_names = {Path(filename).parent.name for filename in filenames}
    if len(split_names) == 1:
        return next(iter(split_names))
    return getattr(datamanager.eval_dataset, "split", "unknown")


def evaluate_refit(
    label: str,
    run_path: Path,
    data_root: Path,
    device: str,
    latent_steps: Optional[int],
) -> Dict[str, Any]:
    run_dir = resolve_run_dir(run_path)
    checkpoint = _latest_checkpoint(run_dir)
    print(f"[load-refit] {label}: {checkpoint}")

    model_config = _build_test_config(run_dir, data_root, latent_steps)
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
    print(f"[eval-refit] {label}: split={split}, images={num_eval}, load={load_stats}")
    metrics = pipeline.get_average_eval_image_metrics(step=1, optimise_latents=True)
    result = {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "split": split,
        "num_eval_images": num_eval,
        "latent_refit": True,
        "load_stats": load_stats,
        "metrics": _clean_metrics(metrics),
    }
    del pipeline
    torch.cuda.empty_cache()
    return result


def evaluate_paper_stored(label: str, run_path: Path, device: str) -> Dict[str, Any]:
    print(f"[load-stored] {label}: {run_path}")
    pipeline, datamanager, _ = load_model(run_path, device=device)
    split = _eval_split_name(datamanager)
    num_eval = len(datamanager.eval_dataset)
    print(f"[eval-stored] {label}: split={split}, images={num_eval}")
    metrics = pipeline.get_average_eval_image_metrics(optimise_latents=False)
    result = {
        "label": label,
        "run_dir": str(resolve_run_dir(run_path)),
        "checkpoint": str(_latest_checkpoint(resolve_run_dir(run_path))),
        "split": split,
        "num_eval_images": num_eval,
        "latent_refit": False,
        "metrics": _clean_metrics(metrics),
    }
    del pipeline, datamanager
    torch.cuda.empty_cache()
    return result


def _deltas(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    return {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in METRICS
        if key in candidate_metrics and key in baseline_metrics
    }


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
    print("\nSummary")
    header = f"{'model':32s} {'PSNR_LDR':>9s} {'SSIM_LDR':>9s} {'LPIPS_LDR':>10s}"
    print(header)
    print("-" * len(header))
    for key, result in results["models"].items():
        metrics = result["metrics"]
        print(
            f"{key:32s} "
            f"{metrics['psnr_ldr']:9.3f} "
            f"{metrics['ssim_ldr']:9.4f} "
            f"{metrics['lpips_ldr']:10.4f}"
        )

    candidate = results["models"]["latent_reset_refit"]
    for baseline_key, delta in results["deltas_vs_latent_reset"].items():
        print(f"\nDelta latent_reset_refit - {baseline_key}")
        for metric, value in delta.items():
            direction = "higher is better" if HIGHER_BETTER[metric] else "lower is better"
            print(f"  {metric}: {value:+.6f} ({direction})")
    del candidate

    paper_cache = results.get("paper_table_cache_ldr")
    if paper_cache is not None:
        metrics = paper_cache["metrics"]
        candidate_metrics = results["models"]["latent_reset_refit"]["metrics"]
        print("\nCached paper-table RENI++ D=100 LDR reference")
        print(
            f"  PSNR/SSIM/LPIPS: "
            f"{metrics['psnr_ldr']:.3f} / {metrics['ssim_ldr']:.4f} / {metrics['lpips_ldr']:.4f}"
        )
        print(
            "  Delta latent_reset_refit - cached table: "
            f"psnr_ldr={candidate_metrics['psnr_ldr'] - metrics['psnr_ldr']:+.6f}, "
            f"ssim_ldr={candidate_metrics['ssim_ldr'] - metrics['ssim_ldr']:+.6f}, "
            f"lpips_ldr={candidate_metrics['lpips_ldr'] - metrics['lpips_ldr']:+.6f}"
        )


def _load_paper_table_cache() -> Optional[Dict[str, Any]]:
    if not PAPER_TABLE_CACHE.exists():
        return None
    metrics = json.loads(PAPER_TABLE_CACHE.read_text()).get("reni_pp/100")
    if metrics is None:
        return None
    return {
        "source": str(PAPER_TABLE_CACHE),
        "key": "reni_pp/100",
        "metrics": {key: float(metrics[key]) for key in ("psnr_ldr", "ssim_ldr", "lpips_ldr")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data", type=Path, default=Path("/home/james/data/RENI_HDR"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-steps", type=int, default=None,
                        help="Override eval latent optimisation steps; default uses the model config.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output stem for JSON/CSV files.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    seed_all(args.seed)

    baseline_stored = evaluate_paper_stored(
        "reni_pp_d100_checkpoint_eval_bank",
        args.baseline,
        args.device,
    )
    baseline_refit = evaluate_refit(
        "reni_pp_d100_refit_test_latents",
        args.baseline,
        args.data,
        args.device,
        args.latent_steps,
    )
    candidate_refit = evaluate_refit(
        "latent_reset_refit_test_latents",
        args.candidate,
        args.data,
        args.device,
        args.latent_steps,
    )

    models = {
        "reni_pp_checkpoint_bank": baseline_stored,
        "reni_pp_refit": baseline_refit,
        "latent_reset_refit": candidate_refit,
    }
    results = {
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "data": str(args.data),
        "device": args.device,
        "latent_steps_override": args.latent_steps,
        "models": models,
        "paper_table_cache_ldr": _load_paper_table_cache(),
        "deltas_vs_latent_reset": {
            key: _deltas(candidate_refit, value)
            for key, value in models.items()
            if key != "latent_reset_refit"
        },
    }

    _write_outputs(args.output, results)
    _print_summary(results)


if __name__ == "__main__":
    main()
