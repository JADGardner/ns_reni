#!/usr/bin/env python3
"""Evaluate the thesis architecture and two-bracket equivariance matrices."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPO_ROOT.parents[1]
LATENT_DIMS = (9, 36, 49, 100)
ARCHITECTURE_TIMESTAMP = "2026-07-14_50k_spherical"
EQUIVARIANCE_TIMESTAMP = "2026-07-14_two_bracket_2cycles"


def _architecture_runs(model_archive: Path, outputs_root: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for dimension in LATENT_DIMS:
        runs[f"reni_d{dimension}"] = model_archive / "old_reni_models" / f"latent_dim_{dimension}"
        runs[f"retrained_d{dimension}"] = (
            outputs_root
            / f"reni_ablation_retrained_d{dimension}"
            / "reni"
            / ARCHITECTURE_TIMESTAMP
        )
        runs[f"transformer_d{dimension}"] = (
            outputs_root
            / f"reni_ablation_transformer_d{dimension}"
            / "reni"
            / ARCHITECTURE_TIMESTAMP
        )
        runs[f"reni_pp_d{dimension}"] = (
            model_archive / "reni_plus_plus_models" / f"latent_dim_{dimension}"
        )
        if dimension == 100:
            runs[f"two_bracket_d{dimension}"] = (
                outputs_root
                / "reni_latent_reset_d100_two_bracket_ldrw3_2cyc"
                / "reni"
                / "2026-07-04_2cycles"
            )
        else:
            runs[f"two_bracket_d{dimension}"] = (
                outputs_root
                / f"reni_two_bracket_w3_2cyc_d{dimension}"
                / "reni"
                / "2026-07-04_2cycles"
            )
    return runs


def _equivariance_runs(outputs_root: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for dimension in LATENT_DIMS:
        for equivariance in ("none", "so3"):
            runs[f"{equivariance}_d{dimension}"] = (
                outputs_root
                / f"reni_equivariance_{equivariance}_two_bracket_2cyc_d{dimension}"
                / "reni"
                / EQUIVARIANCE_TIMESTAMP
            )
        if dimension == 100:
            runs[f"so2_d{dimension}"] = (
                outputs_root
                / "reni_latent_reset_d100_two_bracket_ldrw3_2cyc"
                / "reni"
                / "2026-07-04_2cycles"
            )
        else:
            runs[f"so2_d{dimension}"] = (
                outputs_root
                / f"reni_two_bracket_w3_2cyc_d{dimension}"
                / "reni"
                / "2026-07-04_2cycles"
            )
    return runs


def _evaluate(
    runs: dict[str, Path],
    output: Path,
    data: Path,
    latent_steps: int,
    max_images: int | None,
    checkpoint_step: int | None = None,
    preserve_eval_latents: tuple[str, ...] = (),
) -> None:
    missing = [str(path) for path in runs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing run directories:\n" + "\n".join(missing))

    command = [
        sys.executable,
        "scripts/figures/eval_ablation_ladder.py",
        "--data",
        str(data),
        "--latent-steps",
        str(latent_steps),
        "--output",
        str(output),
    ]
    if max_images is not None:
        command.extend(("--max-images", str(max_images)))
    if checkpoint_step is not None:
        command.extend(("--checkpoint-step", str(checkpoint_step)))
    for label in preserve_eval_latents:
        command.extend(("--preserve-eval-latents", label))
    for label, path in runs.items():
        command.extend(("--run", f"{label}={path}"))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        choices=("architecture", "equivariance", "both"),
        default="both",
    )
    parser.add_argument(
        "--model-archive",
        type=Path,
        default=PHD_ROOT / "model-storage" / "reni_paper_models",
    )
    parser.add_argument("--outputs-root", type=Path, default=PHD_ROOT / "outputs" / "reni")
    parser.add_argument("--data", type=Path, default=PHD_ROOT / "data" / "RENI_HDR")
    parser.add_argument("--latent-steps", type=int, default=2500)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--equivariance-steps",
        type=int,
        nargs="+",
        choices=(50000, 100000, 100001),
        default=(100001,),
        help="Checkpoint endpoints to evaluate; the two-cycle trainer's final save is step 100001.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "evaluations",
    )
    args = parser.parse_args()

    if args.matrix in ("architecture", "both"):
        _evaluate(
            _architecture_runs(args.model_archive, args.outputs_root),
            args.output_dir / "ablation_architecture_fixed_gauge",
            args.data,
            args.latent_steps,
            args.max_images,
            preserve_eval_latents=tuple(f"reni_d{dimension}" for dimension in LATENT_DIMS),
        )
    if args.matrix in ("equivariance", "both"):
        equivariance_runs = _equivariance_runs(args.outputs_root)
        for checkpoint_step in args.equivariance_steps:
            cycle_label = f"{checkpoint_step // 1000}k"
            _evaluate(
                equivariance_runs,
                args.output_dir
                / f"ablation_equivariance_two_bracket_{cycle_label}_fixed_gauge",
                args.data,
                args.latent_steps,
                args.max_images,
                checkpoint_step=checkpoint_step,
            )


if __name__ == "__main__":
    main()
