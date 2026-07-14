#!/usr/bin/env python3
"""Train the two missing RENI++ architecture-ablation size ladders."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RECIPES = {
    "reni_retrained": "reni_ablation_retrained",
    "transformer_decoder": "reni_ablation_transformer",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[9, 36, 49, 100])
    parser.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    parser.add_argument("--timestamp", default="2026-07-14_50k")
    parser.add_argument("--vis", default="wandb")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for recipe in args.recipes:
        stem = RECIPES[recipe]
        for dimension in args.dimensions:
            experiment = f"{stem}_d{dimension}"
            run_dir = args.output_dir / experiment / "reni" / args.timestamp
            checkpoints = sorted((run_dir / "nerfstudio_models").glob("step-*.ckpt"))
            if checkpoints and not args.force:
                print(f"[skip] {experiment}: {checkpoints[-1]}", flush=True)
                continue

            command = [
                sys.executable,
                "scripts/train_reni.py",
                "--data",
                str(args.data),
                "--latent-dim",
                str(dimension),
                "--ablation-recipe",
                recipe,
                "--max-num-iterations",
                "50001",
                "--experiment-name",
                experiment,
                "--timestamp",
                args.timestamp,
                "--output-dir",
                str(args.output_dir),
                "--vis",
                args.vis,
            ]
            print(f"[train] {experiment}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
