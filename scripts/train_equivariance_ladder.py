#!/usr/bin/env python3
"""Train the missing two-bracket None/SO(3) equivariance ladders.

The existing two-bracket, two-cycle SO(2) checkpoints supply the middle
column. Every new run changes only the field equivariance class.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EQUIVARIANCE_CLASSES = ("None", "SO3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[9, 36, 49, 100])
    parser.add_argument(
        "--equivariance",
        nargs="+",
        choices=EQUIVARIANCE_CLASSES,
        default=list(EQUIVARIANCE_CLASSES),
    )
    parser.add_argument("--timestamp", default="2026-07-14_two_bracket_2cycles")
    parser.add_argument("--vis", default="wandb")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for equivariance in args.equivariance:
        equivariance_slug = equivariance.lower()
        for dimension in args.dimensions:
            experiment = f"reni_equivariance_{equivariance_slug}_two_bracket_2cyc_d{dimension}"
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
                "--training-paradigm",
                "latent_reset",
                "--latent-reset-cycles",
                "2",
                "--max-num-iterations",
                "50001",
                "--variant",
                "two_bracket",
                "--ldr-bracket-weight",
                "3",
                "--equivariance",
                equivariance,
                "--keep-checkpoints",
                "--quiet-local-writer",
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
