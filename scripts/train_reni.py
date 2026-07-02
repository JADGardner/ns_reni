#!/usr/bin/env python3
"""Launch a RENI training run without going through the ns-train CLI.

Mirrors neusky's train_nerfosr.py: imports the method config directly and
hands it to nerfstudio's train entry point. This sidesteps environments where
ns-train's CLI layer fails (e.g. the csgpu13 SIF's typeguard/tyro conflict)
and works with PYTHONPATH-only setups where entry points are not registered.

Usage (inside the research container, cwd = code/ns_reni):
    PYTHONPATH=. python scripts/train_reni.py \
        --data /workspace/data/RENI_HDR --latent-dim 49 \
        --training-paradigm latent_reset --latent-reset-cycles 4
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/RENI_HDR"))
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--training-paradigm", default="standard",
                        choices=["standard", "latent_reset"])
    parser.add_argument("--latent-reset-cycles", type=int, default=1)
    parser.add_argument("--max-num-iterations", type=int, default=50001,
                        help="Iterations per cycle under latent_reset.")
    parser.add_argument("--experiment-name", default=None,
                        help="Default: reni_<paradigm>_d<latent_dim>")
    parser.add_argument("--timestamp", default=None,
                        help="Override the run timestamp directory name.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--vis", default="wandb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from nerfstudio.scripts.train import main as ns_train_main
    from reni.configs.reni_config import RENIField

    config = copy.deepcopy(RENIField.config)
    config.data = args.data.expanduser()
    config.pipeline.datamanager.data = config.data
    config.pipeline.model.field.latent_dim = args.latent_dim
    config.training_paradigm = args.training_paradigm
    config.latent_reset_cycles = args.latent_reset_cycles
    config.max_num_iterations = args.max_num_iterations
    config.experiment_name = args.experiment_name or (
        f"reni_{args.training_paradigm}_d{args.latent_dim}"
    )
    if args.timestamp is not None:
        config.timestamp = args.timestamp
    config.output_dir = args.output_dir
    config.vis = args.vis

    ns_train_main(config)


if __name__ == "__main__":
    main()
