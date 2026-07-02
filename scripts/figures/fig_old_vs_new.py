"""RENI vs RENI++ comparison (paper Fig: old_vs_new).

Col 1: GT. Cols 2-4: original RENI at D=9/49/100. Cols 5-7: RENI++ at the
same latent dimensions.

    PYTHONPATH=. python scripts/figures/fig_old_vs_new.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _common import (MODEL_DIRS, add_common_args, add_figure_label,
                     axes_span_center_x, axis_center, collect_model_outputs,
                     resolve_run_dir, save_figure, seed_all)

DIMS = (9, 49, 100)


def _add_old_vs_new_labels(fig, axes, fontsize):
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    label_y = axes[0, 0].get_position().y1 + 0.052
    sublabel_y = axes[0, 0].get_position().y1 + 0.026

    add_figure_label(fig, axis_center(axes[0, 0])[0], label_y, "Ground",
                     fontsize)
    add_figure_label(fig, axis_center(axes[0, 0])[0], sublabel_y, "Truth",
                     fontsize)
    add_figure_label(fig, axes_span_center_x(*axes[0, 1:4]), label_y, "RENI",
                     fontsize)
    add_figure_label(fig, axes_span_center_x(*axes[0, 4:7]), label_y, "RENI++",
                     fontsize)
    for col, label in zip(range(1, 7), ("27", "147", "300", "27", "147", "300")):
        add_figure_label(fig, axis_center(axes[0, col])[0], sublabel_y, label,
                         fontsize)


def _available(path) -> bool:
    try:
        resolve_run_dir(path)
        return True
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "old_vs_new")
    parser.add_argument("--image_indices", type=int, nargs="+",
                        default=[6, 7, 8, 9, 10])
    parser.add_argument("--labels", action="store_true",
                        help="Bake current LaTeX/TikZ labels into the figure")
    parser.add_argument("--label_fontsize", type=float, default=18.0)
    args = parser.parse_args()
    seed_all(args.seed)

    missing = [d for d in DIMS if not _available(MODEL_DIRS["reni_old"][d])]
    if missing:
        raise SystemExit(
            f"Original-RENI checkpoints for D={missing} are not in nerfstudio "
            f"format under checkpoints/reni_original/ (the committed dirs hold "
            f"the original-repo wandb dumps: RENI.pt). The paper's 'RENI "
            f"retrained' nerfstudio checkpoints are needed — fetch from the "
            f"cluster outputs (old_reni_models/latent_dim_*) or retrain.")

    old_specs = {f"old_{d}": MODEL_DIRS["reni_old"][d] for d in DIMS}
    new_specs = {f"new_{d}": MODEL_DIRS["reni_pp"][d] for d in DIMS}
    old_out = collect_model_outputs(old_specs, args.image_indices, args.device,
                                    height=args.height)
    new_out = collect_model_outputs(new_specs, args.image_indices, args.device,
                                    height=args.height)

    n_rows = len(args.image_indices)
    n_cols = len(DIMS) * 2 + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10))
    for i, idx in enumerate(args.image_indices):
        axes[i, 0].imshow(old_out[f"old_{DIMS[0]}"][idx]["gt_img"])
        axes[i, 0].axis("off"); axes[i, 0].set_aspect(1)
        for j, d in enumerate(DIMS):
            axes[i, j + 1].imshow(old_out[f"old_{d}"][idx]["pred_img"])
            axes[i, j + 1].axis("off"); axes[i, j + 1].set_aspect(1)
            axes[i, j + 1 + len(DIMS)].imshow(new_out[f"new_{d}"][idx]["pred_img"])
            axes[i, j + 1 + len(DIMS)].axis("off")
            axes[i, j + 1 + len(DIMS)].set_aspect(1)

    plt.tight_layout()
    if args.labels:
        _add_old_vs_new_labels(fig, axes, args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
