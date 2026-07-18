"""RENI++ vs SH vs SG comparison grid (paper Fig: comparison).

For each test envmap: GT image + log-heatmap on top, then one row per model
size showing RENI++ (image + heatmap), SH, and SG reconstructions.

    PYTHONPATH=. python scripts/figures/fig_comparison.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (MODEL_DIRS, add_common_args, add_figure_label,
                     axes_span_center_x, axis_center, collect_model_outputs,
                     save_figure, seed_all)

RENI_TAGS = ["latent_dim_9", "latent_dim_49", "latent_dim_100"]
SH_TAGS = ["2nd_order", "6th_order", "9th_order"]
SG_TAGS = ["num_param_30", "num_param_150", "num_param_300"]


def _add_comparison_labels(fig, quadrant_axes, fontsize):
    """Bake the labels that are currently added in LaTeX/TikZ."""
    for axes in quadrant_axes:
        first_row = axes["rows"][0]
        label_y = first_row["reni_img"].get_position().y1 + 0.012
        add_figure_label(
            fig,
            axes_span_center_x(first_row["reni_img"], first_row["reni_heatmap"]),
            label_y,
            "RENI++",
            fontsize,
        )
        add_figure_label(
            fig,
            axis_center(first_row["sh_img"])[0],
            label_y,
            "SH",
            fontsize,
        )
        add_figure_label(
            fig,
            axis_center(first_row["sg_img"])[0],
            label_y,
            "SG",
            fontsize,
        )

    for axes in (quadrant_axes[0], quadrant_axes[2]):
        left_x = axes["gt_img"].get_position().x0 - 0.017
        add_figure_label(
            fig,
            left_x,
            axis_center(axes["gt_img"])[1],
            "Ground Truth",
            fontsize,
            rotation=90,
        )
        for row, label in zip(axes["rows"], ("27", "147", "300")):
            add_figure_label(
                fig,
                left_x,
                axis_center(row["reni_img"])[1],
                label,
                fontsize,
                rotation=90,
            )


def plot_images_quadrants(output_images, image_indices, add_labels=False,
                          label_fontsize=22):
    fig = plt.figure(figsize=(26, 16))
    rows_per_idx = len(RENI_TAGS) + 2
    cols_per_model = 4
    quadrant_axes = []

    for i, idx in enumerate(image_indices):
        row_offset = (i // 2) * rows_per_idx
        col_offset = (i % 2) * cols_per_model
        axes = {"rows": []}

        gt_img = output_images[RENI_TAGS[0]][idx]["gt_img"]
        gt_heatmap = output_images[RENI_TAGS[0]][idx]["gt_heatmap"]
        h = gt_img.shape[0]
        padding = int(h * 0.10)
        padded_gt = np.pad(gt_img, [(0, padding), (0, 0), (0, 0)],
                           mode="constant", constant_values=1)
        padded_heat = np.pad(gt_heatmap, [(0, padding), (0, 0), (0, 0)],
                             mode="constant", constant_values=1)

        ax = plt.subplot2grid((2 * rows_per_idx, 2 * cols_per_model),
                              (row_offset, col_offset), rowspan=2, colspan=2)
        ax.imshow(padded_gt); ax.axis("off"); ax.set_aspect(1)
        axes["gt_img"] = ax
        ax = plt.subplot2grid((2 * rows_per_idx, 2 * cols_per_model),
                              (row_offset, col_offset + 2), rowspan=2, colspan=2)
        ax.imshow(padded_heat); ax.axis("off"); ax.set_aspect(1)
        axes["gt_heatmap"] = ax

        for j in range(3):
            row_axes = {}
            for col, tag_set, key, name in (
                (col_offset, RENI_TAGS, "pred_img", "reni_img"),
                (col_offset + 1, RENI_TAGS, "pred_heatmap", "reni_heatmap"),
                (col_offset + 2, SH_TAGS, "pred_img", "sh_img"),
                (col_offset + 3, SG_TAGS, "pred_img", "sg_img"),
            ):
                ax = plt.subplot2grid((2 * rows_per_idx, 2 * cols_per_model),
                                      (row_offset + 2 + j, col))
                ax.imshow(output_images[tag_set[j]][idx][key])
                ax.axis("off"); ax.set_aspect(1)
                row_axes[name] = ax
            axes["rows"].append(row_axes)
        quadrant_axes.append(axes)

    plt.tight_layout()
    if add_labels:
        _add_comparison_labels(fig, quadrant_axes, label_fontsize)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "comparison")
    parser.add_argument("--image_indices", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--labels", action="store_true",
                        help="Bake current LaTeX/TikZ labels into the figure")
    parser.add_argument("--reni-model", default="vnjoint_ortho_2cyc_testfit",
                        help="MODEL_DIRS key for ALL RENI++ rows (D=9/49/100); "
                             "use reni_pp for the paper ladder")
    parser.add_argument("--eval-width", type=int, default=512,
                        help="Eval image width (hi-res figure test set)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/home/james/data/RENI_HDR_hires_figs"),
                        help="Dataset root override for GT images")
    parser.add_argument("--label_fontsize", type=float, default=22.0)
    args = parser.parse_args()
    seed_all(args.seed)

    specs = {}
    for d, tag in zip((9, 49, 100), RENI_TAGS):
        specs[tag] = MODEL_DIRS[args.reni_model][d]
    print(f"[models] RENI++ ladder: {args.reni_model} (D=9/49/100)")
    for o, tag in zip(("2nd", "6th", "9th"), SH_TAGS):
        specs[tag] = MODEL_DIRS["sh"][o]
    for n, tag in zip((30, 150, 300), SG_TAGS):
        specs[tag] = MODEL_DIRS["sg"][n]

    # hi-res dataset only for the RENI rows (GT + refit targets); SH/SG
    # decode analytically from stored coefficients, and loading them at
    # 512px would cache the train split too (~20 GB per model, OOM).
    reni_specs = {k: v for k, v in specs.items() if k in RENI_TAGS}
    basis_specs = {k: v for k, v in specs.items() if k not in RENI_TAGS}
    outputs = collect_model_outputs(reni_specs, args.image_indices, args.device,
                                    height=args.height,
                                    eval_image_width=args.eval_width,
                                    data_override=args.data_dir)
    outputs.update(collect_model_outputs(basis_specs, args.image_indices,
                                         args.device, height=args.height))
    fig = plot_images_quadrants(outputs, args.image_indices,
                                add_labels=args.labels,
                                label_fontsize=args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
