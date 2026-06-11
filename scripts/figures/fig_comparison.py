"""RENI++ vs SH vs SG comparison grid (paper Fig: comparison).

For each test envmap: GT image + log-heatmap on top, then one row per model
size showing RENI++ (image + heatmap), SH, and SG reconstructions.

    PYTHONPATH=. python scripts/figures/fig_comparison.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _common import (MODEL_DIRS, add_common_args, collect_model_outputs,
                     save_figure, seed_all)

RENI_TAGS = ["latent_dim_9", "latent_dim_49", "latent_dim_100"]
SH_TAGS = ["2nd_order", "6th_order", "9th_order"]
SG_TAGS = ["num_param_30", "num_param_150", "num_param_300"]


def plot_images_quadrants(output_images, image_indices):
    fig = plt.figure(figsize=(26, 16))
    rows_per_idx = len(RENI_TAGS) + 2
    cols_per_model = 4

    for i, idx in enumerate(image_indices):
        row_offset = (i // 2) * rows_per_idx
        col_offset = (i % 2) * cols_per_model

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
        ax = plt.subplot2grid((2 * rows_per_idx, 2 * cols_per_model),
                              (row_offset, col_offset + 2), rowspan=2, colspan=2)
        ax.imshow(padded_heat); ax.axis("off"); ax.set_aspect(1)

        for j in range(3):
            for col, tag_set, key in (
                (col_offset, RENI_TAGS, "pred_img"),
                (col_offset + 1, RENI_TAGS, "pred_heatmap"),
                (col_offset + 2, SH_TAGS, "pred_img"),
                (col_offset + 3, SG_TAGS, "pred_img"),
            ):
                ax = plt.subplot2grid((2 * rows_per_idx, 2 * cols_per_model),
                                      (row_offset + 2 + j, col))
                ax.imshow(output_images[tag_set[j]][idx][key])
                ax.axis("off"); ax.set_aspect(1)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "comparison")
    parser.add_argument("--image_indices", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    seed_all(args.seed)

    specs = {}
    for d, tag in zip((9, 49, 100), RENI_TAGS):
        specs[tag] = MODEL_DIRS["reni_pp"][d]
    for o, tag in zip(("2nd", "6th", "9th"), SH_TAGS):
        specs[tag] = MODEL_DIRS["sh"][o]
    for n, tag in zip((30, 150, 300), SG_TAGS):
        specs[tag] = MODEL_DIRS["sg"][n]

    outputs = collect_model_outputs(specs, args.image_indices, args.device,
                                    height=args.height)
    fig = plot_images_quadrants(outputs, args.image_indices)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
