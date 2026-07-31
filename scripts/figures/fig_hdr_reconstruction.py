"""HDR reconstruction: RENI++ vs the two-bracket model (thesis Ch2).

For each chosen test environment: the ground truth, the RENI++ refit and
the two-bracket refit, each shown tonemapped (sRGB) and as log-HDR
luminance (the heat cells make the sun's reconstructed intensity visible,
which tonemapped views hide).

    PYTHONPATH=. python scripts/figures/fig_hdr_reconstruction.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _common import (MODEL_DIRS, REPO_ROOT, add_common_args, load_model,
                     render_eval_image, save_figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "hdr_reconstruction")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--indices", type=int, nargs="+", default=[2, 11, 17])
    parser.add_argument("--models", nargs=2,
                        default=["reni_pp", "two_bracket_w3_1cyc_testfit"])
    parser.add_argument("--labels", nargs=2,
                        default=["RENI++", "Two-Bracket\n(Single Cycle)"])
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "data" / "RENI_HDR",
                        help="RENI HDR dataset root; optional high-resolution "
                             "images are resolved by the shared mapping")
    parser.add_argument("--eval-width", type=int, default=512)
    args = parser.parse_args()

    n_idx = len(args.indices)
    rows = 1 + len(args.models)
    fig, axs = plt.subplots(rows, 2 * n_idx,
                            figsize=(3.4 * n_idx * 2, 1.9 * rows))

    results = {}
    for key in args.models:
        _, datamanager, model = load_model(MODEL_DIRS[key][args.latent_dim],
                                           device=args.device,
                                           eval_image_width=args.eval_width,
                                           data_override=args.data_dir)
        for idx in args.indices:
            results[(key, idx)] = render_eval_image(model, datamanager, idx,
                                                    args.device)
        del model
        torch.cuda.empty_cache()

    def cell(ax, img):
        ax.imshow(img.cpu().numpy() if torch.is_tensor(img) else img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for j, idx in enumerate(args.indices):
        ref = results[(args.models[0], idx)]
        cell(axs[0, 2 * j], ref["gt_img"])
        cell(axs[0, 2 * j + 1], ref["gt_heatmap"])
        axs[0, 2 * j].set_title("Tonemapped", fontsize=10)
        axs[0, 2 * j + 1].set_title("Log-HDR Luminance", fontsize=10)
        for i, key in enumerate(args.models):
            out = results[(key, idx)]
            cell(axs[1 + i, 2 * j], out["pred_img"])
            cell(axs[1 + i, 2 * j + 1], out["pred_heatmap"])

    axs[0, 0].set_ylabel("Ground Truth", fontsize=11)
    for i, label in enumerate(args.labels):
        axs[1 + i, 0].set_ylabel(label, fontsize=11)

    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
