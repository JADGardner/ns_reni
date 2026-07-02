"""RENI++ vs SOLD-Net vs Hosek-Wilkie baseline comparison (thesis Ch.2).

Reconstructs the sky hemisphere of each RENI_HDR test environment map with
RENI++ (D=300), the neural SOLD-Net baseline, and the analytical Hosek-Wilkie
sky model (fit per-image via L-BFGS-B), then emits the qualitative comparison
figure and the LaTeX metrics table (thesis Table 2.2).

The evaluation pipeline is reused verbatim from
``publication/baseline_comparison.py`` (identical GT/prediction handling and
metrics); only the figure rendering and table emission are re-done in the
shared fig_ style so the output lands in ``publication/figures/`` and can bake
the thesis TikZ column labels with ``--labels``.

    PYTHONPATH=. python scripts/figures/fig_baseline_comparison.py
    PYTHONPATH=. python scripts/figures/fig_baseline_comparison.py --labels \
        --output publication/figures/baseline_comparison_labeled
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import (REPO_ROOT, add_common_args, add_figure_label, axis_center,
                     save_figure, seed_all)
from reni.utils.colourspace import linear_to_sRGB

# torch>=2.6 defaults weights_only=True; the reused generate_figures loader (and
# the paper checkpoints, which pickle numpy scalars) predate that. Restore the
# old default for these trusted local checkpoints, matching _common.load_model.
_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):  # noqa: E302
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.insert(0, str(REPO_ROOT / "publication"))
from baseline_comparison import BaselineComparison  # noqa: E402

# images_data keys (as produced by BaselineComparison.run_evaluation) and the
# thesis TikZ column labels that sit above them.
METHOD_KEYS = ["gt", "RENI++", "SOLD-Net", "Hosek-Wilkie"]
COL_LABELS = ["GT", "RENI++", "SOLD-Net", "Hosek-Wilkie"]


def _plot(images_data, indices, add_labels=False, label_fontsize=15.0):
    n_rows, n_cols = len(indices), len(METHOD_KEYS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 0.65),
        gridspec_kw={"wspace": 0.02, "hspace": 0.0},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        for col, key in enumerate(METHOD_KEYS):
            img = torch.from_numpy(images_data[key][idx]).float()
            display = np.clip(linear_to_sRGB(img, use_quantile=True).numpy(), 0, 1)
            axes[row, col].imshow(display)
            axes[row, col].axis("off")

    plt.tight_layout()
    if add_labels:
        for col, text in enumerate(COL_LABELS):
            add_figure_label(
                fig,
                axis_center(axes[0, col])[0],
                axes[0, col].get_position().y1 + 0.01,
                text,
                label_fontsize,
            )
    return fig


def _write_table(metrics, out_stem: Path):
    lines = [
        r"\begin{tabular}{l|ccccc}",
        r"\hline",
        r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & MSE$\downarrow$ & "
        r"LogMSE$\downarrow$ & LDR PSNR$\uparrow$ \\",
        r"\hline",
    ]
    for method, m in metrics.items():
        lines.append(
            f"{method} & {m['PSNR']:.2f} & {m['SSIM']:.4f} & {m['MSE']:.4f} & "
            f"{m['LogMSE']:.4f} & {m['LDR_PSNR']:.2f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    table = "\n".join(lines)
    tex_path = Path(f"{out_stem}_metrics.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(table)
    print(f"[saved] {tex_path}")
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "baseline_comparison")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Limit metric evaluation to the first N test images")
    parser.add_argument("--image_indices", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4],
                        help="Test-image rows shown in the figure")
    parser.add_argument("--labels", action="store_true",
                        help="Bake current thesis TikZ column labels into the figure")
    parser.add_argument("--label_fontsize", type=float, default=15.0)
    args = parser.parse_args()
    seed_all(args.seed)

    comparison = BaselineComparison(
        output_dir=str(args.output.parent),
        device=args.device,
    )
    metrics, images_data = comparison.run_evaluation(num_images=args.num_images)

    n_available = len(images_data["gt"])
    if n_available == 0:
        raise SystemExit("No test images were evaluated; check data/RENI_HDR/test.")
    indices = [i for i in args.image_indices if i < n_available]
    if not indices:
        indices = list(range(min(5, n_available)))

    fig = _plot(images_data, indices, add_labels=args.labels,
                label_fontsize=args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg, dpi=300)
    _write_table(metrics, args.output)


if __name__ == "__main__":
    main()
