"""InverseRenderNet vs RENI++ environment-map estimation comparison (thesis Ch.2).

Given the same 120x120 deg field-of-view crop of each RENI_HDR test environment
map, compare InverseRenderNet's 2nd-order SH lighting prediction against RENI++
outpainting (latent-code optimisation over the visible crop). Emits the 5-column
comparison figure (thesis Fig 2.x) and the LaTeX metrics table.

The evaluation pipeline is reused from ``publication/inversenet_vs_reni_comparison.py``;
only the figure and table are re-done in the shared fig_ style.

    PYTHONPATH=. python scripts/figures/fig_inversenet_comparison.py
    PYTHONPATH=. python scripts/figures/fig_inversenet_comparison.py --labels \
        --output publication/figures/inversenet_comparison_labeled

NaN-CELL FIX: the reused publication module imports LPIPS as
``from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity``,
but torchmetrics>=1.x exposes it at ``torchmetrics.image`` (module is ``lpip``,
not ``lpips``). The wrong path silently set ``HAS_LPIPS=False`` so LPIPS was
never computed and the table printed ``float('nan')``. Here we instantiate LPIPS
via the correct import and only emit the LPIPS column when it is genuinely
available and finite (otherwise the column is dropped, not filled with nan).

KNOWN COST: reusing ``run_evaluation`` re-fits the RENI++ eval latents once per
evaluated image (the fit optimises all eval latents jointly regardless of the
target index), so runtime scales with the number of figure rows x fit steps.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from _common import (REPO_ROOT, add_common_args, add_figure_label, axis_center,
                     save_figure, seed_all)
from reni.utils.colourspace import linear_to_sRGB

# torch>=2.6 defaults weights_only=True; the reused loader / nerfstudio trainer
# and the paper checkpoints (which pickle numpy scalars) predate that. Restore
# the old default for these trusted local checkpoints, matching _common.
_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):  # noqa: E302
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.insert(0, str(REPO_ROOT / "publication"))
from inversenet_vs_reni_comparison import InverseRenderNetVsRENI  # noqa: E402

COL_LABELS = ["Ground Truth", "Input Crop", "RENI++ Conditioning",
              "InverseRenderNet (SH)", "RENI++ (Outpainting)"]


def _make_deterministic_eval(dm):
    """Make next_eval_image deterministic (idx % n over the fixed-order loader).

    nerfstudio's VanillaDataManager.next_eval_image pops a RANDOM eval image and
    ignores the index argument, so the same image can appear on multiple figure
    rows (and the requested row index is meaningless). RENI's own datamanager
    resolves this with ``idx % len(eval_dataset)``; we reproduce that here so
    each requested row is a distinct, reproducible eval image and the fit loop
    cycles predictably through all images.
    """
    loader = getattr(dm, "fixed_indices_eval_dataloader", None)
    if loader is None:
        print("[eval] no fixed_indices_eval_dataloader; leaving random next_eval_image")
        return None
    pairs = list(loader)
    n = len(pairs)
    dm.next_eval_image = lambda idx: pairs[idx % n]
    print(f"[eval] deterministic next_eval_image over {n} fixed eval images")
    return n


def _resize_crop(crop_img, eq_h):
    """Resize an HDR perspective crop to the equirect height (as the original)."""
    crop_h, crop_w = crop_img.shape[:2]
    new_w = int(crop_w * (eq_h / crop_h))
    p999 = np.percentile(crop_img, 99.9) + 1e-8
    crop_ldr = (np.clip(crop_img / p999, 0, 1) * 255).astype(np.uint8)
    resized = np.array(
        Image.fromarray(crop_ldr).resize((new_w, eq_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0
    return resized * p999


def _plot(images_data, indices, add_labels=False, label_fontsize=15.0):
    n_rows = len(indices)
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(17, n_rows * 2.2),
        gridspec_kw={"width_ratios": [2, 1, 2, 2, 2]},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        gt_img = images_data["gt"][idx]
        cells = [
            gt_img,
            _resize_crop(images_data["crop"][idx], gt_img.shape[0]),
            images_data["conditioning"][idx],
            images_data["InverseRenderNet"][idx],
            images_data["RENI++"][idx],
        ]
        for col, img in enumerate(cells):
            display = np.clip(
                linear_to_sRGB(torch.from_numpy(img).float(), use_quantile=True).numpy(),
                0, 1)
            axes[row, col].imshow(display)
            axes[row, col].axis("off")

    plt.tight_layout()
    if add_labels:
        for col, text in enumerate(COL_LABELS):
            add_figure_label(
                fig,
                axis_center(axes[0, col])[0],
                axes[0, col].get_position().y1 + 0.006,
                text,
                label_fontsize,
            )
    return fig


def _write_table(metrics, out_stem: Path):
    # Only emit the LPIPS column if every method has a finite LPIPS value.
    have_lpips = all(
        "LPIPS" in m and math.isfinite(float(m["LPIPS"])) for m in metrics.values()
    ) and len(metrics) > 0
    if have_lpips:
        header = (r"Method & PSNR$\uparrow$ & LDR PSNR$\uparrow$ & "
                  r"SSIM$\uparrow$ & LPIPS$\downarrow$ \\")
        col_spec = r"\begin{tabular}{l|cccc}"
    else:
        print("[lpips] LPIPS unavailable/non-finite -> omitting LPIPS column")
        header = r"Method & PSNR$\uparrow$ & LDR PSNR$\uparrow$ & SSIM$\uparrow$ \\"
        col_spec = r"\begin{tabular}{l|ccc}"

    lines = [col_spec, r"\hline", header, r"\hline"]
    for method, m in metrics.items():
        row = f"{method} & {m['PSNR']:.2f} & {m['LDR_PSNR']:.2f} & {m['SSIM']:.4f}"
        if have_lpips:
            row += f" & {m['LPIPS']:.4f}"
        lines.append(row + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    table = "\n".join(lines)
    tex_path = Path(f"{out_stem}_metrics.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(table)
    print(f"[saved] {tex_path}")
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "inversenet_comparison")
    parser.add_argument("--data_dir", type=str, default="data/RENI_HDR/test")
    parser.add_argument("--inversenet_weights", type=str,
                        default="checkpoints/inverserendernet/model_ckpt.pth")
    parser.add_argument("--crop_fov", type=float, default=120.0)
    parser.add_argument("--crop_v_fov", type=float, default=120.0)
    parser.add_argument("--azimuth", type=float, default=80.0)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--reni_fit_steps", type=int, default=2500)
    parser.add_argument("--custom_val_folder", type=str, default=None)
    parser.add_argument("--image_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Test-image rows shown in the figure / metric set")
    parser.add_argument("--labels", action="store_true",
                        help="Bake current thesis TikZ column labels into the figure")
    parser.add_argument("--label_fontsize", type=float, default=15.0)
    args = parser.parse_args()
    seed_all(args.seed)

    # Fail clearly if the InverseRenderNet weights are missing rather than
    # silently running an uninitialised network (the class only warns).
    from reni.utils.checkpoint_locator import find_checkpoint
    try:
        find_checkpoint(args.inversenet_weights)
    except FileNotFoundError as e:
        raise SystemExit(f"InverseRenderNet weights not found: {e}")

    comparison = InverseRenderNetVsRENI(
        data_dir=args.data_dir,
        inversenet_weights=args.inversenet_weights,
        output_dir=str(args.output.parent),
        device=args.device,
        crop_h_fov=args.crop_fov,
        crop_v_fov=args.crop_v_fov,
        crop_azimuth=args.azimuth,
        crop_elevation=args.elevation,
        reni_fit_steps=args.reni_fit_steps,
        custom_val_folder=args.custom_val_folder,
    )
    if comparison.reni_model is None:
        raise SystemExit("RENI++ failed to load; see traceback above.")
    _make_deterministic_eval(comparison.datamanager)

    # Fix the nan LPIPS cell: instantiate LPIPS via the correct import path.
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        comparison.lpips = LearnedPerceptualImagePatchSimilarity(
            normalize=True).to(args.device)
        print("[lpips] enabled (torchmetrics.image.LearnedPerceptualImagePatchSimilarity)")
    except Exception as e:  # noqa: BLE001
        comparison.lpips = None
        print(f"[lpips] unavailable, column will be dropped: {e!r}")

    metrics, images_data = comparison.run_evaluation(image_indices=args.image_indices)

    n_available = len(images_data.get("gt", []))
    if n_available == 0:
        raise SystemExit("No test images were evaluated.")
    indices = list(range(n_available))

    fig = _plot(images_data, indices, add_labels=args.labels,
                label_fontsize=args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg, dpi=200)
    _write_table(metrics, args.output)


if __name__ == "__main__":
    main()
