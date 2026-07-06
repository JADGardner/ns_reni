"""RENI++ vs SOLD-Net vs Hosek-Wilkie baseline comparison (thesis Ch.2).

Reconstructs the sky hemisphere of each RENI_HDR test environment map with
RENI++ (D=100), the neural SOLD-Net baseline, and the analytical Hosek-Wilkie
sky model (fit per-image via L-BFGS-B), then emits the qualitative comparison
figure and the LaTeX metrics table (thesis Table 2.2).

By default this now uses the thesis two-bracket headline model
(``two_bracket_w3_1cyc_testfit``): a full-supervision reconstruction of the 21
test envmaps from the checkpoint's refit test latents. Two-bracket models are
fixed-gauge (scale-relative), so the RENI++ prediction is decoded via the
two-bracket blend (``model._to_linear_hdr``) and then exposure-aligned
(median-ratio) to the true-scale GT read from the raw test EXRs. SOLD-Net and
Hosek-Wilkie are UNCHANGED and evaluated against that same true-scale GT, so
their numbers are unaffected by the RENI++ swap.

Pass ``--model reni_pp`` to reproduce the original paper version (log/min-max
RENI++ via publication/baseline_comparison.py, no exposure alignment).

    PYTHONPATH=.:scripts/figures python scripts/figures/fig_baseline_comparison.py --labels
    PYTHONPATH=.:scripts/figures python scripts/figures/fig_baseline_comparison.py \
        --model reni_pp --output publication/figures/baseline_comparison
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from _common import (MODEL_DIRS, REPO_ROOT, add_common_args, add_figure_label,
                     axis_center, equirect_ray_bundle, load_model,
                     read_clean_exr, save_figure, seed_all)
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
from baseline_comparison import (BaselineComparison, compute_metrics,  # noqa: E402
                                 logger)
from reni.baselines.soldnet import SOLDNetGlobalModel  # noqa: E402
from reni.baselines.hosek_wilkie import HosekWilkieSkyModel  # noqa: E402
from reni.utils.checkpoint_locator import find_checkpoint  # noqa: E402

# images_data keys (as produced by BaselineComparison.run_evaluation) and the
# thesis TikZ column labels that sit above them.
METHOD_KEYS = ["gt", "RENI++", "SOLD-Net", "Hosek-Wilkie"]
COL_LABELS = ["GT", "RENI++", "SOLD-Net", "Hosek-Wilkie"]


class TwoBracketBaselineComparison(BaselineComparison):
    """Baseline comparison driven by a fixed-gauge two-bracket RENI++ model.

    Overrides only the RENI++ loading and the GT/RENI reconstruction path;
    SOLD-Net, Hosek-Wilkie, the metric function and the sky-hemisphere protocol
    are inherited verbatim. The RENI++ reconstruction is decoded through the
    two-bracket blend and exposure-aligned (median luminance ratio) to the
    true-scale GT, because two-bracket models carry no absolute scale.
    """

    def __init__(self, reni_model_dir, soldnet_checkpoint="checkpoints/SOLD_Net/pretrained_model",
                 output_dir="publication/figures", device="cuda:0"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        logger.info(f"Loading two-bracket RENI++ from {reni_model_dir} ...")
        self.reni_pipeline, self.reni_datamanager, self.reni_model = \
            load_model(Path(reni_model_dir), device=device)
        self.reni_model.eval()
        if not getattr(self.reni_model, "two_bracket", False):
            logger.warning("Loaded RENI++ model is NOT two-bracket; predictions "
                           "will still be exposure-aligned to GT.")

        # Baselines (unchanged from the paper comparison).
        logger.info("Loading SOLD-Net...")
        self.soldnet = SOLDNetGlobalModel(device=device)
        self.soldnet.load_pretrained(str(find_checkpoint(soldnet_checkpoint)))
        logger.info("Loading Hosek-Wilkie...")
        self.hosek = HosekWilkieSkyModel()

        self.n_eval_images = len(self.reni_datamanager.eval_dataset)
        logger.info(f"Found {self.n_eval_images} evaluation images")

    def _gt_true_hdr(self, idx, H, W):
        """True-scale linear-HDR GT [H, W, 3] from the raw test EXR (matches the
        paper's field.unnormalise GT; the raw RENI_HDR test EXRs are 64x128)."""
        name = self.reni_datamanager.eval_dataset._dataparser_outputs.image_filenames[idx]
        img = read_clean_exr(Path(name))
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).float().to(self.device)

    def _reni_hdr_prediction(self, idx, H, W):
        """RENI++ reconstruction [H, W, 3] in linear HDR (fixed gauge) using the
        checkpoint's fitted eval latent for image idx."""
        ray_bundle = equirect_ray_bundle(self.device, idx=idx, height=H)
        with torch.no_grad():
            outputs = self.reni_model.get_outputs_for_camera_ray_bundle(ray_bundle, rotation=None)
            raw = outputs["rgb"]
            # camera-bundle path can view flat [N,6] two-bracket output as [N,3,2];
            # memory order is intact so reshape(-1,6) restores the true layout.
            if getattr(self.reni_model, "two_bracket", False) and raw.shape[-1] != 6:
                raw = raw.reshape(-1, 6)
            pred = self.reni_model._to_linear_hdr(raw).reshape(H, W, 3)
        return pred

    @torch.no_grad()
    def run_evaluation(self, num_images=None):
        n_images = min(num_images, self.n_eval_images) if num_images else self.n_eval_images
        H_full, W, H_sky = 64, 128, 32
        metrics = {"RENI++": [], "SOLD-Net": [], "Hosek-Wilkie": []}
        images_data = {"gt": [], "RENI++": [], "SOLD-Net": [], "Hosek-Wilkie": []}

        for i in tqdm(range(n_images), desc="Evaluating"):
            gt_hdr = self._gt_true_hdr(i, H_full, W)              # true scale
            reni_hdr = self._reni_hdr_prediction(i, H_full, W)    # fixed gauge

            gt_sky_t = gt_hdr[:H_sky, :, :]
            reni_sky_t = reni_hdr[:H_sky, :, :]
            # Two-bracket models are scale-relative (fixed gauge), so give the
            # prediction the single per-image exposure that best matches the
            # true-scale GT over the scored region. This is the MSE-optimal
            # least-squares scale used by reni_model.get_image_metrics_and_images
            # for scale-invariant models; it isolates sky appearance fidelity
            # from the absolute-scale degree of freedom the model discards.
            scale = (gt_sky_t * reni_sky_t).sum() / (reni_sky_t * reni_sky_t).sum().clamp_min(1e-8)
            reni_sky_t = scale * reni_sky_t

            gt_sky = gt_sky_t.cpu().numpy()
            reni_sky = reni_sky_t.cpu().numpy()

            sold_sky = self.evaluate_soldnet_single(gt_sky)
            hosek_sky = self.evaluate_hosek_fitted_single(gt_sky)

            images_data["gt"].append(gt_sky)
            images_data["RENI++"].append(reni_sky)
            images_data["SOLD-Net"].append(sold_sky)
            images_data["Hosek-Wilkie"].append(hosek_sky)

            metrics["RENI++"].append(compute_metrics(gt_sky, reni_sky))
            metrics["SOLD-Net"].append(compute_metrics(gt_sky, sold_sky))
            metrics["Hosek-Wilkie"].append(compute_metrics(gt_sky, hosek_sky))

        avg_metrics = {}
        for method, method_metrics in metrics.items():
            avg_metrics[method] = {
                key: float(np.mean([m[key] for m in method_metrics]))
                for key in method_metrics[0].keys()
            }
        return avg_metrics, images_data


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
    add_common_args(parser, "baseline_comparison_thesis")
    parser.add_argument("--model", default="two_bracket_w3_1cyc_testfit",
                        help="MODEL_DIRS key for the RENI++ reconstruction "
                             "(default: thesis two-bracket 1-cycle test-fit shim; "
                             "use reni_pp for the original paper model)")
    parser.add_argument("--table_output", type=Path,
                        default=REPO_ROOT / "publication" / "tables" / "baseline_comparison_thesis",
                        help="Output stem for the LaTeX metrics table")
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

    if args.model == "reni_pp":
        print("[model] paper RENI++ (log/min-max, no exposure alignment)")
        comparison = BaselineComparison(
            output_dir=str(args.output.parent),
            device=args.device,
        )
    else:
        model_dir = MODEL_DIRS[args.model][100]
        print(f"[model] two-bracket RENI++: {args.model} -> {model_dir}")
        comparison = TwoBracketBaselineComparison(
            reni_model_dir=model_dir,
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
    _write_table(metrics, args.table_output)


if __name__ == "__main__":
    main()
