"""InverseRenderNet vs RENI++ environment-map estimation comparison (thesis Ch.2).

Given the same 120x120 deg field-of-view crop of each RENI_HDR test environment
map, compare InverseRenderNet's 2nd-order SH lighting prediction against RENI++
crop outpainting (latent-code optimisation over the visible crop). Emits the
5-column comparison figure (thesis Fig 2.x) and the LaTeX metrics table.

RENI++ side (thesis two-bracket, default two_bracket_w3_2cyc, the completion-
optimal model): a fresh latent is optimised on the visible crop in the model's
two-bracket target space (bracket-space MSE + cosine on the visible pixels, the
frozen decoder), decoded to linear HDR via the two-bracket blend
(model._to_linear_hdr) and given a per-image least-squares exposure scale
against the true-scale GT. This mirrors scripts/figures/fig_outpainting.py's
fit_latent and replicates the trainable per-image scale the original paper fit
used. The InverseRenderNet (SH) baseline, the FoV crop extraction and the metric
function are reused verbatim from publication/inversenet_vs_reni_comparison.py,
so the SH numbers are unchanged; both methods are scored against the same
true-scale GT read from the raw test EXRs.

    PYTHONPATH=.:scripts/figures python scripts/figures/fig_inversenet_comparison.py --labels
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from _common import (init_fit_latent,
                     MODEL_DIRS, REPO_ROOT, add_common_args, add_figure_label,
                     axis_center, equirect_ray_bundle, load_model,
                     read_clean_exr, save_figure, seed_all)
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import apply_fixed_gauge, encode_two_bracket

# torch>=2.6 defaults weights_only=True; the reused loader / nerfstudio trainer
# and the paper checkpoints (which pickle numpy scalars) predate that. Restore
# the old default for these trusted local checkpoints, matching _common.
_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):  # noqa: E302
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.insert(0, str(REPO_ROOT / "publication"))
# Reused, UNCHANGED InverseRenderNet baseline machinery.
from inversenet_vs_reni_comparison import (InverseRenderNetVsRENI,  # noqa: E402
                                          compute_metrics, extract_fov_crop,
                                          generate_fov_mask)
from reni.baselines.inversenet import (InverseRenderNet,  # noqa: E402
                                       load_pytorch_weights)
from reni.utils.checkpoint_locator import find_checkpoint  # noqa: E402

COL_LABELS = ["Ground Truth", "Input Crop", "RENI++ Conditioning",
              "InverseRenderNet (SH)", "RENI++ (Outpainting)"]


def _make_irn(inversenet_weights, device, envmap_width):
    """InverseRenderNet baseline via the reused class's run_inversenet (SH
    remapping unchanged), without triggering its RENI++ trainer setup."""
    irn = InverseRenderNetVsRENI.__new__(InverseRenderNetVsRENI)
    irn.device = device
    irn.envmap_width = envmap_width
    irn.envmap_height = envmap_width // 2
    irn.inversenet = InverseRenderNet().to(device)
    irn.inversenet.eval()
    load_pytorch_weights(irn.inversenet, str(find_checkpoint(inversenet_weights)))
    return irn


def _fit_reni_latent(model, gt_target, mask, device, steps=2500, lr=1e-2,
                     kld_weight=1e-4, decode_chunk=65536):
    """Optimise a single latent over the visible (masked) pixels and decode the
    full envmap in linear HDR.

    Mirrors fig_outpainting.fit_latent (bracket-space MSE + cosine on the frozen
    decoder), but returns the linear-HDR envmap so the caller can exposure-align
    and score it. gt_target is the model's normalised target [H, W, C] (six
    channels for two-bracket, three otherwise); mask is [H, W] with 1 = visible.
    """
    H, W = gt_target.shape[:2]
    ray_bundle = equirect_ray_bundle(device, idx=0, height=H)
    z = init_fit_latent(model, device)
    target = gt_target.reshape(-1, gt_target.shape[-1]).to(device)
    visible = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)
    if visible.numel() == 0:
        raise ValueError("FoV mask contains no visible rays")
    optimiser = torch.optim.Adam([z], lr=lr)

    for _ in range(steps):
        optimiser.zero_grad()
        fit_samples = model.create_ray_samples(
            ray_bundle.origins[visible],
            ray_bundle.directions[visible],
            ray_bundle.camera_indices[visible],
        )
        latents = z.repeat(fit_samples.shape[0], 1, 1)
        out = model.field.forward(fit_samples, rotation=None,
                                  latent_codes=latents)[RENIFieldHeadNames.RGB]
        mse = F.mse_loss(out, target[visible])
        cosine = 1 - F.cosine_similarity(out, target[visible], dim=-1).mean()
        kld = z.pow(2).mean()
        loss = 10.0 * mse + cosine + kld_weight * kld
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        chunks = []
        for start in range(0, len(ray_bundle), decode_chunk):
            end = start + decode_chunk
            sample_chunk = model.create_ray_samples(
                ray_bundle.origins[start:end],
                ray_bundle.directions[start:end],
                ray_bundle.camera_indices[start:end],
            )
            latents = z.repeat(sample_chunk.shape[0], 1, 1)
            chunks.append(model.field.forward(
                sample_chunk, rotation=None,
                latent_codes=latents)[RENIFieldHeadNames.RGB])
        out = torch.cat(chunks, dim=0)
        if getattr(model, "two_bracket", False):
            if out.shape[-1] != 6:
                out = out.reshape(-1, 6)
            pred = model._to_linear_hdr(out).reshape(H, W, 3)
        else:
            pred = model.field.unnormalise(out).reshape(H, W, 3)
    return pred


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
    add_common_args(parser, "inversenet_comparison_published")
    parser.add_argument(
        "--model",
        default="reni_pp",
        help=(
            "MODEL_DIRS key for the RENI++ decoder (default: published "
            "RENI++; use vnjoint_ortho_2cyc for the thesis model)"
        ),
    )
    parser.add_argument("--data_dir", type=str, default="data/RENI_HDR/test")
    parser.add_argument("--inversenet_weights", type=str,
                        default="checkpoints/inverserendernet/model_ckpt.pth")
    parser.add_argument("--table_output", type=Path,
                        default=REPO_ROOT / "publication" / "tables"
                        / "inversenet_comparison_published",
                        help="Output stem for the LaTeX metrics table")
    parser.add_argument("--crop_fov", type=float, default=120.0)
    parser.add_argument("--crop_v_fov", type=float, default=120.0)
    parser.add_argument("--azimuth", type=float, default=80.0)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--reni_fit_steps", type=int, default=2500)
    parser.add_argument("--image_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Test-image rows shown in the figure / metric set")
    parser.add_argument("--labels", action="store_true",
                        help="Bake current thesis TikZ column labels into the figure")
    parser.add_argument("--label_fontsize", type=float, default=15.0)
    args = parser.parse_args()
    seed_all(args.seed)

    device = args.device
    data_dir = (REPO_ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() \
        else Path(args.data_dir)
    exr_paths = sorted(data_dir.glob("*.exr"))
    if not exr_paths:
        raise SystemExit(f"No test EXRs under {data_dir}")
    indices = [i for i in args.image_indices if i < len(exr_paths)]
    if not indices:
        indices = list(range(min(5, len(exr_paths))))

    model_dir = MODEL_DIRS[args.model][100]
    print(f"[model] RENI++ decoder: {args.model} -> {model_dir}")
    _, _, model = load_model(model_dir, device=device)
    model.eval()

    irn = _make_irn(args.inversenet_weights, device, envmap_width=128)

    # LPIPS for compute_metrics (correct torchmetrics path; the reused module's
    # import path was wrong, silently dropping the column).
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        print("[lpips] enabled")
    except Exception as e:  # noqa: BLE001
        lpips = None
        print(f"[lpips] unavailable, column will be dropped: {e!r}")

    W = 128
    H = W // 2
    metrics = {"InverseRenderNet": [], "RENI++": []}
    images_data = {"gt": [], "crop": [], "conditioning": [],
                   "InverseRenderNet": [], "RENI++": []}

    for i in indices:
        raw = read_clean_exr(exr_paths[i])  # true-scale linear HDR [64,128,3]
        raw_t = torch.from_numpy(raw)

        crop, _ = extract_fov_crop(
            raw, h_fov_deg=args.crop_fov, v_fov_deg=args.crop_v_fov,
            azimuth_deg=args.azimuth, elevation_deg=args.elevation,
            output_size=(args.crop_size, args.crop_size))
        mask_np = generate_fov_mask(
            (H, W), h_fov_deg=args.crop_fov, v_fov_deg=args.crop_v_fov,
            azimuth_deg=args.azimuth, elevation_deg=args.elevation)
        mask_t = torch.from_numpy(mask_np)

        # InverseRenderNet (SH) baseline — unchanged.
        inversenet_envmap = irn.run_inversenet(crop, target_width=W)

        # RENI++ crop outpainting in the model's two-bracket target space.
        if getattr(model, "two_bracket", False):
            gauge = apply_fixed_gauge(raw_t, percentile=0.99, target=1.0)
            gt_target = encode_two_bracket(gauge, m_ldr=model.tonemap_m_ldr,
                                           m_log=model.tonemap_m_log)
        else:
            gt_target = raw_t  # (paper path would need its own normalisation)
        reni_pred = _fit_reni_latent(model, gt_target, mask_t, device,
                                     steps=args.reni_fit_steps)
        # Per-image least-squares exposure scale to the true-scale GT (the
        # scale-relative model's analogue of the paper fit's trainable scale).
        raw_dev = raw_t.to(device)
        scale = (raw_dev * reni_pred).sum() / (reni_pred * reni_pred).sum().clamp_min(1e-8)
        reni_envmap = (scale * reni_pred).cpu().numpy()

        conditioning = raw * mask_np[..., np.newaxis]

        images_data["gt"].append(raw)
        images_data["crop"].append(crop)
        images_data["conditioning"].append(conditioning)
        images_data["InverseRenderNet"].append(inversenet_envmap)
        images_data["RENI++"].append(reni_envmap)

        metrics["InverseRenderNet"].append(
            compute_metrics(raw, inversenet_envmap, lpips, device))
        metrics["RENI++"].append(
            compute_metrics(raw, reni_envmap, lpips, device))
        print(f"[eval] {exr_paths[i].name}: "
              f"IRN PSNR={metrics['InverseRenderNet'][-1]['PSNR']:.2f} "
              f"RENI++ PSNR={metrics['RENI++'][-1]['PSNR']:.2f}")

    avg_metrics = {}
    for method, method_metrics in metrics.items():
        avg_metrics[method] = {
            key: float(np.mean([m[key] for m in method_metrics]))
            for key in method_metrics[0].keys()
        }

    # figure rows are the evaluated images in order (images_data is idx 0..n-1)
    fig = _plot(images_data, list(range(len(indices))), add_labels=args.labels,
                label_fontsize=args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg, dpi=200)
    _write_table(avg_metrics, args.table_output)


if __name__ == "__main__":
    main()
