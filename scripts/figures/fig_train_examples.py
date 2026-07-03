"""Random training-set reconstructions with log-luminance heatmaps.

For N random training images: GT envmap | reconstruction from the FITTED train
latent | GT log heatmap | reconstruction log heatmap. GT comes from the run's
own train dataset (same gauge/target pipeline as training; 6-channel two-bracket
targets are blended back to linear HDR), so latent<->image pairing is exact.
Heatmaps are log1p(BT.709 luminance) with the colour scale set by the GT panel,
matching the fixed wandb heatmap convention.

    PYTHONPATH=. python scripts/figures/fig_train_examples.py \
        --model_dir outputs/reni_latent_reset_d100_two_bracket/reni/<ts> \
        --num 10 --height 128
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _common import (add_common_args, equirect_ray_bundle, load_model,
                     make_ray_samples, save_figure, seed_all)
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import luminance, two_bracket_to_linear


def decode_hdr(model, ray_samples, latent, chunk_size=65536):
    """Decode a latent to LINEAR HDR [N, 3] (two-bracket aware)."""
    chunks = []
    with torch.no_grad():
        for start in range(0, ray_samples.shape[0], chunk_size):
            chunk = ray_samples[start:start + chunk_size]
            out = model.field.forward(chunk, latent_codes=latent.repeat(chunk.shape[0], 1, 1))
            raw = out[RENIFieldHeadNames.RGB]
            if getattr(model, "two_bracket", False):
                hdr = two_bracket_to_linear(
                    raw, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
            else:
                hdr = model.field.unnormalise(raw)
            chunks.append(hdr.cpu())
    return torch.cat(chunks, dim=0)


def to_hdr_image(model, image):
    """Dataset target [H, W, C] -> linear HDR [H, W, 3]."""
    if image.shape[-1] == 6:
        return two_bracket_to_linear(
            image, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
    return model.field.unnormalise(image)


def log_heat(hdr):
    return torch.log1p(luminance(hdr).clamp_min(0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "train_examples")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args()
    seed_all(args.seed)

    _, dm, model = load_model(args.model_dir, device=args.device)
    ray_bundle = equirect_ray_bundle(args.device, idx=0, height=args.height)
    ray_samples = make_ray_samples(model, ray_bundle)

    train_mu = model.field.train_mu.detach()
    idxs = torch.randperm(train_mu.shape[0])[: args.num].tolist()

    cols = ["GT", "Reconstruction", "GT log heatmap", "Recon log heatmap"]
    fig, axs = plt.subplots(args.num, 4, figsize=(13, 1.7 * args.num),
                            constrained_layout=True)
    H, W = args.height, args.height * 2
    for r, i in enumerate(idxs):
        gt_img = dm.train_dataset[i]["image"]
        gt_hdr = to_hdr_image(model, gt_img)
        pred_hdr = decode_hdr(model, ray_samples, train_mu[i].unsqueeze(0)).reshape(H, W, 3)

        gt_srgb = linear_to_sRGB(gt_hdr, use_quantile=True).clamp(0, 1)
        pred_srgb = linear_to_sRGB(pred_hdr, use_quantile=True).clamp(0, 1)
        gt_lh, pred_lh = log_heat(gt_hdr), log_heat(pred_hdr)
        vmin, vmax = float(gt_lh.min()), float(gt_lh.max())

        mse = torch.mean((pred_srgb.mean(0).mean(0) - gt_srgb.mean(0).mean(0)) ** 2)
        print(f"[{r}] train idx {i}: gt {tuple(gt_img.shape)}, "
              f"pred p99 lum {float(luminance(pred_hdr).quantile(0.99)):.3f}, "
              f"srgb mean diff {float(mse):.5f}")

        panels = [gt_srgb.numpy(), pred_srgb.numpy()]
        for c, img in enumerate(panels):
            axs[r, c].imshow(img)
        axs[r, 2].imshow(gt_lh.numpy(), cmap="turbo", vmin=vmin, vmax=vmax)
        axs[r, 3].imshow(pred_lh.numpy(), cmap="turbo", vmin=vmin, vmax=vmax)
        for c in range(4):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(cols[c], fontsize=11)
        axs[r, 0].set_ylabel(f"train {i}", fontsize=8)

    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
