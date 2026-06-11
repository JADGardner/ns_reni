"""Outpainting/completion figure (paper Fig: outpainting).

Default (--mask_mode perspective): each row masks the ground-truth envmap to
the footprint of a perspective camera on the sphere (a different viewing
direction per row), a fresh latent is fitted on the visible pixels with the
frozen RENI++ decoder, and the decoded completion is shown alongside. This
mirrors SphereJEPA's crop-to-sphere completion setting visually.

Columns: GT | full-fit reconstruction | perspective-masked GT | completion.

--mask_mode dataset reproduces the original paper figure (square masks via
the masked-fit checkpoint).

    PYTHONPATH=. python scripts/figures/fig_outpainting.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import (MODEL_DIRS, add_common_args, collect_model_outputs,
                     equirect_ray_bundle, load_model, make_ray_samples,
                     render_eval_image, save_figure, seed_all)
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB


def perspective_footprint_mask(height, width, yaw_deg, pitch_deg=0.0,
                               hfov_deg=90.0, vfov_deg=65.0):
    """Binary ERP mask of a perspective camera's footprint on the sphere.

    y-up equirectangular parameterisation matching the RENI_HDR layout:
    u in [0,1] -> azimuth theta in [-pi,pi), v in [0,1] -> elevation
    phi in [pi/2,-pi/2].
    """
    v, u = torch.meshgrid(torch.linspace(0, 1, height),
                          torch.linspace(0, 1, width), indexing="ij")
    theta = (u - 0.5) * 2 * np.pi
    phi = (0.5 - v) * np.pi
    d = torch.stack((torch.cos(phi) * torch.sin(theta), torch.sin(phi),
                     torch.cos(phi) * torch.cos(theta)), -1)  # [H, W, 3]

    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    forward = torch.tensor([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                            np.cos(pitch) * np.cos(yaw)], dtype=torch.float32)
    world_up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(forward, world_up)
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)

    df = d @ forward
    dr = d @ right
    du = d @ up
    tan_h = np.tan(np.deg2rad(hfov_deg) / 2)
    tan_v = np.tan(np.deg2rad(vfov_deg) / 2)
    mask = (df > 0) & (dr.abs() <= df * tan_h) & (du.abs() <= df * tan_v)
    return mask.float()  # [H, W]


def fit_latent(model, gt_norm, mask, device, steps=600, lr=1e-2,
               kld_weight=1e-4):
    """Fit a latent on the masked (visible) pixels with the frozen decoder.

    gt_norm: [H, W, 3] in the model's normalised (log) domain.
    mask:    [H, W] 1 = visible.
    """
    H, W = gt_norm.shape[:2]
    ray_bundle = equirect_ray_bundle(device, idx=0, height=H)
    ray_samples = make_ray_samples(model, ray_bundle)

    z = torch.zeros(1, model.field.latent_dim, 3, device=device,
                    requires_grad=True)
    target = gt_norm.reshape(-1, 3).to(device)
    m = mask.reshape(-1).to(device).bool()
    optimiser = torch.optim.Adam([z], lr=lr)

    for step in range(steps):
        optimiser.zero_grad()
        latents = z.repeat(ray_samples.shape[0], 1, 1)
        out = model.field.forward(ray_samples, rotation=None,
                                  latent_codes=latents)[RENIFieldHeadNames.RGB]
        mse = torch.nn.functional.mse_loss(out[m], target[m])
        cosine = 1 - torch.nn.functional.cosine_similarity(
            out[m], target[m], dim=-1).mean()
        kld = z.pow(2).mean()
        loss = 10.0 * mse + cosine + kld_weight * kld
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        latents = z.repeat(ray_samples.shape[0], 1, 1)
        out = model.field.forward(ray_samples, rotation=None,
                                  latent_codes=latents)[RENIFieldHeadNames.RGB]
        pred = model.field.unnormalise(out).reshape(H, W, 3)
    return linear_to_sRGB(pred, use_quantile=True).cpu().detach(), \
        float(mse.detach())


def perspective_figure(args):
    device = args.device
    _, datamanager, model = load_model(MODEL_DIRS["reni_pp"][100],
                                       device=device)
    n = len(args.image_indices)
    # Spread viewing directions over the sphere, one per row
    yaws = np.linspace(-150, 150, n)
    pitches = np.tile([10, -5, 0], (n + 2) // 3)[:n]

    fig, axes = plt.subplots(n, 4, figsize=(12, 1.7 * n))
    for i, idx in enumerate(args.image_indices):
        full = render_eval_image(model, datamanager, idx, device)
        gt_img = full["gt_img"]
        H, W = gt_img.shape[:2]
        mask = perspective_footprint_mask(H, W, yaws[i], pitches[i],
                                          args.hfov, args.vfov)

        batch = datamanager.eval_dataset[idx]
        gt_norm = batch["image"]
        if gt_norm.dim() == 4:
            gt_norm = gt_norm[0]
        completion, fit_mse = fit_latent(model, gt_norm, mask, device,
                                         steps=args.fit_steps)
        print(f"[fit] idx={idx} yaw={yaws[i]:+.0f} pitch={pitches[i]:+.0f} "
              f"masked-mse={fit_mse:.4f}")

        masked_gt = gt_img * mask.unsqueeze(-1)
        for col, img in enumerate(
                (gt_img, full["pred_img"], masked_gt, completion)):
            axes[i, col].imshow(img.numpy() if torch.is_tensor(img) else img)
            axes[i, col].axis("off")
            axes[i, col].set_aspect(1)

    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


def dataset_figure(args):
    """Original square-mask figure via the masked-fit checkpoint."""
    specs = {
        "full": MODEL_DIRS["reni_pp"][100],
        "masked": MODEL_DIRS["reni_pp_masked"][100],
    }
    outputs = collect_model_outputs(specs, args.image_indices, args.device,
                                    height=args.height)
    n_rows = len(args.image_indices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 10))
    for i, idx in enumerate(args.image_indices):
        for col, (key, img_key) in enumerate(
            (("full", "gt_img"), ("full", "pred_img"),
             ("masked", "gt_img"), ("masked", "pred_img"))):
            axes[i, col].imshow(outputs[key][idx][img_key])
            axes[i, col].axis("off")
            axes[i, col].set_aspect(1)
    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "outpainting")
    parser.add_argument("--image_indices", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--mask_mode", choices=["perspective", "dataset"],
                        default="perspective")
    parser.add_argument("--hfov", type=float, default=90.0,
                        help="Horizontal FoV of the footprint (deg)")
    parser.add_argument("--vfov", type=float, default=65.0,
                        help="Vertical FoV of the footprint (deg)")
    parser.add_argument("--fit_steps", type=int, default=600)
    args = parser.parse_args()
    seed_all(args.seed)

    if args.mask_mode == "perspective":
        perspective_figure(args)
    else:
        dataset_figure(args)


if __name__ == "__main__":
    main()
