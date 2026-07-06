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
                     add_figure_label, axis_center, equirect_ray_bundle,
                     eval_image_tensor, load_model, render_eval_image,
                     save_figure, seed_all)
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB

COLUMN_LABELS = (
    "Ground\nTruth",
    "RENI++\nFull",
    "Masked\nGround Truth",
    "RENI++\nOutpainting",
)

DEFAULT_ROW_RPY = (
    (3.820935, -6.460099, 175.683607),
    (-30.532792, 21.338249, 68.085577),
    (22.934182, 7.193256, -10.635712),
    (9.216508, -13.638064, -10.635712),
    (18.066142, -2.829290, 68.085577),
    (-10.183182, -13.638064, -93.300564),
)


def _add_column_labels(fig, axes, fontsize):
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    label_y = axes[0, 0].get_position().y1 + 0.020
    for col, label in enumerate(COLUMN_LABELS):
        add_figure_label(fig, axis_center(axes[0, col])[0], label_y, label,
                         fontsize)


def sample_view_angles(n, seed, pitch_range_deg=25.0, roll_range_deg=35.0):
    """Stratified-random camera angles: broad yaw coverage, shuffled order."""
    rng = np.random.default_rng(seed)
    yaw_step = 360.0 / max(n, 1)
    yaws = np.linspace(-180.0, 180.0, n, endpoint=False) + yaw_step * 0.5
    yaws += rng.uniform(-0.45 * yaw_step, 0.45 * yaw_step, size=n)
    rng.shuffle(yaws)
    yaws = ((yaws + 180.0) % 360.0) - 180.0
    pitches = rng.uniform(-pitch_range_deg, pitch_range_deg, size=n)
    rolls = rng.uniform(-roll_range_deg, roll_range_deg, size=n)
    return yaws, pitches, rolls


def _parse_rpy(text):
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid row pose '{text}', expected ROLL,PITCH,YAW"
        )
    return tuple(float(part) for part in parts)


def row_poses(args, n):
    """Return per-row roll, pitch, yaw values in degrees."""
    if n <= len(DEFAULT_ROW_RPY):
        poses = list(DEFAULT_ROW_RPY[:n])
    else:
        view_seed = args.seed if args.view_seed is None else args.view_seed
        yaws, pitches, rolls = sample_view_angles(
            n,
            view_seed,
            pitch_range_deg=args.pitch_range,
            roll_range_deg=args.roll_range,
        )
        poses = list(zip(rolls, pitches, yaws))

    if not args.row_rpy:
        return poses

    if all(":" not in spec for spec in args.row_rpy):
        if len(args.row_rpy) != n:
            raise ValueError(
                f"--row_rpy expects {n} ROLL,PITCH,YAW entries or "
                "1-indexed ROW:ROLL,PITCH,YAW overrides"
            )
        return [_parse_rpy(spec) for spec in args.row_rpy]

    for spec in args.row_rpy:
        try:
            row_text, pose_text = spec.split(":", 1)
            row = int(row_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --row_rpy value '{spec}', expected "
                "ROW:ROLL,PITCH,YAW"
            ) from exc
        if row < 1 or row > n:
            raise ValueError(f"Invalid row {row} for {n} outpainting rows")
        poses[row - 1] = _parse_rpy(pose_text)
    return poses


def perspective_footprint_mask(height, width, yaw_deg, pitch_deg=0.0,
                               roll_deg=0.0, hfov_deg=110.0,
                               vfov_deg=80.0):
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

    yaw, pitch, roll = np.deg2rad((yaw_deg, pitch_deg, roll_deg))
    forward = torch.tensor([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                            np.cos(pitch) * np.cos(yaw)], dtype=torch.float32)
    world_up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(forward, world_up, dim=0)
    if right.norm() < 1e-6:
        world_up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.linalg.cross(forward, world_up, dim=0)
    right = right / right.norm()
    up = torch.linalg.cross(right, forward, dim=0)
    camera_right = right * np.cos(roll) + up * np.sin(roll)
    camera_up = -right * np.sin(roll) + up * np.cos(roll)

    df = d @ forward
    dr = d @ camera_right
    du = d @ camera_up
    tan_h = np.tan(np.deg2rad(hfov_deg) / 2)
    tan_v = np.tan(np.deg2rad(vfov_deg) / 2)
    mask = (df > 0) & (dr.abs() <= df * tan_h) & (du.abs() <= df * tan_v)
    return mask.float()  # [H, W]


def _resize_image(image, height, mode):
    if height <= 0 or image.shape[0] == height:
        return image
    if image.dim() == 2:
        resized = torch.nn.functional.interpolate(
            image[None, None].float(),
            size=(height, height * 2),
            mode=mode,
        )[0, 0]
    else:
        kwargs = {"align_corners": False} if mode != "nearest" else {}
        resized = torch.nn.functional.interpolate(
            image.permute(2, 0, 1)[None].float(),
            size=(height, height * 2),
            mode=mode,
            **kwargs,
        )[0].permute(1, 2, 0)
    return resized


def fit_latent(model, gt_norm, mask, device, steps=600, lr=1e-2,
               kld_weight=1e-4, fit_height=0, rays_per_step=0,
               decode_chunk=65536):
    """Fit a latent on the masked (visible) pixels with the frozen decoder.

    gt_norm: [H, W, 3] in the model's normalised (log) domain.
    mask:    [H, W] 1 = visible.
    """
    output_height = gt_norm.shape[0]
    gt_fit = _resize_image(gt_norm, fit_height, mode="bilinear")
    mask_fit = _resize_image(mask, fit_height, mode="nearest")
    H, W = gt_fit.shape[:2]
    ray_bundle = equirect_ray_bundle(device, idx=0, height=H)

    z = torch.zeros(1, model.field.latent_dim, 3, device=device,
                    requires_grad=True)
    target = gt_fit.reshape(-1, gt_fit.shape[-1]).to(device)
    visible = torch.nonzero(mask_fit.reshape(-1).bool(), as_tuple=False)
    visible = visible.squeeze(1).to(device)
    if visible.numel() == 0:
        raise ValueError("Perspective footprint mask contains no visible rays")
    optimiser = torch.optim.Adam([z], lr=lr)

    for step in range(steps):
        optimiser.zero_grad()
        if rays_per_step and visible.numel() > rays_per_step:
            draw = torch.randint(visible.numel(), (rays_per_step,),
                                 device=device)
            fit_indices = visible[draw]
        else:
            fit_indices = visible
        fit_samples = model.create_ray_samples(
            ray_bundle.origins[fit_indices],
            ray_bundle.directions[fit_indices],
            ray_bundle.camera_indices[fit_indices],
        )
        latents = z.repeat(fit_samples.shape[0], 1, 1)
        out = model.field.forward(fit_samples, rotation=None,
                                  latent_codes=latents)[RENIFieldHeadNames.RGB]
        mse = torch.nn.functional.mse_loss(out, target[fit_indices])
        cosine = 1 - torch.nn.functional.cosine_similarity(
            out, target[fit_indices], dim=-1).mean()
        kld = z.pow(2).mean()
        loss = 10.0 * mse + cosine + kld_weight * kld
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        output_bundle = equirect_ray_bundle(device, idx=0,
                                            height=output_height)
        chunks = []
        for start in range(0, len(output_bundle), decode_chunk):
            end = start + decode_chunk
            sample_chunk = model.create_ray_samples(
                output_bundle.origins[start:end],
                output_bundle.directions[start:end],
                output_bundle.camera_indices[start:end],
            )
            latents = z.repeat(sample_chunk.shape[0], 1, 1)
            out_chunk = model.field.forward(
                sample_chunk,
                rotation=None,
                latent_codes=latents,
            )[RENIFieldHeadNames.RGB]
            chunks.append(out_chunk)
        out = torch.cat(chunks, dim=0)
        if getattr(model, "two_bracket", False):
            pred = model._to_linear_hdr(out).reshape(output_height,
                                                     output_height * 2, 3)
        else:
            pred = model.field.unnormalise(out).reshape(output_height,
                                                        output_height * 2, 3)
    return linear_to_sRGB(pred, use_quantile=True).cpu().detach(), \
        float(mse.detach())


def perspective_figure(args):
    device = args.device
    _, datamanager, model = load_model(MODEL_DIRS[args.model][100],
                                       device=device)
    n = len(args.image_indices)
    poses = row_poses(args, n)
    print(f"[views] hfov={args.hfov:.0f} "
          f"vfov={args.vfov:.0f} fit_height={args.fit_height or 'native'} "
          f"fit_rays_per_step={args.fit_rays_per_step or 'all-visible'}")
    for row, (roll, pitch, yaw) in enumerate(poses, start=1):
        print(f"[views] row={row} roll={roll:+.1f} pitch={pitch:+.1f} "
              f"yaw={yaw:+.1f}")

    fig, axes = plt.subplots(n, 4, figsize=(12, 1.7 * n))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    for i, idx in enumerate(args.image_indices):
        roll, pitch, yaw = poses[i]
        full = render_eval_image(model, datamanager, idx, device)
        gt_img = full["gt_img"]
        H, W = gt_img.shape[:2]
        mask = perspective_footprint_mask(H, W, yaw, pitch, roll,
                                          args.hfov, args.vfov)

        batch = datamanager.eval_dataset[idx]
        gt_norm = eval_image_tensor(datamanager.eval_dataset, idx, batch=batch)
        if getattr(model, "two_bracket", False) and gt_norm.shape[-1] != 6:
            # bracket-form GT may arrive as [H, W, 3, 2]; flatten to [H, W, 6]
            gt_norm = gt_norm.reshape(H, W, 6)
        completion, fit_mse = fit_latent(model, gt_norm, mask, device,
                                         steps=args.fit_steps,
                                         fit_height=args.fit_height,
                                         rays_per_step=args.fit_rays_per_step,
                                         decode_chunk=args.decode_chunk)
        print(f"[fit] idx={idx} roll={roll:+.0f} pitch={pitch:+.0f} "
              f"yaw={yaw:+.0f} masked-mse={fit_mse:.4f}")

        masked_gt = gt_img * mask.unsqueeze(-1)
        for col, img in enumerate(
                (gt_img, full["pred_img"], masked_gt, completion)):
            axes[i, col].imshow(
                img.numpy() if torch.is_tensor(img) else img,
                interpolation="lanczos",
            )
            axes[i, col].axis("off")
            axes[i, col].set_aspect(1)

    plt.tight_layout()
    if args.labels:
        _add_column_labels(fig, axes, args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg, dpi=args.dpi)


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
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    for i, idx in enumerate(args.image_indices):
        for col, (key, img_key) in enumerate(
            (("full", "gt_img"), ("full", "pred_img"),
             ("masked", "gt_img"), ("masked", "pred_img"))):
            axes[i, col].imshow(outputs[key][idx][img_key])
            axes[i, col].axis("off")
            axes[i, col].set_aspect(1)
    plt.tight_layout()
    if args.labels:
        _add_column_labels(fig, axes, args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "outpainting")
    parser.add_argument("--image_indices", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--model", default="two_bracket_w3_2cyc",
                        help="MODEL_DIRS key for the perspective figure "
                             "(default: thesis two-bracket 2-cycle, the "
                             "completion-optimal model; reni_pp = paper)")
    parser.add_argument("--mask_mode", choices=["perspective", "dataset"],
                        default="perspective")
    parser.add_argument("--hfov", type=float, default=110.0,
                        help="Horizontal FoV of the footprint (deg)")
    parser.add_argument("--vfov", type=float, default=80.0,
                        help="Vertical FoV of the footprint (deg)")
    parser.add_argument("--fit_steps", type=int, default=600)
    parser.add_argument("--fit_height", type=int, default=0,
                        help="Height used for latent fitting; 0 = output/native")
    parser.add_argument("--fit_rays_per_step", type=int, default=32768,
                        help="Visible rays sampled per fit step; 0 = all visible")
    parser.add_argument("--view_seed", type=int, default=None,
                        help="Seed for randomized perspective viewpoints")
    parser.add_argument("--pitch_range", type=float, default=25.0,
                        help="Sample pitch from +/- this many degrees")
    parser.add_argument("--roll_range", type=float, default=35.0,
                        help="Sample roll from +/- this many degrees")
    parser.add_argument("--row_rpy", nargs="*", default=[],
                        metavar="ROW:ROLL,PITCH,YAW",
                        help="Per-row roll,pitch,yaw overrides in degrees")
    parser.add_argument("--labels", action="store_true",
                        help="Bake current LaTeX/TikZ labels into the figure")
    parser.add_argument("--label_fontsize", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=400,
                        help="Raster output DPI; 400 preserves 1024-wide panels")
    args = parser.parse_args()
    seed_all(args.seed)

    if args.mask_mode == "perspective":
        perspective_figure(args)
    else:
        dataset_figure(args)


if __name__ == "__main__":
    main()
