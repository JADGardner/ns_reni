"""Latent interpolation figure (paper Fig: interpolations_and_random_samples).

Rows 1-2: interpolations between random latent draws.
Row 3: interpolation between two fitted eval latents, with the pure (decoded)
endpoints shown at either end.

    PYTHONPATH=. python scripts/figures/fig_interpolations.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _common import (MODEL_DIRS, add_common_args, axis_center, decode_latents, equirect_ray_bundle,
                     load_model, make_ray_samples, save_figure, seed_all)


def _resolve_model_dir(path: Path) -> Path:
    if path.suffix == ".ckpt":
        return path.parent.parent
    if path.name == "nerfstudio_models":
        return path.parent
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "interpolations_and_random_samples")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--model_dir", type=Path, default=None,
                        help="Checkpoint run directory, nerfstudio_models directory, "
                             "or .ckpt path. Defaults to RENI++ (thesis) "
                             "for --latent_dim.")
    parser.add_argument("--steps", type=int, default=6, help="Interpolation steps")
    parser.add_argument("--idx1", type=int, default=6, help="Eval latent endpoint 1")
    parser.add_argument("--idx2", type=int, default=12, help="Eval latent endpoint 2")
    parser.add_argument("--interp_seed", type=int, default=54,
                        help="Seed for the random-sample rows (paper used 54)")
    parser.add_argument("--random_source", choices=("normal", "train_mu"),
                        default="train_mu",
                        help="Source for the two random interpolation rows. "
                             "'normal' preserves the paper figure; 'train_mu' "
                             "samples fitted training latents, which is better "
                             "for latent-reset checkpoints.")
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model_dir) if args.model_dir else \
        MODEL_DIRS["vnjoint_ortho_2cyc_testfit"][args.latent_dim]
    _, datamanager, model = load_model(model_dir, device=args.device)
    ray_bundle = equirect_ray_bundle(args.device, idx=0, height=args.height)
    ray_samples = make_ray_samples(model, ray_bundle)

    seed_all(args.interp_seed)
    K = args.steps
    fig, axs = plt.subplots(3, K + 2, figsize=(20, 4))

    # Rows 1-2: random-draw interpolations (endpoints hidden, as in the paper)
    train_indices = None
    if args.random_source == "train_mu":
        train_latents = model.field.train_mu.detach()
        if train_latents.shape[0] < 4:
            raise ValueError("--random_source train_mu needs at least 4 train latents")
        train_indices = torch.randperm(train_latents.shape[0],
                                       device=train_latents.device)[:4]

    for row in range(2):
        axs[row, 0].axis("off")
        axs[row, -1].axis("off")
        if args.random_source == "train_mu":
            z1 = train_latents[train_indices[2 * row]].unsqueeze(0)
            z2 = train_latents[train_indices[2 * row + 1]].unsqueeze(0)
        else:
            z1 = torch.randn(1, model.field.latent_dim, 3, device=args.device)
            z2 = torch.randn(1, model.field.latent_dim, 3, device=args.device)
        for col in range(1, K + 1):
            t = (col - 1) / (K - 1)
            img = decode_latents(model, ray_samples, torch.lerp(z1, z2, t),
                                 height=args.height,
                                 chunk_size=args.decode_chunk)
            axs[row, col].imshow(img.numpy())
            axs[row, col].axis("off")

    # Row 3: fitted eval latents with pure endpoints
    z1 = model.field.eval_mu[args.idx1].unsqueeze(0)
    z2 = model.field.eval_mu[args.idx2].unsqueeze(0)
    for col, z in zip([0, -1], [z1, z2]):
        img = decode_latents(model, ray_samples, z, height=args.height,
                             chunk_size=args.decode_chunk)
        axs[2, col].imshow(img.numpy())
        axs[2, col].axis("off")
    for col in range(1, K + 1):
        t = (col - 1) / (K - 1)
        img = decode_latents(model, ray_samples, torch.lerp(z1, z2, t),
                             height=args.height,
                             chunk_size=args.decode_chunk)
        axs[2, col].imshow(img.numpy())
        axs[2, col].axis("off")

    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
