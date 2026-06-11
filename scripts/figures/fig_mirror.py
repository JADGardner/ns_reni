"""Latent axis-negation figure (paper Fig: mirror).

Decodes a fitted train latent, then the same latent with each coordinate
axis negated, demonstrating the O(3) structure of the latent space.

    PYTHONPATH=. python scripts/figures/fig_mirror.py
"""

import argparse

import matplotlib.pyplot as plt
import torch

from _common import (MODEL_DIRS, add_common_args, decode_latents,
                     equirect_ray_bundle, load_model, make_ray_samples,
                     save_figure, seed_all)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "mirror")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--image_idx", type=int, default=96,
                        help="Train-set latent to decode (paper used 96)")
    args = parser.parse_args()
    seed_all(args.seed)

    _, _, model = load_model(MODEL_DIRS["reni_pp"][args.latent_dim],
                             device=args.device)
    ray_bundle = equirect_ray_bundle(args.device, idx=0, height=args.height)
    ray_samples = make_ray_samples(model, ray_bundle)

    base = model.field.train_mu[args.image_idx]
    images = {"normal": decode_latents(model, ray_samples, base.unsqueeze(0),
                                       height=args.height)}
    for name, axis in (("x_negated", 0), ("y_negated", 1), ("z_negated", 2)):
        negate = torch.eye(3, dtype=torch.float32).type_as(base)
        negate[axis, axis] = -1
        z = torch.matmul(base, negate).unsqueeze(0)
        images[name] = decode_latents(model, ray_samples, z, height=args.height)

    fig, axs = plt.subplots(2, 3, figsize=(12, 5))
    axs[0, 0].axis("off"); axs[0, 2].axis("off")
    layout = {(0, 1): "normal", (1, 0): "x_negated",
              (1, 1): "z_negated", (1, 2): "y_negated"}
    for (r, c), key in layout.items():
        axs[r, c].imshow(images[key].numpy())
        axs[r, c].axis("off")

    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
