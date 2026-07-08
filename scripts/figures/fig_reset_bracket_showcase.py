"""Random samples and interpolations: RENI++ vs the latent-reset two-bracket
model (thesis Ch2).

Four rows: random latent draws decoded by each model (same seed, so both
models decode the same latent codes), and an interpolation between two
fitted test latents for each model (decoded endpoints at either side).

    PYTHONPATH=. python scripts/figures/fig_reset_bracket_showcase.py
"""

import argparse

import matplotlib.pyplot as plt
import torch

from _common import (MODEL_DIRS, add_common_args, decode_latents,
                     equirect_ray_bundle, load_model, make_ray_samples,
                     save_figure, seed_all)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "reset_bracket_showcase")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--idx1", type=int, default=6)
    parser.add_argument("--idx2", type=int, default=12)
    parser.add_argument("--sample_seed", type=int, default=54)
    parser.add_argument("--models", nargs=2,
                        default=["reni_pp", "two_bracket_w3_1cyc_testfit"])
    parser.add_argument("--labels", nargs=2,
                        default=["RENI++", "two-bracket\nlatent-reset"])
    args = parser.parse_args()

    K = args.steps
    fig, axs = plt.subplots(4, K + 2, figsize=(2.4 * (K + 2), 5.4))

    for mi, (key, label) in enumerate(zip(args.models, args.labels)):
        _, _, model = load_model(MODEL_DIRS[key][args.latent_dim],
                                 device=args.device)
        ray_bundle = equirect_ray_bundle(args.device, idx=0,
                                         height=args.height)
        ray_samples = make_ray_samples(model, ray_bundle)

        # random samples: same seed for both models, so column j shows the
        # same latent code under each decoder
        seed_all(args.sample_seed)
        r = 2 * mi
        axs[r, 0].axis("off")
        axs[r, -1].axis("off")
        for col in range(1, K + 1):
            z = torch.randn(1, model.field.latent_dim, 3, device=args.device)
            img = decode_latents(model, ray_samples, z, height=args.height,
                                 chunk_size=args.decode_chunk)
            axs[r, col].imshow(img.numpy())
            axs[r, col].set_xticks([]); axs[r, col].set_yticks([])
        axs[r, 1].set_ylabel(f"{label}\nsamples", fontsize=11)

        # interpolation between two fitted test latents
        r = 2 * mi + 1
        z1 = model.field.eval_mu[args.idx1].unsqueeze(0)
        z2 = model.field.eval_mu[args.idx2].unsqueeze(0)
        for col, z in ((0, z1), (K + 1, z2)):
            img = decode_latents(model, ray_samples, z, height=args.height,
                                 chunk_size=args.decode_chunk)
            axs[r, col].imshow(img.numpy())
            axs[r, col].set_xticks([]); axs[r, col].set_yticks([])
        for col in range(1, K + 1):
            t = (col - 1) / (K - 1)
            img = decode_latents(model, ray_samples,
                                 torch.lerp(z1, z2, t), height=args.height,
                                 chunk_size=args.decode_chunk)
            axs[r, col].imshow(img.numpy())
            axs[r, col].set_xticks([]); axs[r, col].set_yticks([])
        axs[r, 0].set_ylabel(f"{label}\ninterpolation", fontsize=11)

        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
