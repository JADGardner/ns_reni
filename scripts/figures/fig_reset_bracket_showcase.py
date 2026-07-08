"""Random samples: RENI++ vs the latent-reset two-bracket model (thesis Ch2).

Two rows per model, five columns: ten prior samples z ~ N(0, I) decoded
by each model, drawn with the same seed so both models decode the same
latent codes. --random_source train_mu draws fitted training latents
instead (reconstructions rather than samples).

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
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--sample_seed", type=int, default=54)
    parser.add_argument("--random_source", choices=("normal", "train_mu"),
                        default="normal")
    parser.add_argument("--models", nargs=2,
                        default=["reni_pp", "two_bracket_w3_1cyc_testfit"])
    parser.add_argument("--labels", nargs=2,
                        default=["RENI++", "two-bracket\nlatent-reset"])
    args = parser.parse_args()

    N, C = args.num_samples, args.columns
    rows_per_model = (N + C - 1) // C
    fig, axs = plt.subplots(2 * rows_per_model, C,
                            figsize=(2.4 * C, 1.45 * 2 * rows_per_model))

    for mi, (key, label) in enumerate(zip(args.models, args.labels)):
        _, _, model = load_model(MODEL_DIRS[key][args.latent_dim],
                                 device=args.device)
        ray_bundle = equirect_ray_bundle(args.device, idx=0,
                                         height=args.height)
        ray_samples = make_ray_samples(model, ray_bundle)

        seed_all(args.sample_seed)
        if args.random_source == "train_mu":
            bank = model.field.train_mu.detach()
            indices = torch.randperm(bank.shape[0], device=bank.device)[:N]
            latents = [bank[i].unsqueeze(0) for i in indices]
        else:
            latents = [torch.randn(1, model.field.latent_dim, 3,
                                   device=args.device) for _ in range(N)]

        for i, z in enumerate(latents):
            r = mi * rows_per_model + i // C
            c = i % C
            img = decode_latents(model, ray_samples, z, height=args.height,
                                 chunk_size=args.decode_chunk)
            axs[r, c].imshow(img.numpy())
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[mi * rows_per_model, 0].set_ylabel(label, fontsize=11)

        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
