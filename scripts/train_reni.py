#!/usr/bin/env python3
"""Launch a RENI training run without going through the ns-train CLI.

Mirrors neusky's train_nerfosr.py: imports the method config directly and
hands it to nerfstudio's train entry point. This sidesteps environments where
ns-train's CLI layer fails (e.g. the csgpu13 SIF's typeguard/tyro conflict)
and works with PYTHONPATH-only setups where entry points are not registered.

Usage (inside the research container, cwd = code/ns_reni):
    PYTHONPATH=. python scripts/train_reni.py \
        --data /workspace/data/RENI_HDR --latent-dim 49 \
        --training-paradigm latent_reset --latent-reset-cycles 4
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/RENI_HDR"))
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--training-paradigm", default="standard",
                        choices=["standard", "latent_reset"])
    parser.add_argument("--latent-reset-cycles", type=int, default=1)
    parser.add_argument("--max-num-iterations", type=int, default=50001,
                        help="Iterations per cycle under latent_reset.")
    parser.add_argument("--experiment-name", default=None,
                        help="Default: reni_<paradigm>_d<latent_dim>")
    parser.add_argument("--timestamp", default=None,
                        help="Override the run timestamp directory name.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--vis", default="wandb")
    parser.add_argument(
        "--quiet-local-writer",
        action="store_true",
        help="Disable Nerfstudio's continuously refreshed terminal progress table.",
    )
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "two_bracket", "luminance_control"],
                        help="G7h experiment variant. baseline = unchanged log-space "
                             "behaviour; two_bracket = fixed gauge + 6-channel sigmoid "
                             "two-bracket targets; luminance_control = baseline losses "
                             "with luminance-weighted scale-invariant loss.")
    parser.add_argument("--m-ldr", type=float, default=16.0,
                        help="Extended-Reinhard white point for the LDR bracket.")
    parser.add_argument("--m-log", type=float, default=10000.0,
                        help="Log bracket range M_log.")
    parser.add_argument("--gauge-percentile", type=float, default=0.99)
    parser.add_argument("--gauge-target", type=float, default=1.0)
    parser.add_argument("--blended-recon", default="on", choices=["on", "off"],
                        help="Include the blended-HDR reconstruction loss (two_bracket only).")
    parser.add_argument("--blended-recon-domain", default="log1p", choices=["log1p", "linear"])
    parser.add_argument("--lum-weight-power", type=float, default=1.0)
    parser.add_argument("--lum-weight-cap", type=float, default=100.0)
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Retain every 10k checkpoint instead of only the latest "
                             "(trajectory analysis: psnr_ldr vs step/cycle).")
    parser.add_argument("--ldr-bracket-weight", type=float, default=1.0,
                        help="Loss coefficient for the LDR bracket MSE (two_bracket only). "
                             ">1 shifts capacity toward the low/mid range the LDR metrics "
                             "measure, at some cost to highlight fidelity.")
    parser.add_argument("--invariant-function", default=None,
                        choices=["GramMatrix", "VN", "VNJoint", "VNCanonical", "Norms"],
                        help="Override the field invariant function; default keeps the "
                             "method config's value (VN). VNJoint predicts one shared "
                             "Vector Neuron frame from all latent channels, retaining "
                             "inter-channel structure in the conditioning at unchanged "
                             "dimensionality. VNCanonical additionally expresses the "
                             "query direction in that frame, making the directional "
                             "input constant-size (4 numbers) instead of latent_dim+2.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the nerfstudio machine seed (replicate runs).")
    parser.add_argument("--canonical-frame-ortho", action="store_true",
                        help="VNJoint/VNCanonical: Gram-Schmidt orthonormalise the "
                             "shared frame (mitigates the frame collapse observed "
                             "in the unmitigated 2026-07-15 runs).")
    parser.add_argument(
        "--equivariance",
        default=None,
        choices=["None", "SO2", "SO3"],
        help=(
            "Override the field equivariance class. This is primarily used to "
            "rerun the paper's None/SO(2)/SO(3) ablation with an otherwise "
            "identical training recipe."
        ),
    )
    parser.add_argument(
        "--ablation-recipe",
        default="none",
        choices=["none", "reni_retrained", "transformer_decoder"],
        help=(
            "Reproduce a paper architecture-ablation row. reni_retrained uses "
            "the original concatenation decoder for 50k steps; transformer_decoder "
            "adds mirror augmentation and the attention decoder while retaining the "
            "old fixed log-range reconstruction objective."
        ),
    )
    return parser.parse_args()


def apply_variant(config, args) -> None:
    """Apply the G7h variant presets on top of the default RENI config."""
    if args.variant == "baseline":
        return

    model = config.pipeline.model
    loss_inclusions = dict(model.loss_inclusions)

    if args.variant == "luminance_control":
        model.luminance_weighted_loss = True
        model.luminance_weight_power = args.lum_weight_power
        model.luminance_weight_cap = args.lum_weight_cap
        return

    # two_bracket
    dataparser = config.pipeline.datamanager.dataparser
    dataparser.convert_to_log_domain = False
    dataparser.fixed_gauge_normalisation = True
    dataparser.fixed_gauge_percentile = args.gauge_percentile
    dataparser.fixed_gauge_target = args.gauge_target
    dataparser.tonemap_targets = True
    dataparser.tonemap_m_ldr = args.m_ldr
    dataparser.tonemap_m_log = args.m_log

    model.field.out_features = 6
    model.field.output_activation = "sigmoid"
    model.blended_recon_domain = args.blended_recon_domain

    loss_inclusions["scale_inv_loss"] = False
    loss_inclusions["ldr_bracket_mse_loss"] = True
    loss_inclusions["log_bracket_mse_loss"] = True
    loss_inclusions["blended_recon_loss"] = args.blended_recon == "on"
    model.loss_inclusions = loss_inclusions
    if args.ldr_bracket_weight != 1.0:
        coefficients = dict(model.loss_coefficients)
        coefficients["ldr_bracket_mse_loss"] = args.ldr_bracket_weight
        model.loss_coefficients = coefficients


def apply_ablation_recipe(config, recipe: str) -> None:
    """Apply the preserved 2023 architecture-ablation training settings."""
    if recipe == "none":
        return

    from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
    from reni.data.reni_pixel_sampler import RENIEquirectangularPixelSamplerConfig

    current_datamanager = config.pipeline.datamanager
    config.pipeline.datamanager = RENIDataManagerConfig(
        data=current_datamanager.data,
        dataparser=current_datamanager.dataparser,
        pixel_sampler=RENIEquirectangularPixelSamplerConfig(
            full_image_per_batch=False,
            images_per_batch=1,
            is_equirectangular=True,
        ),
        images_on_gpu=True,
        masks_on_gpu=False,
        train_num_rays_per_batch=8192,
        eval_num_rays_per_batch=8192,
    )
    datamanager = config.pipeline.datamanager
    dataparser = datamanager.dataparser
    model = config.pipeline.model
    field = model.field

    dataparser.convert_to_ldr = False
    dataparser.convert_to_log_domain = True
    dataparser.min_max_normalize = (-18.0536, 11.4533)
    dataparser.augment_with_rotation = False
    dataparser.apply_eval_rotation = False
    field.conditioning = "Concat"
    field.encoded_input = "None"
    field.equivariance = "SO2"
    field.hidden_layers = 5
    field.output_activation = "None"
    field.out_features = 3

    losses = dict(model.loss_inclusions)
    losses["log_mse_loss"] = True
    losses["scale_inv_loss"] = False
    losses["ldr_bracket_mse_loss"] = False
    losses["log_bracket_mse_loss"] = False
    losses["blended_recon_loss"] = False
    model.loss_inclusions = losses

    config.optimizers["field"]["optimizer"].lr = 1e-5
    dataparser.augment_with_mirror = False

    if recipe == "transformer_decoder":
        dataparser.augment_with_mirror = True
        field.conditioning = "Attention"
        field.encoded_input = "Directions"
        field.hidden_layers = 9
        config.optimizers["field"]["optimizer"].lr = 1e-3


def main() -> None:
    args = parse_args()

    if args.ablation_recipe != "none":
        if args.variant != "baseline":
            raise ValueError("Ablation recipes cannot be combined with --variant.")
        if args.training_paradigm != "standard" or args.latent_reset_cycles != 1:
            raise ValueError("Architecture ablations use one standard 50k training run.")

    from nerfstudio.scripts.train import main as ns_train_main
    from reni.configs.reni_config import RENIField

    config = copy.deepcopy(RENIField.config)
    config.data = args.data.expanduser()
    config.pipeline.datamanager.data = config.data
    config.pipeline.model.field.latent_dim = args.latent_dim
    if args.invariant_function is not None:
        config.pipeline.model.field.invariant_function = args.invariant_function
    if args.canonical_frame_ortho:
        config.pipeline.model.field.canonical_frame_orthonormalise = True
    if args.seed is not None:
        config.machine.seed = args.seed
    if args.equivariance is not None:
        config.pipeline.model.field.equivariance = args.equivariance
    config.training_paradigm = args.training_paradigm
    config.latent_reset_cycles = args.latent_reset_cycles
    config.max_num_iterations = args.max_num_iterations
    if args.keep_checkpoints:
        config.save_only_latest_checkpoint = False
    apply_ablation_recipe(config, args.ablation_recipe)
    apply_variant(config, args)
    config.experiment_name = args.experiment_name or (
        f"reni_{args.training_paradigm}_d{args.latent_dim}"
        + (f"_{args.variant}" if args.variant != "baseline" else "")
    )
    if args.timestamp is not None:
        config.timestamp = args.timestamp
    config.output_dir = args.output_dir
    config.vis = args.vis
    if args.quiet_local_writer:
        config.logging.local_writer.enable = False

    ns_train_main(config)


if __name__ == "__main__":
    main()
