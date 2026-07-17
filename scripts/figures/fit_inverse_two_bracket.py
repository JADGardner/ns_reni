"""Fit the RENI++ inverse-rendering task with a two-bracket decoder (thesis Ch.2).

Optimises per-image illumination latents (+ a per-image scale) with a FROZEN
two-bracket RENI++ decoder: bunny/teapot are rendered under the unknown
illumination through the BlinnPhong shader and the latents are optimised to
match the target renders. The illumination material (albedo / specular /
shininess) is supplied by the dataparser as known per-specular-level values and
is NOT optimised (this matches the paper reni-inverse protocol; only the decoder
and the two-bracket blend before shading differ).

This reuses the registered ``reni-inverse`` TrainerConfig, overriding only the
decoder (out_features=6, sigmoid) + the two-bracket flags + the frozen decoder
checkpoint, so no plugin re-registration is needed.

    PYTHONPATH=.:scripts/figures python scripts/figures/fit_inverse_two_bracket.py \
        --decoder-ckpt /workspace/phd/outputs/reni/_figshim_w3_2cyc --decoder-step 100001 \
        --output-dir /workspace/phd/outputs/reni_inverse_two_bracket --max-iters 50400
"""

import argparse
import copy
from pathlib import Path

from reni.configs.reni_inverse_config import RENIInverse

# (object, environment_map_idx) cells shown in fig_inverse_rendering's
# OUTPUT_CONFIG; --figure-subset fits only these (x6 speculars = 24 images) so
# the fit is ~12 min instead of the full 252-image (~2 h) run, while the env_idx
# references stay aligned with the paper SH column and GT.
FIGURE_CELLS = {("bunny", 5), ("teapot", 1), ("bunny", 2), ("teapot", 3)}


def _figure_subset_indices(dataparser_config, split):
    """Flat render-metadata indices for the figure's shown (object, env) cells."""
    dp = copy.deepcopy(dataparser_config)
    dp.subset_index = None
    meta = dp.setup().get_dataparser_outputs(split=split).metadata["render_metadata"]
    subset = []
    for i, m in enumerate(meta):
        obj = "bunny" if m["normal_map_path"].name.startswith("bunny") else "teapot"
        if (obj, m["environment_map_idx"]) in FIGURE_CELLS:
            subset.append(i)
    return subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-ckpt", required=True,
                        help="Dir with nerfstudio_models/ holding the two-bracket "
                             "RENI++ decoder checkpoint (e.g. the _figshim_w3_2cyc "
                             "shim; its decoder weights equal the raw 2-cycle run)")
    parser.add_argument("--decoder-step", type=int, default=100001)
    parser.add_argument("--data", default="data/RENI_HDR")
    parser.add_argument("--output-dir", default="/workspace/phd/outputs/reni_inverse_two_bracket")
    parser.add_argument("--experiment-name", default="reni_inverse_two_bracket")
    parser.add_argument("--max-iters", type=int, default=50400)
    parser.add_argument("--steps-per-save", type=int, default=2000)
    parser.add_argument("--split", default="test",
                        help="Envmap split to fit (test = the paper's 21 envmaps)")
    parser.add_argument("--figure-subset", action="store_true",
                        help="Fit only the cells fig_inverse_rendering displays "
                             "(24 images); pair with --max-iters 4800 (200/image)")
    parser.add_argument("--m-ldr", type=float, default=16.0)
    parser.add_argument("--m-log", type=float, default=10000.0)
    parser.add_argument("--invariant-function", default=None,
                        choices=["GramMatrix", "VN", "VNJoint", "VNCanonical", "Norms"],
                        help="Override the field invariant function to match "
                             "the decoder checkpoint (config default: VN).")
    parser.add_argument("--canonical-frame-ortho", action="store_true",
                        help="Gram-Schmidt frame decoders: must match training.")
    parser.add_argument("--vis", default="tensorboard")
    args = parser.parse_args()

    config = copy.deepcopy(RENIInverse.config)
    m = config.pipeline.model
    # Two-bracket decoder: six sigmoid channels blended to linear HDR.
    m.illumination_field.out_features = 6
    m.illumination_field.output_activation = "sigmoid"
    if args.invariant_function is not None:
        m.illumination_field.invariant_function = args.invariant_function
    if args.canonical_frame_ortho:
        m.illumination_field.canonical_frame_orthonormalise = True
    m.two_bracket = True
    m.tonemap_m_ldr = args.m_ldr
    m.tonemap_m_log = args.m_log
    m.illumination_field_ckpt_path = Path(args.decoder_ckpt)
    m.illumination_field_ckpt_step = args.decoder_step

    config.pipeline.datamanager.dataparser.data = Path(args.data)
    if args.figure_subset:
        subset = _figure_subset_indices(config.pipeline.datamanager.dataparser, args.split)
        config.pipeline.datamanager.dataparser.subset_index = subset
        print(f"[fit] figure subset: {len(subset)} images {subset}")
    config.max_num_iterations = args.max_iters
    config.steps_per_save = args.steps_per_save
    # Keep all periodic-eval sentinels above max so they never fire mid-fit. The
    # eval-image path (next_eval_image) asserts Cameras.camera_indices, which
    # newer nerfstudio Cameras lack; the fit only needs the final latents and the
    # figure renders through the robust eval_image_at_idx path instead.
    config.steps_per_eval_batch = args.max_iters + 1
    config.steps_per_eval_image = args.max_iters + 1
    config.steps_per_eval_all_images = args.max_iters + 1
    config.output_dir = Path(args.output_dir)
    config.experiment_name = args.experiment_name
    config.vis = args.vis
    config.set_timestamp()
    # ns-train's launcher writes config.yml via save_config(); we call setup()
    # directly, so do it here or the figure loader has no config to read back.
    config.save_config()

    print(f"[fit] two-bracket inverse: decoder={args.decoder_ckpt} "
          f"step={args.decoder_step} split={args.split} iters={args.max_iters} "
          f"out={args.output_dir}")
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup(test_mode=args.split)
    trainer.train()
    print("[fit] done")


if __name__ == "__main__":
    main()
