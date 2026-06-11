"""Inverse rendering figure (paper Fig: inverse_rendering).

Bunny and teapot renders under unknown illumination across six specular
levels: ground truth alongside SH (9th order) and RENI++ (D=100) inverse
fits, with the recovered environment maps. Uses the fitted inverse-task
checkpoints from the paper archive; the collage layout is ported verbatim
from publication/inverse_task.ipynb.

    PYTHONPATH=. python scripts/figures/fig_inverse_rendering.py
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from _common import PAPER_MODELS, add_common_args, save_figure, seed_all

from reni.configs.reni_inverse_config import RENIInverse  # noqa: E402
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig  # noqa: E402
from reni.illumination_fields.sh_illumination_field import (  # noqa: E402
    SphericalHarmonicIlluminationFieldConfig)
from reni.illumination_fields.sg_illumination_field import (  # noqa: E402
    SphericalGaussianFieldConfig)

# (object, env_idx) combinations shown in the paper figure
OUTPUT_CONFIG = {
    0: {"object": "bunny", "env_idx": 5},
    1: {"object": "teapot", "env_idx": 1},
    2: {"object": "bunny", "env_idx": 2},
    3: {"object": "teapot", "env_idx": 3},
}
SPECULARS = ["0.000000", "0.200000", "0.400000", "0.600000", "0.800000",
             "1.000000"]


def _remap_field_ckpt(path_components):
    """Saved configs hold container paths like /workspace/outputs/reni/<fam>/
    <variant>; remap onto the paper archive."""
    parts = list(path_components)
    if "reni" in parts:
        rest = parts[parts.index("reni") + 1:]
        candidate = PAPER_MODELS.joinpath(*rest)
        if candidate.exists():
            return candidate
    return Path(os.path.join(*parts))


def load_inverse_model(load_dir: Path, device: str):
    load_dir = Path(load_dir)
    ckpt_dir = load_dir / "nerfstudio_models"
    load_step = sorted(int(x[x.find("-") + 1: x.find(".")])
                       for x in os.listdir(ckpt_dir))[-1]
    ckpt = torch.load(ckpt_dir / f"step-{load_step:09d}.ckpt",
                      map_location=device)
    model_dict = {k[7:]: v for k, v in ckpt["pipeline"].items()
                  if k.startswith("_model.")}

    with open(load_dir / "config.yml") as f:
        config = yaml.safe_load(re.sub(r"!!python[^\s]*", "", f.read()))

    model_config = RENIInverse.config
    saved_dp = config["pipeline"]["datamanager"]["dataparser"]
    dp = model_config.pipeline.datamanager.dataparser
    dp.shininess = saved_dp["shininess"]
    dp.subset_index = saved_dp["subset_index"]
    dp.envmap_remove_indicies = saved_dp["envmap_remove_indicies"]

    saved_model = config["pipeline"]["model"]
    model_config.pipeline.model.illumination_field_ckpt_path = \
        _remap_field_ckpt(saved_model["illumination_field_ckpt_path"])
    fld = saved_model["illumination_field"]
    if "latent_dim" in fld:
        keys = ("conditioning", "invariant_function", "equivariance",
                "axis_of_invariance", "positional_encoding", "encoded_input",
                "latent_dim", "hidden_features", "hidden_layers",
                "mapping_layers", "mapping_features", "num_attention_heads",
                "num_attention_layers", "output_activation",
                "last_layer_linear", "trainable_scale", "old_implementation")
        model_config.pipeline.model.illumination_field = RENIFieldConfig(
            **{k: fld[k] for k in keys})
    elif "spherical_harmonic_order" in fld:
        model_config.pipeline.model.illumination_field = \
            SphericalHarmonicIlluminationFieldConfig(
                spherical_harmonic_order=fld["spherical_harmonic_order"])
    elif "row_col_gaussian_dims" in fld:
        model_config.pipeline.model.illumination_field = \
            SphericalGaussianFieldConfig(
                row_col_gaussian_dims=fld["row_col_gaussian_dims"])

    # 252 fitted latents = 2 objects x 21 TEST envmaps x 6 speculars; the
    # notebook's test_mode='val' predates the current val/test folder split.
    pipeline = model_config.pipeline.setup(
        device=device, test_mode="test", world_size=1, local_rank=0,
        grad_scaler=None)
    model = pipeline.model
    model.to(device)
    model.load_state_dict(model_dict)
    model.eval()
    return pipeline, pipeline.datamanager, model


def generate_images(model_paths, device):
    all_outputs = {}
    for model_path in model_paths:
        name = str(model_path).rstrip("/").split("/")[-1]
        print(f"[load] {name}")
        pipeline, dm, model = load_inverse_model(model_path, device)
        meta = dm.train_dataset.metadata["render_metadata"]

        indices = {k: [] for k in OUTPUT_CONFIG}
        for k, v in OUTPUT_CONFIG.items():
            for idx, m in enumerate(meta):
                if (m["normal_map_path"].name.startswith(v["object"])
                        and m["environment_map_idx"] == v["env_idx"]):
                    indices[k].append(idx)
            indices[k].sort(key=lambda i: meta[i]["specular_term"])

        outputs = {}
        for i, idx_list in enumerate(indices.values()):
            for s, idx in enumerate(idx_list):
                first, batch = dm.eval_image_at_idx(idx)
                from nerfstudio.cameras.cameras import Cameras
                if isinstance(first, Cameras):
                    # newer dataloaders yield Cameras; model wants RayBundle
                    ray_bundle = first.generate_rays(0)
                    ray_bundle.camera_indices = torch.ones_like(
                        ray_bundle.camera_indices) * idx
                else:
                    ray_bundle = first
                with torch.no_grad():
                    out = pipeline.model.get_outputs_for_camera_ray_bundle(
                        ray_bundle, batch)
                _, images = pipeline.model.get_image_metrics_and_images(
                    out, batch)
                rgb = images["img"]
                env = images["reni_envmap"]
                W = rgb.shape[1]
                gt_rgb, pred_rgb = rgb[:, :W // 2, :], rgb[:, W // 2:, :]
                We = env.shape[1]
                gt_env = env[:, :We // 3, :]
                pred_env = env[:, We // 3:2 * We // 3, :]
                mask = batch["mask"]
                gt_rgb[~mask.repeat(1, 1, 3)] = 1.0
                pred_rgb[~mask.repeat(1, 1, 3)] = 1.0
                outputs[f"{i}_{0.2 * s:1f}"] = {
                    "gt_rgb": gt_rgb.cpu(), "predicted_rgb": pred_rgb.cpu(),
                    "gt_envmap": gt_env.cpu(), "pred_envmap": pred_env.cpu()}
        all_outputs[name] = outputs
        del pipeline, dm, model
        torch.cuda.empty_cache()
    return all_outputs


def compose(out, sh_key, reni_key):
    """Collage layout ported verbatim from inverse_task.ipynb."""
    plt.rc("font", family="serif")
    dpi = 1
    fig, ax = plt.subplots(figsize=(4000 / dpi, 2000 / dpi), dpi=dpi)
    ax.set_facecolor("white")
    ax.set_axis_off()
    P = {"gt_envmap": 300.0, "gt_rgb": 100.0, "pred_envmap": 110.0,
         "pred_rgb": 90.0}
    fw, fh = 4000.0, 2000.0
    bx = 600

    def put(img, x, y, zoom):
        ab = AnnotationBbox(OffsetImage(img.numpy(), zoom=zoom),
                            (x / fw, y / fh), frameon=False, pad=0)
        ax.add_artist(ab)

    for i in range(4):
        put(out[sh_key][f"{i}_0.000000"]["gt_envmap"], 0.0,
            2000.0 - 300 - i * 500, P["gt_envmap"])
    for i in range(4):
        for idx, s in enumerate(SPECULARS):
            if i not in [0, 2]:
                put(out[sh_key][f"{i}_{s}"]["gt_rgb"], 700 + idx * bx,
                    2000.0 - 180 - i * 500, P["gt_rgb"])
                put(out[sh_key][f"{i}_{s}"]["predicted_rgb"],
                    550 + idx * bx, 2000.0 - 450 - i * 500, P["pred_rgb"])
                put(out[reni_key][f"{i}_{s}"]["predicted_rgb"],
                    850 + idx * bx, 2000.0 - 450 - i * 500, P["pred_rgb"])
    for i in range(4):
        for idx, s in enumerate(SPECULARS):
            if i in [0, 2]:
                put(out[sh_key][f"{i}_{s}"]["gt_rgb"], 700 + idx * bx,
                    2000.0 - 150 - i * 500, P["gt_rgb"])
            put(out[sh_key][f"{i}_{s}"]["pred_envmap"], 550 + idx * bx,
                2000.0 - 330 - i * 500, P["pred_envmap"])
            put(out[reni_key][f"{i}_{s}"]["pred_envmap"], 850 + idx * bx,
                2000.0 - 330 - i * 500, P["pred_envmap"])
            if i in [0, 2]:
                put(out[sh_key][f"{i}_{s}"]["predicted_rgb"],
                    550 + idx * bx, 2000.0 - 500 - i * 500, P["pred_rgb"])
                put(out[reni_key][f"{i}_{s}"]["predicted_rgb"],
                    850 + idx * bx, 2000.0 - 500 - i * 500, P["pred_rgb"])
    return fig, dpi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "inverse_rendering")
    args = parser.parse_args()
    seed_all(args.seed)

    model_paths = [
        PAPER_MODELS / "inverse_task" / "spherical_harmonics" / "9th_order",
        PAPER_MODELS / "inverse_task" / "reni_plus_plus" / "latent_dim_100",
    ]
    out = generate_images(model_paths, args.device)
    fig, dpi = compose(out, "9th_order", "latent_dim_100")
    save_figure(fig, args.output, svg=args.svg, dpi=dpi)


if __name__ == "__main__":
    main()
