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
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from _common import (PAPER_MODELS, add_common_args, equirect_ray_bundle,
                     mapped_eval_image_path, read_clean_exr, save_figure,
                     seed_all)
from nerfstudio.cameras.rays import Frustums, RaySamples  # noqa: E402

from reni.configs.reni_inverse_config import RENIInverse  # noqa: E402
from reni.illumination_fields.reni_illumination_field import (  # noqa: E402
    RENIField, RENIFieldConfig)
from reni.illumination_fields.sh_illumination_field import (  # noqa: E402
    SphericalHarmonicIlluminationFieldConfig)
from reni.illumination_fields.sg_illumination_field import (  # noqa: E402
    SphericalGaussianFieldConfig)
from reni.field_components.field_heads import RENIFieldHeadNames  # noqa: E402
from reni.utils.colourspace import linear_to_sRGB  # noqa: E402

# (object, env_idx) combinations shown in the paper figure
OUTPUT_CONFIG = {
    0: {"object": "bunny", "env_idx": 5},
    1: {"object": "teapot", "env_idx": 1},
    2: {"object": "bunny", "env_idx": 2},
    3: {"object": "teapot", "env_idx": 3},
}
SPECULARS = ["0.000000", "0.200000", "0.400000", "0.600000", "0.800000",
             "1.000000"]

INVERSE_PSNR = {
    0: (("23.10 dB", "32.03 dB"), ("24.69 dB", "35.48 dB"),
        ("22.29 dB", "33.35 dB"), ("21.77 dB", "27.88 dB")),
    1: (("22.54 dB", "30.80 dB"), ("23.90 dB", "32.60 dB"),
        ("21.43 dB", "31.24 dB"), ("20.36 dB", "27.04 dB")),
    2: (("21.31 dB", "29.59 dB"), ("22.65 dB", "29.73 dB"),
        ("20.71 dB", "29.15 dB"), ("19.75 dB", "26.09 dB")),
    3: (("20.26 dB", "27.97 dB"), ("21.28 dB", "29.40 dB"),
        ("19.47 dB", "27.50 dB"), ("19.00 dB", "25.00 dB")),
    4: (("19.84 dB", "26.52 dB"), ("20.41 dB", "27.02 dB"),
        ("19.28 dB", "25.99 dB"), ("18.18 dB", "23.55 dB")),
    5: (("19.90 dB", "24.53 dB"), ("19.61 dB", "23.98 dB"),
        ("19.50 dB", "24.00 dB"), ("17.04 dB", "21.54 dB")),
}


def _tikz_to_collage_xy(x, y):
    """Map the old TikZ overlay coordinates to the 4000x2000 collage axes."""
    return (174.93 * x + 1828.28) / 4000.0, (161.89 * y + 954.98) / 2000.0


def _add_inverse_text(ax, x, y, text, fontsize, rotation=0):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontfamily="serif",
        rotation=rotation,
        color="black",
        zorder=30,
        clip_on=False,
    )


def _add_inverse_rendering_labels(ax):
    title_fs = 1200
    method_fs = 900
    row_fs = 1150
    tiny_fs = 650

    gt_title_x = _tikz_to_collage_xy(-10.4, 0)[0]
    _add_inverse_text(ax, gt_title_x, 0.970, "Ground Truth", title_fs)
    _add_inverse_text(ax, gt_title_x, 0.948, "Environment Map", title_fs)

    row_label_x = _tikz_to_collage_xy(-8.2, 0)[0]
    row_centres = (
        {"target": 1850, "recon": 1670, "fit": 1500},
        {"target": 1320, "recon": 1170, "fit": 1050},
        {"target": 850, "recon": 670, "fit": 500},
        {"target": 320, "recon": 170, "fit": 50},
    )
    for centres in row_centres:
        _add_inverse_text(ax, row_label_x, centres["target"] / 2000,
                          "Target", row_fs, rotation=90)
        _add_inverse_text(ax, row_label_x, centres["recon"] / 2000,
                          "Recon", row_fs, rotation=90)
        _add_inverse_text(ax, row_label_x, centres["fit"] / 2000,
                          "Fit", row_fs, rotation=90)

    _add_inverse_text(ax, 0.5, 1.040, "Increasing Specularity", title_fs)
    ax.annotate(
        "",
        xy=(0.86, 1.015),
        xytext=(0.22, 1.015),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=120),
        zorder=30,
        annotation_clip=False,
    )
    for x, label in zip((-6.45, -3.05, 0.41, 3.83, 7.3, 10.7),
                        ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")):
        _add_inverse_text(ax, _tikz_to_collage_xy(x, 0)[0], 0.995, label,
                          method_fs)

    method_xs = (
        (-7.3, -5.55),
        (-3.9, -2.10),
        (-0.44, 1.31),
        (2.98, 4.73),
        (6.40, 8.2),
        (9.80, 11.6),
    )
    method_y = 0.875
    psnr_offsets = (92.5, 70, 92.5, 70)
    psnr_y = tuple((centres["fit"] - offset) / 2000
                   for centres, offset in zip(row_centres, psnr_offsets))
    for spec_idx, (sh_x, reni_x) in enumerate(method_xs):
        _add_inverse_text(ax, _tikz_to_collage_xy(sh_x, 0)[0], method_y,
                          "SH", method_fs)
        _add_inverse_text(ax, _tikz_to_collage_xy(reni_x, 0)[0], method_y,
                          "RENI++", method_fs)
        for row_idx, y in enumerate(psnr_y):
            sh_value, reni_value = INVERSE_PSNR[spec_idx][row_idx]
            _add_inverse_text(ax, _tikz_to_collage_xy(sh_x, 0)[0], y,
                              sh_value, tiny_fs)
            _add_inverse_text(ax, _tikz_to_collage_xy(reni_x, 0)[0], y,
                              reni_value, tiny_fs)


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


@contextmanager
def _torch_load_default_map_location(device: str):
    """Make nested checkpoint loads respect the figure script's target device."""
    original_load = torch.load

    def load_with_map_location(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_map_location
    try:
        yield
    finally:
        torch.load = original_load


def load_inverse_model(load_dir: Path, device: str,
                       normal_resolution: int | None = None):
    load_dir = Path(load_dir)
    ckpt_dir = load_dir / "nerfstudio_models"
    load_step = sorted(int(x[x.find("-") + 1: x.find(".")])
                       for x in os.listdir(ckpt_dir))[-1]
    ckpt = torch.load(
        ckpt_dir / f"step-{load_step:09d}.ckpt",
        map_location=device,
        weights_only=False,
    )
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
    if normal_resolution is not None:
        dp.normal_map_resolution = normal_resolution

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
    with _torch_load_default_map_location(device):
        pipeline = model_config.pipeline.setup(
            device=device, test_mode="test", world_size=1, local_rank=0,
            grad_scaler=None)
    model = pipeline.model
    model.to(device)
    model.load_state_dict(model_dict)
    model.eval()
    return pipeline, pipeline.datamanager, model


def _resize_chw(image, size, mode="bilinear"):
    image = image.permute(2, 0, 1)[None].float()
    kwargs = {"size": size, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(image, **kwargs)[0].permute(1, 2, 0)


def _align_inverse_batch_resolution(batch):
    """Keep batch tensors aligned when the figure requests larger normals."""
    target_hw = batch["normal"].shape[:2]
    if batch["image"].shape[:2] != target_hw:
        batch["image"] = _resize_chw(batch["image"], target_hw)
    if batch["mask"].shape[:2] != target_hw:
        mask = batch["mask"].float()
        batch["mask"] = _resize_chw(mask, target_hw, mode="nearest").bool()
    return batch


def _inverse_gt_envmap(dataset, image_idx: int, width: int):
    render_meta = dataset.metadata["render_metadata"][image_idx]
    env_idx = render_meta["environment_map_idx"]
    low_path = dataset.metadata["environment_maps_filenames"][env_idx]
    env_path = mapped_eval_image_path("test", low_path.name) or low_path
    if not env_path.exists():
        raise FileNotFoundError(f"Inverse GT envmap missing: {env_path}")

    envmap = torch.from_numpy(read_clean_exr(env_path)).float()
    height = width // 2
    if envmap.shape[:2] != (height, width):
        envmap = _resize_chw(envmap, (height, width))
    return linear_to_sRGB(envmap, use_quantile=True)


def _render_inverse_envmap(model, image_idx: int, width: int, device: str,
                           chunk_size: int):
    height = width // 2
    bundle = equirect_ray_bundle(device, idx=image_idx, height=height)
    n_rays = bundle.directions.shape[0]
    ray_samples = RaySamples(
        frustums=Frustums(
            origins=bundle.origins,
            directions=bundle.directions,
            starts=torch.zeros((n_rays, 1), device=device),
            ends=torch.ones((n_rays, 1), device=device),
            pixel_area=torch.ones((n_rays, 1), device=device),
        ),
        camera_indices=bundle.camera_indices.reshape(n_rays, 1),
    )
    if chunk_size <= 0:
        chunk_size = ray_samples.shape[0]

    chunks = []
    with torch.no_grad():
        for start in range(0, ray_samples.shape[0], chunk_size):
            end = min(start + chunk_size, ray_samples.shape[0])
            chunk = ray_samples[start:end]
            indices = chunk.camera_indices.squeeze().long()
            latents = model.illumination_latents[indices]
            if isinstance(model.illumination_field, RENIField):
                scale = model.scale[indices] if model.scale is not None else None
                outputs = model.illumination_field.forward(
                    ray_samples=chunk,
                    latent_codes=latents,
                    scale=scale,
                )
            else:
                outputs = model.illumination_field.forward(
                    ray_samples=chunk,
                    latent_codes=latents,
                )
            hdr = model.illumination_field.unnormalise(
                outputs[RENIFieldHeadNames.RGB],
            )
            chunks.append(hdr.cpu())

    hdr_envmap = torch.cat(chunks, dim=0).reshape(height, width, 3)
    return linear_to_sRGB(hdr_envmap, use_quantile=True)


def generate_images(model_paths, device, normal_resolution=None,
                    envmap_width=128, envmap_chunk=65536):
    all_outputs = {}
    for model_path in model_paths:
        name = str(model_path).rstrip("/").split("/")[-1]
        print(f"[load] {name}")
        pipeline, dm, model = load_inverse_model(
            model_path,
            device,
            normal_resolution=normal_resolution,
        )
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
                batch = _align_inverse_batch_resolution(batch)
                with torch.no_grad():
                    out = pipeline.model.get_outputs_for_camera_ray_bundle(
                        ray_bundle, batch)
                _, images = pipeline.model.get_image_metrics_and_images(
                    out, batch)
                rgb = images["img"]
                W = rgb.shape[1]
                gt_rgb, pred_rgb = rgb[:, :W // 2, :], rgb[:, W // 2:, :]
                gt_env = _inverse_gt_envmap(dm.train_dataset, idx, envmap_width)
                pred_env = _render_inverse_envmap(
                    pipeline.model,
                    idx,
                    envmap_width,
                    device,
                    envmap_chunk,
                )
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


def compose(out, sh_key, reni_key, add_labels=False):
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

    def put(img, x, y, zoom, reference_height):
        display_zoom = zoom * reference_height / img.shape[0]
        ab = AnnotationBbox(OffsetImage(img.numpy(), zoom=display_zoom),
                            (x / fw, y / fh), frameon=False, pad=0)
        ax.add_artist(ab)

    for i in range(4):
        put(out[sh_key][f"{i}_0.000000"]["gt_envmap"], 0.0,
            2000.0 - 300 - i * 500, P["gt_envmap"], 64)
    for i in range(4):
        for idx, s in enumerate(SPECULARS):
            if i not in [0, 2]:
                put(out[sh_key][f"{i}_{s}"]["gt_rgb"], 700 + idx * bx,
                    2000.0 - 180 - i * 500, P["gt_rgb"], 128)
                put(out[sh_key][f"{i}_{s}"]["predicted_rgb"],
                    550 + idx * bx, 2000.0 - 450 - i * 500, P["pred_rgb"],
                    128)
                put(out[reni_key][f"{i}_{s}"]["predicted_rgb"],
                    850 + idx * bx, 2000.0 - 450 - i * 500, P["pred_rgb"],
                    128)
    for i in range(4):
        for idx, s in enumerate(SPECULARS):
            if i in [0, 2]:
                put(out[sh_key][f"{i}_{s}"]["gt_rgb"], 700 + idx * bx,
                    2000.0 - 150 - i * 500, P["gt_rgb"], 128)
            put(out[sh_key][f"{i}_{s}"]["pred_envmap"], 550 + idx * bx,
                2000.0 - 330 - i * 500, P["pred_envmap"], 64)
            put(out[reni_key][f"{i}_{s}"]["pred_envmap"], 850 + idx * bx,
                2000.0 - 330 - i * 500, P["pred_envmap"], 64)
            if i in [0, 2]:
                put(out[sh_key][f"{i}_{s}"]["predicted_rgb"],
                    550 + idx * bx, 2000.0 - 500 - i * 500, P["pred_rgb"],
                    128)
                put(out[reni_key][f"{i}_{s}"]["predicted_rgb"],
                    850 + idx * bx, 2000.0 - 500 - i * 500, P["pred_rgb"],
                    128)
    if add_labels:
        _add_inverse_rendering_labels(ax)
    return fig, dpi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "inverse_rendering")
    parser.add_argument("--labels", action="store_true",
                        help="Bake current LaTeX/TikZ labels into the figure")
    parser.add_argument("--normal_resolution", type=int, default=None,
                        help="Optional bunny/teapot render resolution; "
                             "defaults to the checkpoint/data parser value")
    parser.add_argument("--envmap_width", type=int, default=128,
                        help="Displayed inverse-task envmap width")
    parser.add_argument("--envmap_chunk", type=int, default=65536,
                        help="Rays per chunk for inverse envmap visualization")
    args = parser.parse_args()
    seed_all(args.seed)

    model_paths = [
        PAPER_MODELS / "inverse_task" / "spherical_harmonics" / "9th_order",
        PAPER_MODELS / "inverse_task" / "reni_plus_plus" / "latent_dim_100",
    ]
    out = generate_images(
        model_paths,
        args.device,
        normal_resolution=args.normal_resolution,
        envmap_width=args.envmap_width,
        envmap_chunk=args.envmap_chunk,
    )
    fig, dpi = compose(out, "9th_order", "latent_dim_100",
                       add_labels=args.labels)
    save_figure(fig, args.output, svg=args.svg, dpi=dpi)


if __name__ == "__main__":
    main()
