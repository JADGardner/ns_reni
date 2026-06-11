"""Shared helpers for the RENI++ figure scripts.

Headless, deterministic, inference-driven figure generation (same philosophy
as 360anything/scripts/sphere_jepa/figures): each fig_*.py loads trained
checkpoints, runs inference, and writes PNG+PDF (+optional SVG) without any
notebook or display dependency.

Run from the ns_reni repo root with the reni++ environment, e.g.:

    PYTHONPATH=. python scripts/figures/fig_comparison.py

Checkpoint layout expected under <repo>/checkpoints/ (all committed):
    reni_plus_plus_models/latent_dim_{9,36,49,100}
    reni_plus_plus_models/masked_models/latent_dim_100
    reni_original/ndims_{9,20,36,49,100}
    spherical_harmonics/{2nd,5th,6th,9th}_order
    spherical_gaussians/num_param_{30,108,150,300}
"""

import os
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# Relative paths in saved configs (data/RENI_HDR etc.) resolve from repo root.
os.chdir(REPO_ROOT)

from nerfstudio.cameras.cameras import Cameras, CameraType  # noqa: E402
from nerfstudio.utils import colormaps  # noqa: E402

from reni.configs.reni_config import RENIField  # noqa: E402
from reni.configs.sh_sg_envmap_configs import SHField, SGField  # noqa: E402
from reni.field_components.field_heads import RENIFieldHeadNames  # noqa: E402
from reni.utils.colourspace import linear_to_sRGB  # noqa: E402
from reni.utils.utils import rot_y, rot_z  # noqa: E402

CHECKPOINTS = REPO_ROOT / "checkpoints"

# Single shared resolver (no symlinks): maps legacy 'checkpoints/<rel>' names
# onto the repo extras / paper-model archive / baselines archive roots.
from reni.utils.checkpoint_locator import find_checkpoint  # noqa: E402

PAPER_MODELS = find_checkpoint("reni_plus_plus_models").parent

MODEL_DIRS = {
    "reni_pp": {d: PAPER_MODELS / "reni_plus_plus_models" / f"latent_dim_{d}"
                for d in (9, 36, 49, 100)},
    "reni_pp_masked": {100: PAPER_MODELS / "reni_plus_plus_models" / "masked_models" / "latent_dim_100"},
    "reni_old": {d: PAPER_MODELS / "old_reni_models" / f"latent_dim_{d}"
                 for d in (9, 20, 36, 49, 100)},
    "sh": {o: PAPER_MODELS / "spherical_harmonics" / f"{o}_order"
           for o in ("2nd", "5th", "6th", "9th")},
    "sg": {n: PAPER_MODELS / "spherical_gaussians" / f"num_param_{n}"
           for n in (30, 108, 150, 300)},
}


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean_yaml(content: str):
    return yaml.safe_load(re.sub(r"!!python[^\s]*", "", content))


def resolve_run_dir(path: Path) -> Path:
    """Find the nerfstudio run dir under path (handles nested
    <variant>/RENI_HDR/<method>/<timestamp>/ layouts; picks latest run)."""
    path = Path(path)
    if (path / "nerfstudio_models").exists():
        return path
    candidates = sorted(p.parent for p in path.glob("**/nerfstudio_models"))
    if not candidates:
        raise FileNotFoundError(f"No nerfstudio_models dir under {path}")
    return candidates[-1]


def _remap_workspace_path(p: Path) -> Path:
    """Saved configs may hold container-absolute /workspace/... paths."""
    p = Path(p)
    if p.exists():
        return p
    parts = p.parts
    if "/workspace" in str(p):
        idx = parts.index("workspace")
        candidate = REPO_ROOT.joinpath(*parts[idx + 1:])
        if candidate.exists():
            return candidate
    return p


def load_model(load_dir: Path, device: str = "cuda:0", load_step: Optional[int] = None):
    """Load a RENI++/RENI/SH/SG pipeline from a nerfstudio checkpoint dir.

    Returns (pipeline, datamanager, model). Ported from
    publication/figures_and_tables.ipynb so figures match the paper exactly.
    """
    load_dir = resolve_run_dir(load_dir)
    ckpt_dir = load_dir / "nerfstudio_models"
    if load_step is None:
        load_step = sorted(
            int(x[x.find("-") + 1: x.find(".")]) for x in os.listdir(ckpt_dir)
        )[-1]

    ckpt = torch.load(ckpt_dir / f"step-{load_step:09d}.ckpt", map_location=device)
    model_dict = {k[7:]: v for k, v in ckpt["pipeline"].items() if k.startswith("_model.")}

    with open(load_dir / "config.yml") as f:
        config = _clean_yaml(f.read())

    def _copy_dataparser(model_config):
        """Apply the SAVED dataparser normalisation to the rebuilt config —
        critical for LDR metrics: unnormalise() must invert exactly the
        normalisation the fit was trained with."""
        dp = model_config.pipeline.datamanager.dataparser
        saved_dp = config["pipeline"]["datamanager"]["dataparser"]
        dp.convert_to_ldr = saved_dp["convert_to_ldr"]
        dp.convert_to_log_domain = saved_dp["convert_to_log_domain"]
        dp.eval_mask_path = (
            _remap_workspace_path(Path(os.path.join(*saved_dp["eval_mask_path"])))
            if saved_dp["eval_mask_path"] is not None else None)
        mmn = saved_dp["min_max_normalize"]
        dp.min_max_normalize = tuple(mmn) if isinstance(mmn, list) else mmn
        dp.augment_with_mirror = saved_dp["augment_with_mirror"]

    field_cfg = config["pipeline"]["model"]["field"]
    if "latent_dim" in field_cfg:
        model_config = RENIField.config
        _copy_dataparser(model_config)
        m = model_config.pipeline.model
        m.loss_inclusions = config["pipeline"]["model"]["loss_inclusions"]
        for key in ("conditioning", "invariant_function", "equivariance",
                    "axis_of_invariance", "positional_encoding", "encoded_input",
                    "latent_dim", "hidden_features", "hidden_layers",
                    "mapping_layers", "mapping_features", "num_attention_heads",
                    "num_attention_layers", "output_activation",
                    "last_layer_linear", "trainable_scale", "old_implementation"):
            setattr(m.field, key, field_cfg[key])
    elif "spherical_harmonic_order" in field_cfg:
        model_config = SHField.config
        _copy_dataparser(model_config)
        model_config.pipeline.model.field.spherical_harmonic_order = \
            field_cfg["spherical_harmonic_order"]
        model_config.pipeline.model.loss_inclusions = \
            config["pipeline"]["model"]["loss_inclusions"]
        model_config.pipeline.model.loss_inclusions.setdefault(
            "scale_inv_grad_loss", False)
    elif "row_col_gaussian_dims" in field_cfg:
        model_config = SGField.config
        _copy_dataparser(model_config)
        model_config.pipeline.model.field.row_col_gaussian_dims = \
            field_cfg["row_col_gaussian_dims"]
        model_config.pipeline.model.loss_inclusions = \
            config["pipeline"]["model"]["loss_inclusions"]
        model_config.pipeline.model.loss_inclusions.setdefault(
            "scale_inv_grad_loss", False)
    else:
        raise ValueError(f"Unrecognised field config in {load_dir}")

    # Choose the eval split by matching the checkpoint's per-image eval bank
    # size against the dataset folders (test=21, val=10 for RENI_HDR). Saved
    # test_mode keys are unreliable across these heterogeneous fits.
    n_eval = None
    for k, v in model_dict.items():
        if "eval" in k.rsplit(".", 1)[-1] and v.dim() >= 2:
            n_eval = v.shape[0]
            break
    test_mode = "test"
    if n_eval is not None:
        data_root = REPO_ROOT / "data" / "RENI_HDR"
        for split in ("test", "val"):
            if len(list((data_root / split).glob("*.exr"))) == n_eval:
                test_mode = split if split != "val" else "val"
                break
    model_config.pipeline.test_mode = test_mode
    pipeline = model_config.pipeline.setup(
        device=device, test_mode=test_mode,
        world_size=1, local_rank=0, grad_scaler=None,
    )
    # The SH/SG datamanager lacks .to(); move its ray generators explicitly
    # so pipeline-level eval (make_tables) indexes on the right device.
    for attr in ("eval_ray_generator", "train_ray_generator"):
        gen = getattr(pipeline.datamanager, attr, None)
        if gen is not None:
            gen.to(device)

    model = pipeline.model
    model.to(device)

    # The committed checkpoints are a mix of full training runs and test-set
    # fits, so per-image parameter banks (RENI latents, SG/SH per-image
    # coefficients) can disagree in count with whatever splits the rebuilt
    # datamanager chose. Graft any shape-mismatched tensors directly from the
    # checkpoint at their native sizes and load everything else strictly.
    model_state = model.state_dict()
    grafted, filtered = {}, {}
    for k, v in model_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            grafted[k] = v
        else:
            filtered[k] = v
    for k, v in grafted.items():
        module_path, _, param_name = k.rpartition(".")
        mod = model.get_submodule(module_path) if module_path else model
        setattr(mod, param_name,
                torch.nn.Parameter(v.to(device), requires_grad=False))
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # Metric networks (LPIPS) ship torchvision-initialised and are absent
    # from converted original-RENI checkpoints; that is fine.
    real_missing = [k for k in missing
                    if k not in grafted
                    and "lpips" not in k and "device_indicator" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {load_dir}: missing={real_missing}, "
            f"unexpected={list(unexpected)}")
    if grafted:
        # Bank counts are baked into field/model attributes used in .view()
        # calls at inference time; sync them with the grafted shapes.
        for k, v in grafted.items():
            name = k.split(".")[-1]
            for obj in (model, model.field):
                if "train" in name and hasattr(obj, "num_train_data"):
                    obj.num_train_data = v.shape[0]
                if "eval" in name and hasattr(obj, "num_eval_data"):
                    obj.num_eval_data = v.shape[0]
        print(f"[graft] {len(grafted)} per-image banks at checkpoint sizes")

    # SH/SG fits were run with the test set as the 'training' set, so the
    # fitted coefficients live in the train_* banks and the eval_* banks hold
    # their zero/random initialisation. Mirror train -> eval when the eval
    # bank is clearly unfitted (an order of magnitude smaller than train);
    # fitted RENI eval latents and fixed geometry banks are left untouched.
    field_state = model.field.state_dict()
    for k, v in field_state.items():
        if "eval" in k and v.is_floating_point():
            train_key = k.replace("eval", "train")
            if train_key in field_state and \
                    field_state[train_key].shape == v.shape and \
                    v.abs().mean() < 0.1 * field_state[train_key].abs().mean():
                attr_eval = k.split(".")[-1]
                owner = model.field if "." not in k else \
                    model.field.get_submodule(k.rsplit(".", 1)[0])
                setattr(owner, attr_eval, torch.nn.Parameter(
                    field_state[train_key].clone().to(device),
                    requires_grad=False))
                print(f"[mirror] field.{k} <- field.{train_key} "
                      f"(eval bank was empty)")
    model.eval()
    return pipeline, pipeline.datamanager, model


def equirect_ray_bundle(device, idx: int = 0, height: int = 64):
    """Equirectangular ray bundle at the given resolution (W = 2H)."""
    H, W = height, height * 2
    cameras = Cameras(
        fx=torch.tensor(H, dtype=torch.float32).repeat(1),
        fy=torch.tensor(H, dtype=torch.float32).repeat(1),
        cx=torch.tensor(W // 2, dtype=torch.float32).repeat(1),
        cy=torch.tensor(H // 2, dtype=torch.float32).repeat(1),
        camera_to_worlds=torch.tensor(
            [[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]], dtype=torch.float32),
        camera_type=CameraType.EQUIRECTANGULAR,
    )
    ray_bundle = cameras.generate_rays(0).flatten().to(device)
    ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * idx
    return ray_bundle


def make_ray_samples(model, ray_bundle):
    """Ray samples for direct field queries (handles grad-loss ray doubling)."""
    if len(ray_bundle.directions.shape) == 3:  # [2, num_rays, 3]
        return model.create_ray_samples(
            ray_bundle.origins[0], ray_bundle.directions[0], ray_bundle.camera_indices[0])
    return model.create_ray_samples(
        ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)


def rotation_fn(model):
    return rot_y if getattr(model.field, "old_implementation", False) else rot_z


def decode_latents(model, ray_samples, latent_code, rotation=None, height: int = 64):
    """Decode a [1, latent_dim, 3] latent code to an sRGB envmap [H, W, 3]."""
    H, W = height, height * 2
    latents = latent_code.repeat(ray_samples.shape[0], 1, 1)
    outputs = model.field.forward(ray_samples, rotation=rotation, latent_codes=latents)
    img = model.field.unnormalise(outputs[RENIFieldHeadNames.RGB]).reshape(H, W, 3)
    return linear_to_sRGB(img, use_quantile=True).cpu().detach()


def render_eval_image(model, datamanager, idx: int, device, height: int = 64):
    """Render eval image idx with the model's fitted eval latents.

    Returns dict with pred_img, gt_img (sRGB, masked if eval mask present) and
    log-domain heatmaps, matching the paper's comparison figure cells.

    The eval dataset is indexed directly (datamanager.next_eval_image pops a
    RANDOM image in current nerfstudio, which would desynchronise the GT from
    the fitted latent at camera index idx).
    """
    model.eval()
    batch = datamanager.eval_dataset[idx]
    gt_raw = batch["image"]
    if gt_raw.dim() == 4:
        gt_raw = gt_raw[0]
    H, W = gt_raw.shape[0], gt_raw.shape[1]
    del height  # render at the dataset's native resolution so GT/pred align

    ray_bundle = equirect_ray_bundle(device, idx=idx, height=H)
    gt_raw = gt_raw.to(device)

    outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle, rotation=None)
    pred = model.field.unnormalise(outputs["rgb"]).reshape(H, W, 3)
    gt = model.field.unnormalise(gt_raw).reshape(H, W, 3)

    gt_gray = torch.mean(gt, dim=-1, keepdim=True)
    pred_gray = torch.mean(pred, dim=-1, keepdim=True)
    gt_min, gt_max = torch.min(gt_gray), torch.max(gt_gray)
    combined = colormaps.apply_depth_colormap(
        torch.cat([gt_gray, pred_gray], dim=1), near_plane=gt_min, far_plane=gt_max)
    gt_heat, pred_heat = combined[:, :W, :], combined[:, W:, :]

    pred_img = linear_to_sRGB(pred, use_quantile=True)
    gt_img = linear_to_sRGB(gt, use_quantile=True)
    if "mask" in batch:
        mask = batch["mask"].reshape(H, W, -1)[..., :1].expand_as(gt_img).to(device)
        gt_img = gt_img * mask

    return {
        "pred_img": pred_img.cpu().detach(),
        "gt_img": gt_img.cpu().detach(),
        "pred_heatmap": pred_heat.cpu().detach(),
        "gt_heatmap": gt_heat.cpu().detach(),
    }


def collect_model_outputs(model_specs, image_indices, device, height: int = 64):
    """Run render_eval_image over several models. model_specs: {name: path}."""
    all_outputs = {}
    for name, path in model_specs.items():
        print(f"[load] {name}: {path}")
        _, datamanager, model = load_model(Path(path), device=device)
        all_outputs[name] = {
            idx: render_eval_image(model, datamanager, idx, device, height)
            for idx in image_indices
        }
        del model, datamanager
        torch.cuda.empty_cache()
    return all_outputs


def save_figure(fig, out_stem: Path, svg: bool = False, dpi: int = 200):
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    if svg:
        fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    print(f"[saved] {out_stem}.png / .pdf" + (" / .svg" if svg else ""))


def add_common_args(parser, default_output: str):
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=64,
                        help="Envmap render height (width = 2x)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svg", action="store_true", help="Also write SVG")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "publication" / "figures" / default_output,
                        help="Output stem (no extension)")
    return parser
