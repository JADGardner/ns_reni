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
import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Nimbus Roman",
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})

import numpy as np
import cv2
import pyexr
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPO_ROOT.parents[1]
# Relative paths in saved configs (data/RENI_HDR etc.) resolve from repo root.
os.chdir(REPO_ROOT)

from nerfstudio.cameras.cameras import Cameras, CameraType  # noqa: E402
from nerfstudio.utils import colormaps  # noqa: E402

from reni.configs.reni_config import RENIField  # noqa: E402
from reni.configs.sh_sg_envmap_configs import SHField, SGField  # noqa: E402
from reni.field_components.field_heads import RENIFieldHeadNames  # noqa: E402
from reni.utils.colourspace import linear_to_sRGB  # noqa: E402
from reni.utils.tonemap import luminance as tonemap_luminance  # noqa: E402
from reni.utils.tonemap import two_bracket_to_linear  # noqa: E402
from reni.utils.utils import rot_y, rot_z  # noqa: E402

CHECKPOINTS = REPO_ROOT / "checkpoints"

# Single shared resolver (no symlinks): maps legacy 'checkpoints/<rel>' names
# onto the repo extras / paper-model archive / baselines archive roots.
from reni.utils.checkpoint_locator import find_checkpoint  # noqa: E402

try:
    PAPER_MODELS = find_checkpoint("reni_plus_plus_models").parent
except FileNotFoundError:
    # Explicit --model_dir scripts do not need the paper archive at import time.
    PAPER_MODELS = Path(os.environ.get(
        "RENI_PAPER_MODELS",
        Path.home() / "model-storage" / "reni_paper_models",
    ))

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

# Thesis-era checkpoints living in the phd umbrella outputs (present when
# running inside the phd container/repo; override via PHD_OUTPUTS).
PHD_OUTPUTS = Path(os.environ.get("PHD_OUTPUTS", "/workspace/phd/outputs"))
MODEL_DIRS["two_bracket_w3_1cyc"] = {100: PHD_OUTPUTS / "reni" / "ldrw3_2cyc_step50000"}
# test-split refit shim (eval latents fitted to the 21 test images; generated
# by eval_two_bracket_compare --save-fitted-shim) - use for figure scripts
MODEL_DIRS["two_bracket_w3_1cyc_testfit"] = {100: PHD_OUTPUTS / "reni" / "_figshim_w3_1cyc"}
MODEL_DIRS["two_bracket_w3_2cyc"] = {
    100: PHD_OUTPUTS / "reni" / "reni_latent_reset_d100_two_bracket_ldrw3_2cyc"}

_EVAL_MAPPING = None


def _eval_mapping():
    """Optional split/filename -> high-res EXR mapping for figure GT images."""
    global _EVAL_MAPPING
    if _EVAL_MAPPING is not None:
        return _EVAL_MAPPING

    mapping_path = os.environ.get("RENI_FIGURE_EVAL_MAPPING")
    if not mapping_path:
        _EVAL_MAPPING = {}
        return _EVAL_MAPPING

    high_root = Path(os.environ.get(
        "RENI_FIGURE_HIGH_ROOT",
        PHD_ROOT / "data" / "RENI_HDR_512x1024",
    )).expanduser()
    mapping = {}
    with open(Path(mapping_path).expanduser(), newline="") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            target = row["target_file"]
            high_rel = row["best_high_path"]
            if high_rel:
                mapping[(split, target)] = high_root / high_rel
    _EVAL_MAPPING = mapping
    print(f"[eval-mapping] loaded {len(mapping)} entries from {mapping_path}")
    return _EVAL_MAPPING


def mapped_eval_image_path(split: str, filename: str):
    """Return the mapped high-res EXR path for a low-res eval filename."""
    mapping = _eval_mapping()
    if not mapping:
        return None
    return mapping.get((split, filename)) or mapping.get(("test", filename))


def read_clean_exr(path: Path):
    """Read an EXR as finite, positive float32 RGB."""
    return _clean_numpy_image(pyexr.read(str(path)))


def _clean_numpy_image(image):
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    image = image[:, :, :3].astype("float32", copy=False)
    finite = np.isfinite(image)
    finite_max = float(np.max(image[finite])) if np.any(finite) else 0.0
    image = np.nan_to_num(image, nan=0.0, posinf=finite_max, neginf=0.0)
    positive = image[image > 0]
    floor = float(np.min(positive)) if positive.size else 0.0
    image[image <= 0] = floor
    return image


def _mapped_eval_image(dataset, idx: int):
    split = getattr(dataset, "split", "test")
    original_name = dataset._dataparser_outputs.image_filenames[idx].name
    image_path = mapped_eval_image_path(split, original_name)
    if image_path is None:
        return None
    if not image_path.exists():
        raise FileNotFoundError(f"Mapped high-res eval image missing: {image_path}")

    image = read_clean_exr(image_path)
    width = int(os.environ.get("RENI_FIGURE_EVAL_WIDTH", image.shape[1]))
    if image.shape[1] != width:
        image = cv2.resize(
            image,
            (width, width // 2),
            interpolation=cv2.INTER_LINEAR,
        )

    tensor = torch.from_numpy(image)
    metadata = dataset._dataparser_outputs.metadata
    if metadata["convert_to_ldr"]:
        tensor = linear_to_sRGB(tensor)
    if metadata["convert_to_log_domain"]:
        tensor = torch.log(tensor + 1e-8)
    if metadata["min_max_normalize"]:
        min_val, max_val = dataset.metadata["min_max"]
        tensor = (tensor - min_val) / (max_val - min_val) * 2 - 1
    return tensor


def eval_image_tensor(dataset, idx: int, batch=None):
    """Return the figure GT tensor, using the high-res mapping when enabled."""
    gt_raw = _mapped_eval_image(dataset, idx)
    if gt_raw is None:
        if batch is None:
            batch = dataset[idx]
        gt_raw = batch["image"]
    if gt_raw.dim() == 4:
        gt_raw = gt_raw[0]
    return gt_raw


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

    ckpt = torch.load(
        ckpt_dir / f"step-{load_step:09d}.ckpt",
        map_location=device,
        weights_only=False,
    )
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
        # Two-bracket (complementary tonemapping) fits carry fixed-gauge
        # normalisation and 6-channel tonemap targets instead of min-max/log.
        # Absent in older configs -> defaults leave the 3-channel path unchanged.
        dp.fixed_gauge_normalisation = saved_dp.get("fixed_gauge_normalisation", False)
        dp.fixed_gauge_percentile = saved_dp.get("fixed_gauge_percentile", 0.99)
        dp.fixed_gauge_target = saved_dp.get("fixed_gauge_target", 1.0)
        dp.tonemap_targets = saved_dp.get("tonemap_targets", False)
        dp.tonemap_m_ldr = saved_dp.get("tonemap_m_ldr", 16.0)
        dp.tonemap_m_log = saved_dp.get("tonemap_m_log", 10000.0)

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
        # Two-bracket fits emit six sigmoid channels; older configs omit the key
        # and stay at the 3-channel RGB default.
        if "out_features" in field_cfg:
            m.field.out_features = field_cfg["out_features"]
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
        zero_stride_target = (
            k in model_state and any(stride == 0 for stride in model_state[k].stride())
        )
        if k in model_state and (model_state[k].shape != v.shape or zero_stride_target):
            grafted[k] = v
        else:
            filtered[k] = v
    for k, v in grafted.items():
        module_path, _, param_name = k.rpartition(".")
        mod = model.get_submodule(module_path) if module_path else model
        value = v.to(device)
        if param_name in mod._parameters:
            setattr(mod, param_name,
                    torch.nn.Parameter(value, requires_grad=False))
        elif param_name in mod._buffers:
            mod._buffers[param_name] = value
        else:
            setattr(mod, param_name, value)
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


def decode_latents(model, ray_samples, latent_code, rotation=None,
                   height: int = 64, chunk_size: Optional[int] = None):
    """Decode a [1, latent_dim, 3] latent code to an sRGB envmap [H, W, 3]."""
    H, W = height, height * 2
    if chunk_size is None:
        chunk_size = int(os.environ.get("RENI_FIGURE_DECODE_CHUNK", 65536))
    if chunk_size <= 0:
        chunk_size = ray_samples.shape[0]

    chunks = []
    with torch.no_grad():
        for start in range(0, ray_samples.shape[0], chunk_size):
            end = min(start + chunk_size, ray_samples.shape[0])
            chunk = ray_samples[start:end]
            latents = latent_code.repeat(chunk.shape[0], 1, 1)
            outputs = model.field.forward(
                chunk,
                rotation=rotation,
                latent_codes=latents,
            )
            raw = outputs[RENIFieldHeadNames.RGB]
            # Two-bracket models emit six sigmoid channels; reconstruct linear
            # HDR exactly as RENIModel._to_linear_hdr does (bracket inversion +
            # sigmoid blend, tau/delta at the tonemap module defaults). The
            # 3-channel path is unchanged.
            if getattr(model, "two_bracket", False):
                rgb = two_bracket_to_linear(
                    raw, m_ldr=model.tonemap_m_ldr, m_log=model.tonemap_m_log)
            else:
                rgb = model.field.unnormalise(raw)
            chunks.append(rgb.cpu())

    img = torch.cat(chunks, dim=0).reshape(H, W, 3)
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
    gt_raw = eval_image_tensor(datamanager.eval_dataset, idx, batch=batch)
    H, W = gt_raw.shape[0], gt_raw.shape[1]
    del height  # render at the dataset's native resolution so GT/pred align

    ray_bundle = equirect_ray_bundle(device, idx=idx, height=H)
    gt_raw = gt_raw.to(device)

    with torch.no_grad():
        outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle, rotation=None)
        # _to_linear_hdr handles both the standard log/min-max unnormalise and
        # the 6-channel two-bracket blend; field.unnormalise alone is wrong
        # (identity) for two-bracket models.
        def _to_linear(x):
            # The camera-bundle path views flat [N, 6] two-bracket outputs
            # through the flattened bundle's shape (e.g. [N, 3, 2]), which
            # interleaves channels if consumed dimension-wise. Memory order
            # is untouched, so a straight reshape restores the true
            # [R,G,B]_ldr + [R,G,B]_log layout. Log-domain models pass
            # through to the standard unnormalise.
            if getattr(model, "two_bracket", False) and x.shape[-1] != 6:
                x = x.reshape(-1, 6)
            return model._to_linear_hdr(x)

        pred = _to_linear(outputs["rgb"]).reshape(H, W, 3)
        gt = _to_linear(gt_raw).reshape(H, W, 3)

        # Log-compressed BT.709 luminance: linear HDR under a linear min/max
        # colormap crushes everything but the sun to the colormap floor.
        gt_gray = torch.log1p(tonemap_luminance(gt).clamp_min(0.0)).unsqueeze(-1)
        pred_gray = torch.log1p(tonemap_luminance(pred).clamp_min(0.0)).unsqueeze(-1)
        gt_min, gt_max = torch.min(gt_gray), torch.max(gt_gray)
        combined = colormaps.apply_depth_colormap(
            torch.cat([gt_gray, pred_gray], dim=1),
            near_plane=gt_min,
            far_plane=gt_max,
        )
        gt_heat, pred_heat = combined[:, :W, :], combined[:, W:, :]

        pred_img = linear_to_sRGB(pred, use_quantile=True)
        gt_img = linear_to_sRGB(gt, use_quantile=True)
    if "mask" in batch:
        mask = batch["mask"]
        if mask.dim() == 4:
            mask = mask[0]
        if mask.shape[:2] != (H, W):
            mask = torch.nn.functional.interpolate(
                mask.permute(2, 0, 1)[None].float(),
                size=(H, W),
                mode="nearest",
            )[0].permute(1, 2, 0).bool()
        mask = mask.reshape(H, W, -1)[..., :1].expand_as(gt_img).to(device)
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


def axis_center(ax):
    bbox = ax.get_position()
    return (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2


def axes_span_center_x(*axes):
    bboxes = [ax.get_position() for ax in axes]
    return (min(b.x0 for b in bboxes) + max(b.x1 for b in bboxes)) / 2


def add_figure_label(fig, x, y, text, fontsize, rotation=0):
    fig.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontfamily="serif",
        rotation=rotation,
        color="black",
        zorder=20,
    )


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
    parser.add_argument("--decode_chunk", type=int,
                        default=int(os.environ.get("RENI_FIGURE_DECODE_CHUNK", 65536)),
                        help="Rays per chunk for direct latent envmap decoding; "
                             "use 0 to decode a full envmap in one pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svg", action="store_true", help="Also write SVG")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "publication" / "figures" / default_output,
                        help="Output stem (no extension)")
    return parser
