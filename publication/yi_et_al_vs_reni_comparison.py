#!/usr/bin/env python3
"""
Yi et al. (CVPR 2023) vs RENI++ Comparison Script

Compares Yi et al.'s inverse rendering SH lighting estimation against RENI++
on the task of recovering illumination from a synthetically rendered object.

Pipeline:
1. Load GT HDR environment maps from Laval Sky HDR dataset
2. Render a synthetic object (bunny) under each GT envmap using Blinn-Phong shading
3. Feed the rendered object image + mask to Yi et al.'s model → SH coefficients
4. Run RENI++ inverse rendering (latent code optimisation) → neural field envmap
5. Compare both recovered illuminations against GT (PSNR, SSIM)

Usage:
    python publication/yi_et_al_vs_reni_comparison.py
    python publication/yi_et_al_vs_reni_comparison.py --num_images 5
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

# Limit CPU thread usage to prevent system freezing
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# RENI++ imports
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.illumination_fields.sh_illumination_field import shReconstructSignal
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni.model_components.shaders import BlinnPhongShader
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.utils import find_nerfstudio_project_root
from nerfstudio.cameras.cameras import Cameras, CameraType

# Yi et al. imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdparty" / "Yi_et_al_relighting" / "InverseRendering"))
from model import InverseRenderModel

# EXR loading
import pyexr

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yi et al. model loading and inference
# ---------------------------------------------------------------------------

def load_yi_model(model_path: Path, device: torch.device) -> InverseRenderModel:
    """Load Yi et al.'s pretrained InverseRenderModel."""
    net = InverseRenderModel()
    logger.info(f"Loading Yi et al. model from {model_path}")
    state_dict = torch.load(str(model_path), map_location=device, weights_only=False)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    return net


def yi_predict_sh(
    net: InverseRenderModel,
    rendered_image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run Yi et al.'s model on a rendered object image to get SH coefficients.

    Args:
        net: Loaded InverseRenderModel
        rendered_image: LDR rendered image [H, W, 3] float32 in [0, 1]
        mask: Binary mask [H, W] float32
        device: torch device

    Returns:
        SH coefficients [9, 3] (monochrome SH broadcast to 3 channels)
    """
    import torchvision.transforms as transforms

    h, w = rendered_image.shape[:2]

    # Prepare normalised input (ImageNet-style normalisation as used in their code)
    norm_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    plain_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Convert to PIL for their transform pipeline
    img_uint8 = (np.clip(rendered_image.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)

    mask_uint8 = (np.clip(mask.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode="L")

    input_norm = norm_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 256, 256]
    input_nonorm = plain_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 256, 256]
    mask_tensor = plain_transform(mask_pil).unsqueeze(0).to(device)  # [1, 1, 256, 256]
    mask_tensor[mask_tensor < 0.5] = 0
    mask_tensor[mask_tensor >= 0.5] = 1.0

    # Apply mask to input (as done in their code)
    input_norm = input_norm * mask_tensor

    with torch.no_grad():
        albedo, shading, normal, sh_coeffs = net.forward(input_norm, input_nonorm, mask_tensor)

    # sh_coeffs is [1, 9] (monochrome) — broadcast to [9, 3]
    sh_coeffs = sh_coeffs[0]  # [9]
    sh_coeffs_rgb = sh_coeffs.unsqueeze(-1).expand(-1, 3)  # [9, 3]

    return sh_coeffs_rgb


# ---------------------------------------------------------------------------
# RENI++ inverse rendering
# ---------------------------------------------------------------------------

def load_reni_decoder(ckpt_path: Path, ckpt_step: int, device: torch.device) -> RENIField:
    """Load pretrained RENI++ decoder."""
    config = RENIFieldConfig(
        conditioning="Attention",
        invariant_function="VN",
        equivariance="SO2",
        axis_of_invariance="z",
        positional_encoding="NeRF",
        encoded_input="Directions",
        latent_dim=100,
        hidden_features=128,
        hidden_layers=9,
        mapping_layers=5,
        mapping_features=128,
        num_attention_heads=8,
        num_attention_layers=6,
        output_activation="None",
        last_layer_linear=True,
        fixed_decoder=True,
        trainable_scale=False,
    )

    field = config.setup(num_train_data=None, num_eval_data=None)

    project_root = Path(__file__).resolve().parent.parent
    full_ckpt_path = project_root / ckpt_path / "nerfstudio_models" / f"step-{ckpt_step:09d}.ckpt"

    if not full_ckpt_path.exists():
        raise ValueError(f"Checkpoint not found at {full_ckpt_path}")

    logger.info(f"Loading RENI decoder from {full_ckpt_path}")
    ckpt = torch.load(str(full_ckpt_path), map_location=device, weights_only=False)

    illumination_field_dict = {}
    match_str = "_model.field."
    ignore_strs = ["train_logvar", "eval_logvar", "train_mu", "eval_mu"]

    for key in ckpt["pipeline"].keys():
        if key.startswith(match_str) and not any(ig in key for ig in ignore_strs):
            illumination_field_dict[key[len(match_str):]] = ckpt["pipeline"][key]

    field.load_state_dict(illumination_field_dict, strict=False)
    field = field.to(device)
    field.eval()

    for param in field.parameters():
        param.requires_grad = False

    return field


def decode_reni_to_envmap(
    reni_field: RENIField,
    latent_codes: torch.Tensor,
    scale: torch.Tensor,
    ray_samples,
    num_dirs: int,
    sampler_height: int,
    sampler_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode latent codes through RENI to get environment map."""
    latents_expanded = latent_codes.unsqueeze(0).expand(num_dirs, -1, -1)
    scale_expanded = scale.expand(num_dirs)

    with torch.set_grad_enabled(latent_codes.requires_grad):
        outputs = reni_field.forward(
            ray_samples=ray_samples, latent_codes=latents_expanded, scale=scale_expanded
        )

    hdr_colors = outputs[RENIFieldHeadNames.RGB]
    hdr_colors = reni_field.unnormalise(hdr_colors)
    env_map = hdr_colors.reshape(sampler_height, sampler_width, 3)
    return env_map


def reni_inverse_render(
    reni_field: RENIField,
    gt_render: torch.Tensor,
    normals: torch.Tensor,
    mask: torch.Tensor,
    light_directions: torch.Tensor,
    view_directions: torch.Tensor,
    shader: BlinnPhongShader,
    sampler,
    device: torch.device,
    num_steps: int = 500,
    lr: float = 0.01,
    specular_term: float = 0.2,
    shininess: float = 500.0,
) -> torch.Tensor:
    """Run RENI++ inverse rendering optimisation and return recovered envmap."""
    latent_codes = nn.Parameter(torch.zeros(reni_field.latent_dim, 3, device=device))
    scale = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([latent_codes, scale], lr=lr)
    l2_loss_fn = nn.MSELoss()
    cosine_sim = nn.CosineSimilarity(dim=-1)

    # Pre-compute ray samples on GPU once (avoids CPU→GPU transfer every step)
    ray_samples = sampler.generate_direction_samples().to(device)
    num_dirs = ray_samples.frustums.directions.shape[0]
    ray_samples.camera_indices = torch.zeros(num_dirs, dtype=torch.long, device=device)

    for step in range(num_steps + 1):
        optimizer.zero_grad()

        pred_envmap = decode_reni_to_envmap(
            reni_field, latent_codes, torch.exp(scale),
            ray_samples, num_dirs, sampler.height, sampler.width, device,
        )
        pred_render = render_with_environment(
            normals, mask, pred_envmap, light_directions, view_directions,
            shader, specular_term, shininess,
        )

        q = torch.quantile(gt_render.flatten(), 0.98)
        gt_srgb = linear_to_sRGB(gt_render, q=q, clamp=False)
        pred_srgb = linear_to_sRGB(pred_render, q=q, clamp=False)

        rgb_loss = l2_loss_fn(pred_srgb, gt_srgb)
        cos_loss = 1.0 - cosine_sim(pred_srgb.reshape(-1, 3), gt_srgb.reshape(-1, 3)).mean()
        prior_loss = torch.mean(latent_codes ** 2)
        total_loss = 100.0 * rgb_loss + 1.0 * cos_loss + 0.001 * prior_loss

        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_envmap = decode_reni_to_envmap(
            reni_field, latent_codes, torch.exp(scale),
            ray_samples, num_dirs, sampler.height, sampler.width, device,
        )

    return final_envmap


# ---------------------------------------------------------------------------
# Rendering utilities (from inverse_rendering_demo.py)
# ---------------------------------------------------------------------------

def load_environment_map(env_path: Path, target_height: int = 64, target_width: int = 128) -> torch.Tensor:
    """Load and preprocess an HDR environment map."""
    env_map = pyexr.read(str(env_path)).astype("float32")
    # Take only RGB channels (EXR may have alpha)
    if env_map.ndim == 3 and env_map.shape[2] > 3:
        env_map = env_map[:, :, :3]
    env_map[env_map == np.inf] = np.nanmax(env_map[env_map != np.inf])
    env_map[env_map <= 0] = np.nanmin(env_map[env_map > 0])
    env_map = torch.tensor(env_map).float()
    env_map = (
        F.interpolate(
            env_map.unsqueeze(0).permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    return env_map


def load_normal_map(normal_path: Path, target_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load normal map and create mask."""
    normals = pyexr.read(str(normal_path)).astype("float32")
    normals = torch.tensor(normals).float()
    normals = (
        F.interpolate(
            normals.unsqueeze(0).permute(0, 3, 1, 2),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    norms = torch.norm(normals, dim=-1, keepdim=True)
    mask = (norms.squeeze(-1) > 0.5) & (norms.squeeze(-1) < 1.5)
    normals[mask] = normals[mask] / norms[mask]
    normals[~mask] = 0
    normals[:, :, 1] = -normals[:, :, 1]
    return normals, mask


def create_view_directions(image_size: int, device: torch.device) -> torch.Tensor:
    """Create view directions for rendering."""
    camera_angle_x = 0.6911112070083618
    focal_length = 0.5 * image_size / np.tan(0.5 * camera_angle_x)
    cameras = Cameras(
        camera_to_worlds=torch.eye(4)[:3].unsqueeze(0),
        fx=torch.tensor([focal_length]),
        fy=torch.tensor([focal_length]),
        cx=torch.tensor([image_size / 2.0]),
        cy=torch.tensor([image_size / 2.0]),
        width=torch.tensor([image_size]),
        height=torch.tensor([image_size]),
        camera_type=CameraType.PERSPECTIVE,
    )
    rays = cameras.generate_rays(0)
    view_dirs = rays.directions.reshape(-1, 3)
    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
    return view_dirs.to(device)


def render_with_environment(
    normals: torch.Tensor,
    mask: torch.Tensor,
    env_map: torch.Tensor,
    light_directions: torch.Tensor,
    view_directions: torch.Tensor,
    shader: BlinnPhongShader,
    specular_term: float = 0.2,
    shininess: float = 500.0,
    background_color: float = 0.0,
) -> torch.Tensor:
    """Render object with environment map illumination (masked rendering)."""
    image_size = normals.shape[0]
    normals_flat = normals.reshape(-1, 3)
    mask_flat = mask.reshape(-1)

    env_flat = env_map.reshape(-1, 3)
    light_colors = env_flat.unsqueeze(0)
    light_dirs = light_directions.unsqueeze(0)

    valid_indices = mask_flat.nonzero(as_tuple=True)[0]
    num_valid = valid_indices.shape[0]

    if num_valid > 0:
        valid_normals = normals_flat[valid_indices]
        valid_view_dirs = view_directions[valid_indices]
        valid_specular = torch.ones_like(valid_normals) * specular_term
        valid_albedo = 1 - valid_specular
        valid_shin = torch.ones(num_valid, device=normals.device) * shininess

        valid_rendered = shader(
            albedo=valid_albedo,
            normals=valid_normals,
            light_directions=light_dirs,
            light_colors=light_colors,
            specular=valid_specular,
            shininess=valid_shin,
            view_directions=valid_view_dirs,
            detach_normals=True,
        )

        rendered_flat = torch.full(
            (normals_flat.shape[0], 3), background_color, device=normals.device
        )
        rendered_flat[valid_indices] = valid_rendered
    else:
        rendered_flat = torch.full(
            (normals_flat.shape[0], 3), background_color, device=normals.device
        )

    return rendered_flat.reshape(image_size, image_size, 3)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute LDR PSNR and SSIM between two HDR environment maps."""
    gt = np.clip(gt, 0, None)
    pred = np.clip(pred, 0, None)

    # LDR tone-mapping for perceptually meaningful metrics
    p999 = np.percentile(gt, 99.9)
    if p999 > 0:
        gt_norm = gt / p999
        pred_norm = pred / p999
    else:
        gt_norm = gt
        pred_norm = pred

    gt_ldr = np.clip(gt_norm ** (1.0 / 2.2), 0, 1)
    pred_ldr = np.clip(pred_norm ** (1.0 / 2.2), 0, 1)

    gt_uint8 = (gt_ldr * 255).astype(np.uint8)
    pred_uint8 = (pred_ldr * 255).astype(np.uint8)

    ldr_psnr = psnr(gt_uint8, pred_uint8, data_range=255)
    ssim_val = ssim(gt_uint8, pred_uint8, data_range=255, channel_axis=2)

    return {"LDR_PSNR": ldr_psnr, "SSIM": ssim_val}


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Yi et al. vs RENI++ inverse rendering comparison")
    parser.add_argument("--num_images", type=int, default=10, help="Number of test envmaps")
    parser.add_argument("--output_dir", type=str, default="publication/figures_yi_et_al")
    parser.add_argument("--reni_steps", type=int, default=500, help="RENI++ optimisation steps")
    parser.add_argument("--image_size", type=int, default=128, help="Rendered object image size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    illumination_width = 128
    illumination_height = 64

    # --- Load models ---
    logger.info("Loading Yi et al. model...")
    yi_model_path = project_root / "thirdparty" / "Yi_et_al_relighting" / "InverseRendering" / "path" / "invrender.pth"
    yi_net = load_yi_model(yi_model_path, device)

    logger.info("Loading RENI++ decoder...")
    reni_ckpt_path = Path("checkpoints/reni_plus_plus_models/latent_dim_100")
    reni_field = load_reni_decoder(reni_ckpt_path, ckpt_step=50000, device=device)

    # --- Load normal map ---
    normal_map_path = project_root / "data" / "RENI_HDR" / "3d_models" / "normal_maps" / "bunny_normals.exr"
    normals, mask = load_normal_map(normal_map_path, args.image_size)
    normals = normals.to(device)
    mask = mask.to(device)

    # --- Setup rendering ---
    sampler = EquirectangularSamplerConfig(
        width=illumination_width, apply_random_rotation=False, remove_lower_hemisphere=False,
    ).setup()
    shader = BlinnPhongShader()

    ray_samples = sampler.generate_direction_samples()
    light_directions = ray_samples.frustums.directions.to(device)
    light_directions = light_directions / torch.norm(light_directions, dim=-1, keepdim=True)
    view_directions = create_view_directions(args.image_size, device)

    # --- Find test environment maps ---
    data_dir = project_root / "data" / "RENI_HDR" / "val"
    exr_files = sorted(data_dir.glob("*.exr"))[:args.num_images]
    logger.info(f"Found {len(exr_files)} test environment maps")

    # --- Run comparison ---
    all_metrics = {"Yi et al.": [], "RENI++": []}
    saved_results = []  # Store per-image results for grid figure

    for idx, exr_path in enumerate(tqdm(exr_files, desc="Processing")):
        logger.info(f"\n--- Image {idx + 1}/{len(exr_files)}: {exr_path.name} ---")

        # Load GT environment map
        gt_envmap = load_environment_map(exr_path, illumination_height, illumination_width).to(device)

        # Render object under GT illumination
        with torch.no_grad():
            gt_render = render_with_environment(
                normals, mask, gt_envmap, light_directions, view_directions, shader,
                background_color=1.0,
            )

        # --- Yi et al. ---
        # Convert HDR render to LDR for Yi et al. (their model expects LDR input)
        with torch.no_grad():
            gt_render_ldr = linear_to_sRGB(gt_render, use_quantile=True)
            gt_render_ldr = torch.clamp(gt_render_ldr, 0, 1)
            # Composite white background after tone-mapping (avoids quantile crushing bg)
            mask_3ch = mask.unsqueeze(-1).expand_as(gt_render_ldr).float()
            gt_render_ldr = gt_render_ldr * mask_3ch + (1.0 - mask_3ch)

        yi_sh_coeffs = yi_predict_sh(yi_net, gt_render_ldr, mask.float(), device)
        yi_envmap = shReconstructSignal(yi_sh_coeffs, width=illumination_width, device=device)
        yi_envmap = torch.clamp(yi_envmap, min=0)

        # --- RENI++ ---
        reni_envmap = reni_inverse_render(
            reni_field, gt_render, normals, mask, light_directions, view_directions,
            shader, sampler, device, num_steps=args.reni_steps,
        )

        # --- Compute metrics ---
        gt_np = gt_envmap.cpu().numpy()
        yi_np = yi_envmap.detach().cpu().numpy()
        reni_np = reni_envmap.detach().cpu().numpy()

        yi_metrics = compute_metrics(gt_np, yi_np)
        reni_metrics = compute_metrics(gt_np, reni_np)

        all_metrics["Yi et al."].append(yi_metrics)
        all_metrics["RENI++"].append(reni_metrics)

        logger.info(f"  Yi et al.:  LDR PSNR={yi_metrics['LDR_PSNR']:.2f}, SSIM={yi_metrics['SSIM']:.4f}")
        logger.info(f"  RENI++:     LDR PSNR={reni_metrics['LDR_PSNR']:.2f}, SSIM={reni_metrics['SSIM']:.4f}")

        saved_results.append({
            "gt_envmap": gt_envmap.cpu(),
            "yi_envmap": yi_envmap.detach().cpu(),
            "reni_envmap": reni_envmap.detach().cpu(),
            "gt_render_ldr": gt_render_ldr.cpu(),
        })

        # --- Save individual images ---
        img_dir = output_dir / f"image_{idx:03d}"
        img_dir.mkdir(parents=True, exist_ok=True)

        render_np = np.clip(gt_render_ldr.cpu().numpy(), 0, 1)
        gt_env_np = np.clip(linear_to_sRGB(gt_envmap, use_quantile=True).cpu().numpy(), 0, 1)
        yi_env_np = np.clip(linear_to_sRGB(yi_envmap, use_quantile=True).detach().cpu().numpy(), 0, 1)
        reni_env_np = np.clip(linear_to_sRGB(reni_envmap, use_quantile=True).detach().cpu().numpy(), 0, 1)

        Image.fromarray((render_np * 255).astype(np.uint8)).save(img_dir / "input_render.png")
        Image.fromarray((gt_env_np * 255).astype(np.uint8)).save(img_dir / "gt_envmap.png")
        Image.fromarray((yi_env_np * 255).astype(np.uint8)).save(img_dir / "yi_envmap.png")
        Image.fromarray((reni_env_np * 255).astype(np.uint8)).save(img_dir / "reni_envmap.png")
        logger.info(f"  Saved individual images to {img_dir}")

    # --- Aggregate metrics ---
    mean_metrics = {}
    for method, metrics_list in all_metrics.items():
        mean_metrics[method] = {
            k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]
        }

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    for method, m in mean_metrics.items():
        logger.info(f"  {method:20s}  LDR PSNR={m['LDR_PSNR']:.2f}  SSIM={m['SSIM']:.4f}")

    # --- Generate LaTeX table ---
    table_lines = [
        r"\begin{tabular}{l|cc}",
        r"\hline",
        r"Method & LDR PSNR$\uparrow$ & SSIM$\uparrow$ \\",
        r"\hline",
    ]
    for method, m in mean_metrics.items():
        table_lines.append(f"{method} & {m['LDR_PSNR']:.2f} & {m['SSIM']:.4f} \\\\")
    table_lines.extend([r"\hline", r"\end{tabular}"])
    table_str = "\n".join(table_lines)

    table_path = output_dir / "metrics.tex"
    with open(table_path, "w") as f:
        f.write(table_str)
    logger.info(f"Saved {table_path}")

    # --- Generate comparison figure (last image) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), gridspec_kw={'width_ratios': [1, 2, 2, 2]})

    gt_disp = linear_to_sRGB(gt_envmap, use_quantile=True).cpu().numpy()
    yi_disp = linear_to_sRGB(yi_envmap, use_quantile=True).detach().cpu().numpy()
    reni_disp = linear_to_sRGB(reni_envmap, use_quantile=True).detach().cpu().numpy()
    render_disp = gt_render_ldr.cpu().numpy()

    axes[0].imshow(render_disp)
    axes[0].set_title("Rendered Object\n(Input)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(np.clip(yi_disp, 0, 1))
    axes[1].set_title(f"Yi et al. (SH)\nLDR PSNR: {yi_metrics['LDR_PSNR']:.1f}", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(np.clip(reni_disp, 0, 1))
    axes[2].set_title(f"RENI++\nLDR PSNR: {reni_metrics['LDR_PSNR']:.1f}", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(np.clip(gt_disp, 0, 1))
    axes[3].set_title("Ground Truth", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path}")

    # --- Generate multi-image comparison grid ---
    if len(saved_results) > 1:
        n_show = min(len(saved_results), 5)
        fig, axes = plt.subplots(n_show, 4, figsize=(20, 4 * n_show), gridspec_kw={'width_ratios': [1, 2, 2, 2]})
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for row_idx in range(n_show):
            res = saved_results[row_idx]
            gt_d = linear_to_sRGB(res["gt_envmap"], use_quantile=True).numpy()
            yi_d = linear_to_sRGB(res["yi_envmap"], use_quantile=True).numpy()
            reni_d = linear_to_sRGB(res["reni_envmap"], use_quantile=True).numpy()

            axes[row_idx, 0].imshow(res["gt_render_ldr"].numpy())
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].imshow(np.clip(yi_d, 0, 1))
            axes[row_idx, 1].axis("off")
            axes[row_idx, 2].imshow(np.clip(reni_d, 0, 1))
            axes[row_idx, 2].axis("off")
            axes[row_idx, 3].imshow(np.clip(gt_d, 0, 1))
            axes[row_idx, 3].axis("off")

            if row_idx == 0:
                axes[row_idx, 0].set_title("Rendered Object", fontsize=11)
                axes[row_idx, 1].set_title("Yi et al. (SH)", fontsize=11)
                axes[row_idx, 2].set_title("RENI++", fontsize=11)
                axes[row_idx, 3].set_title("Ground Truth", fontsize=11)

        plt.tight_layout()
        grid_path = output_dir / "comparison_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {grid_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
