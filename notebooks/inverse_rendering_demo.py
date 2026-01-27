"""
RENI++ Inverse Rendering Demo Script

This script demonstrates the inverse rendering process:
1. Load a pretrained RENI decoder (frozen)
2. Render an object under ground truth HDR illumination
3. Optimize latent codes so RENI-decoded illumination matches the GT render
4. Save comparison images at various steps

Usage:
    python scripts/inverse_rendering_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyexr
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nerfstudio.cameras.cameras import Cameras, CameraType
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni.model_components.shaders import BlinnPhongShader
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.utils import find_nerfstudio_project_root


def load_pretrained_reni_decoder(
    ckpt_path: Path, ckpt_step: int, device: torch.device
) -> RENIField:
    """Load the pretrained RENI decoder from checkpoint."""

    # Create RENI field config matching the pretrained model
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

    # Setup the field
    field = config.setup(num_train_data=None, num_eval_data=None)

    # Load checkpoint
    project_root = find_nerfstudio_project_root(Path(__file__))
    full_ckpt_path = (
        project_root / ckpt_path / "nerfstudio_models" / f"step-{ckpt_step:09d}.ckpt"
    )

    if not full_ckpt_path.exists():
        raise ValueError(f"Checkpoint not found at {full_ckpt_path}")

    print(f"Loading RENI decoder from {full_ckpt_path}")
    ckpt = torch.load(str(full_ckpt_path), map_location=device)

    # Extract decoder weights
    illumination_field_dict = {}
    match_str = "_model.field."
    ignore_strs = ["train_logvar", "eval_logvar", "train_mu", "eval_mu"]

    for key in ckpt["pipeline"].keys():
        if key.startswith(match_str) and not any(
            ignore in key for ignore in ignore_strs
        ):
            illumination_field_dict[key[len(match_str) :]] = ckpt["pipeline"][key]

    field.load_state_dict(illumination_field_dict, strict=False)
    field = field.to(device)
    field.eval()

    # Freeze decoder
    for param in field.parameters():
        param.requires_grad = False

    return field


def load_environment_map(
    env_path: Path, target_height: int = 64, target_width: int = 128
) -> torch.Tensor:
    """Load and preprocess an HDR environment map."""
    env_map = pyexr.read(str(env_path)).astype("float32")

    # Handle inf/negative values
    env_map[env_map == np.inf] = np.nanmax(env_map[env_map != np.inf])
    env_map[env_map <= 0] = np.nanmin(env_map[env_map > 0])

    env_map = torch.tensor(env_map).float()

    # Resize to target resolution
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


def load_normal_map(normal_path: Path, target_size: int = 128) -> tuple:
    """Load normal map and create mask."""
    # Use pyexr for EXR files to preserve float32 [-1, 1] range
    if str(normal_path).endswith(".exr"):
        normals = pyexr.read(str(normal_path)).astype("float32")
    else:
        import imageio

        normals = imageio.v2.imread(str(normal_path))
        # Convert uint8 [0, 255] to [-1, 1] if needed
        if normals.max() > 1.0:
            normals = normals / 255.0 * 2.0 - 1.0

    normals = torch.tensor(normals).float()

    # Resize
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

    # Create mask based on normal magnitude
    # Valid normals should have magnitude close to 1
    norms = torch.norm(normals, dim=-1, keepdim=True)
    mask = (norms.squeeze(-1) > 0.5) & (norms.squeeze(-1) < 1.5)

    # Renormalize valid normals to unit length
    normals[mask] = normals[mask] / norms[mask]

    # Set invalid normals to zero
    normals[~mask] = 0

    # Invert y axis to match nerfstudio convention
    normals[:, :, 1] = -normals[:, :, 1]

    return normals, mask


def create_camera_and_view_directions(
    image_size: int, device: torch.device
) -> torch.Tensor:
    """Create view directions for rendering."""
    # Simple orthographic-like camera looking at object
    camera_angle_x = 0.6911112070083618  # ~40 degrees FOV
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
    background_color: float = 1.0,  # White background
    use_masked_rendering: bool = True,  # Only compute for valid pixels
) -> torch.Tensor:
    """Render object with given environment map illumination.

    Optimized for memory efficiency using:
    - Broadcasting instead of tensor expansion (saves ~3GB)
    - Masked rendering to only compute for valid pixels (saves ~50% compute)
    """
    image_size = normals.shape[0]
    normals_flat = normals.reshape(-1, 3)
    mask_flat = mask.reshape(-1)

    # Prepare light colors and directions for BROADCASTING (1, M, 3) not (N, M, 3)
    env_flat = env_map.reshape(-1, 3)  # M x 3
    light_colors = env_flat.unsqueeze(0)  # (1, M, 3) - will broadcast
    light_dirs = light_directions.unsqueeze(0)  # (1, M, 3) - will broadcast

    if use_masked_rendering:
        # Only compute shading for valid pixels - saves significant compute
        valid_indices = mask_flat.nonzero(as_tuple=True)[0]
        num_valid = valid_indices.shape[0]

        if num_valid > 0:
            # Extract only valid pixels
            valid_normals = normals_flat[valid_indices]  # K x 3
            valid_view_dirs = view_directions[valid_indices]  # K x 3

            # Material properties for valid pixels only
            valid_specular = torch.ones_like(valid_normals) * specular_term
            valid_albedo = 1 - valid_specular
            valid_shin = torch.ones(num_valid, device=normals.device) * shininess

            # Render only valid pixels
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

            # Scatter back to full image with background color
            rendered_flat = torch.full(
                (normals_flat.shape[0], 3), background_color, device=normals.device
            )
            rendered_flat[valid_indices] = valid_rendered
        else:
            rendered_flat = torch.full(
                (normals_flat.shape[0], 3), background_color, device=normals.device
            )
    else:
        # Full rendering but still using broadcasting for memory efficiency
        specular = torch.ones_like(normals_flat) * specular_term
        albedo = 1 - specular
        shin = torch.ones(normals_flat.shape[0], device=normals.device) * shininess

        # Zero out masked regions
        albedo[~mask_flat] = 0
        specular[~mask_flat] = 0
        shin[~mask_flat] = 0

        rendered_flat = shader(
            albedo=albedo,
            normals=normals_flat,
            light_directions=light_dirs,
            light_colors=light_colors,
            specular=specular,
            shininess=shin,
            view_directions=view_directions,
            detach_normals=True,
        )
        # Set background
        rendered_flat[~mask_flat] = background_color

    rendered = rendered_flat.reshape(image_size, image_size, 3)
    return rendered


def decode_reni_to_envmap(
    reni_field: RENIField,
    latent_codes: torch.Tensor,
    scale: torch.Tensor,
    sampler,
    device: torch.device,
) -> torch.Tensor:
    """Decode latent codes through RENI to get environment map."""
    ray_samples = sampler.generate_direction_samples()
    ray_samples = ray_samples.to(device)

    # Expand latents for all directions
    num_dirs = ray_samples.frustums.directions.shape[0]
    latents_expanded = latent_codes.unsqueeze(0).expand(num_dirs, -1, -1)
    scale_expanded = scale.expand(num_dirs)

    ray_samples.camera_indices = torch.zeros(num_dirs, dtype=torch.long, device=device)

    with torch.set_grad_enabled(latent_codes.requires_grad):
        outputs = reni_field.forward(
            ray_samples=ray_samples, latent_codes=latents_expanded, scale=scale_expanded
        )

    hdr_colors = outputs[RENIFieldHeadNames.RGB]
    hdr_colors = reni_field.unnormalise(hdr_colors)

    height = sampler.height
    width = sampler.width
    env_map = hdr_colors.reshape(height, width, 3)

    return env_map


def save_comparison_image(
    gt_render: torch.Tensor,
    pred_render: torch.Tensor,
    gt_envmap: torch.Tensor,
    pred_envmap: torch.Tensor,
    step: int,
    output_dir: Path,
    loss: float,
):
    """Save a comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convert to displayable format
    gt_render_disp = linear_to_sRGB(gt_render, use_quantile=True).cpu().numpy()
    pred_render_disp = linear_to_sRGB(pred_render, use_quantile=True).cpu().numpy()
    gt_env_disp = linear_to_sRGB(gt_envmap, use_quantile=True).cpu().numpy()
    pred_env_disp = linear_to_sRGB(pred_envmap, use_quantile=True).cpu().numpy()

    axes[0, 0].imshow(gt_render_disp)
    axes[0, 0].set_title("Ground Truth Render")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_render_disp)
    axes[0, 1].set_title(f"RENI Predicted Render (Step {step})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gt_env_disp)
    axes[1, 0].set_title("Ground Truth Environment")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_env_disp)
    axes[1, 1].set_title("RENI Decoded Environment")
    axes[1, 1].axis("off")

    fig.suptitle(f"Step {step} | Loss: {loss:.6f}", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"step_{step:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    project_root = find_nerfstudio_project_root(Path(__file__))
    ckpt_path = Path("outputs/reni/reni_plus_plus_models/latent_dim_100")
    ckpt_step = 50000

    data_dir = project_root / "data" / "RENI_HDR"
    env_map_path = data_dir / "val" / "00007.exr"  # Use a validation env map
    normal_map_path = data_dir / "3d_models" / "normal_maps" / "bunny_normals.exr"

    output_dir = project_root / "outputs" / "inverse_rendering_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    image_size = 128
    illumination_width = 128
    illumination_height = 64
    num_steps = 500
    lr = 0.01
    save_every = 50
    specular_term = 0.2
    shininess = 500.0

    print("=" * 60)
    print("RENI++ Inverse Rendering Demo")
    print("=" * 60)

    # Load pretrained RENI decoder
    print("\n[1/5] Loading pretrained RENI decoder...")
    reni_field = load_pretrained_reni_decoder(ckpt_path, ckpt_step, device)

    # Load ground truth environment map
    print("\n[2/5] Loading environment map...")
    gt_envmap = load_environment_map(
        env_map_path, illumination_height, illumination_width
    )
    gt_envmap = gt_envmap.to(device)
    print(f"  Environment map shape: {gt_envmap.shape}")

    # Load normal map
    print("\n[3/5] Loading normal map...")
    normals, mask = load_normal_map(normal_map_path, image_size)
    normals = normals.to(device)
    mask = mask.to(device)
    print(f"  Normal map shape: {normals.shape}")

    # Setup illumination sampler and shader
    print("\n[4/5] Setting up rendering components...")
    sampler = EquirectangularSamplerConfig(
        width=illumination_width,
        apply_random_rotation=False,
        remove_lower_hemisphere=False,
    ).setup()

    shader = BlinnPhongShader()

    # Get light directions
    ray_samples = sampler.generate_direction_samples()
    light_directions = ray_samples.frustums.directions.to(device)
    light_directions = light_directions / torch.norm(
        light_directions, dim=-1, keepdim=True
    )

    # Get view directions
    view_directions = create_camera_and_view_directions(image_size, device)

    # Render ground truth image
    print("\n[5/5] Rendering ground truth...")
    with torch.no_grad():
        gt_render = render_with_environment(
            normals,
            mask,
            gt_envmap,
            light_directions,
            view_directions,
            shader,
            specular_term,
            shininess,
        )
    print(f"  GT render shape: {gt_render.shape}")

    # Initialize learnable latent codes (start from zero = prior mean)
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60)

    latent_codes = nn.Parameter(torch.zeros(reni_field.latent_dim, 3, device=device))
    scale = nn.Parameter(torch.zeros(1, device=device))  # log scale

    optimizer = torch.optim.Adam([latent_codes, scale], lr=lr)
    l2_loss_fn = nn.MSELoss()
    cosine_sim = nn.CosineSimilarity(dim=-1)

    # Optimization loop
    pbar = tqdm(range(num_steps + 1), desc="Optimizing")
    for step in pbar:
        optimizer.zero_grad()

        # Decode latents through RENI
        pred_envmap = decode_reni_to_envmap(
            reni_field, latent_codes, torch.exp(scale), sampler, device
        )

        # Render with predicted environment
        pred_render = render_with_environment(
            normals,
            mask,
            pred_envmap,
            light_directions,
            view_directions,
            shader,
            specular_term,
            shininess,
        )

        # Compute loss in sRGB space
        q = torch.quantile(gt_render.flatten(), 0.98)
        gt_srgb = linear_to_sRGB(gt_render, q=q, clamp=False)
        pred_srgb = linear_to_sRGB(pred_render, q=q, clamp=False)

        rgb_loss = l2_loss_fn(pred_srgb, gt_srgb)
        cos_loss = (
            1.0 - cosine_sim(pred_srgb.reshape(-1, 3), gt_srgb.reshape(-1, 3)).mean()
        )
        prior_loss = torch.mean(latent_codes**2)

        total_loss = 100.0 * rgb_loss + 1.0 * cos_loss + 0.001 * prior_loss

        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "rgb": f"{rgb_loss.item():.4f}",
                "scale": f"{torch.exp(scale).item():.2f}",
            }
        )

        # Save comparison images
        if step % save_every == 0:
            with torch.no_grad():
                save_comparison_image(
                    gt_render,
                    pred_render,
                    gt_envmap,
                    pred_envmap,
                    step,
                    output_dir,
                    total_loss.item(),
                )

    print("\n" + "=" * 60)
    print(f"Demo complete! Results saved to {output_dir}")
    print("=" * 60)

    # Print final stats
    print(f"\nFinal scale: {torch.exp(scale).item():.4f}")
    print(f"Final loss: {total_loss.item():.6f}")
    print(f"Latent code norm: {torch.norm(latent_codes).item():.4f}")


if __name__ == "__main__":
    main()
