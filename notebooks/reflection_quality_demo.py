"""
Reflection Quality Demo — RENI++ Specular Reflection Estimation

Demonstrates that RENI++ accurately estimates illumination for highly reflective
objects by optimizing only lighting (latent codes) with known materials and a
fixed viewpoint. Produces a publication-quality figure.

Usage:
    python notebooks/reflection_quality_demo.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebooks"))

from inverse_rendering_demo import (
    create_camera_and_view_directions,
    decode_reni_to_envmap,
    load_environment_map,
    load_normal_map,
    load_pretrained_reni_decoder,
    render_with_environment,
)
from reni.illumination_fields.reni_illumination_field import RENIField
from reni.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni.model_components.shaders import BlinnPhongShader
from reni.utils.colourspace import linear_to_sRGB


# ── Configuration ──────────────────────────────────────────────────────────────

OBJECTS = {
    "Sphere": {"file": "sphere_normals.exr", "specular": 0.95, "shininess": 3000},
    "Bunny": {"file": "bunny_normals.exr", "specular": 0.90, "shininess": 2000},
    "Teapot": {"file": "teapot_normals.exr", "specular": 0.90, "shininess": 2000},
    "Torus": {"file": "torus_normals.exr", "specular": 0.90, "shininess": 2000},
    "Knot": {"file": "knot_normals.exr", "specular": 0.90, "shininess": 2000},
}

ENV_MAPS = ["00002.exr", "00005.exr", "00007.exr"]

IMAGE_SIZE = 256
ILLUMINATION_WIDTH = 128
ILLUMINATION_HEIGHT = 64
NUM_STEPS = 1000
LR = 0.01
SAVE_INDIVIDUAL = True


# ── Optimization ───────────────────────────────────────────────────────────────


def optimize_lighting(
    reni_field: RENIField,
    normals: torch.Tensor,
    mask: torch.Tensor,
    gt_envmap: torch.Tensor,
    light_directions: torch.Tensor,
    view_directions: torch.Tensor,
    shader: BlinnPhongShader,
    sampler,
    device: torch.device,
    specular_term: float,
    shininess: float,
    num_steps: int = NUM_STEPS,
    lr: float = LR,
) -> dict:
    """Optimize RENI latent codes to match GT render of a reflective object."""

    # Render GT
    with torch.no_grad():
        gt_render = render_with_environment(
            normals, mask, gt_envmap, light_directions,
            view_directions, shader, specular_term, shininess,
        )

    # Initialize latent codes
    latent_codes = nn.Parameter(torch.zeros(reni_field.latent_dim, 3, device=device))
    scale = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([latent_codes, scale], lr=lr)
    l2_loss_fn = nn.MSELoss()
    cosine_sim = nn.CosineSimilarity(dim=-1)

    # Pre-compute GT sRGB with fixed quantile
    with torch.no_grad():
        q = torch.quantile(gt_render.flatten(), 0.98)
        gt_srgb = linear_to_sRGB(gt_render, q=q, clamp=False)

    pbar = tqdm(range(num_steps + 1), desc="  Optimizing", leave=False)
    for step in pbar:
        optimizer.zero_grad()

        pred_envmap = decode_reni_to_envmap(
            reni_field, latent_codes, torch.exp(scale), sampler, device
        )
        pred_render = render_with_environment(
            normals, mask, pred_envmap, light_directions,
            view_directions, shader, specular_term, shininess,
        )

        pred_srgb = linear_to_sRGB(pred_render, q=q, clamp=False)
        rgb_loss = l2_loss_fn(pred_srgb, gt_srgb)
        cos_loss = 1.0 - cosine_sim(pred_srgb.reshape(-1, 3), gt_srgb.reshape(-1, 3)).mean()
        prior_loss = torch.mean(latent_codes**2)
        total_loss = 100.0 * rgb_loss + 1.0 * cos_loss + 0.001 * prior_loss

        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", scale=f"{torch.exp(scale).item():.2f}")

    # Final outputs
    with torch.no_grad():
        pred_envmap = decode_reni_to_envmap(
            reni_field, latent_codes, torch.exp(scale), sampler, device
        )
        pred_render = render_with_environment(
            normals, mask, pred_envmap, light_directions,
            view_directions, shader, specular_term, shininess,
        )

    return {
        "gt_render": gt_render,
        "pred_render": pred_render,
        "gt_envmap": gt_envmap,
        "pred_envmap": pred_envmap,
        "q": q,
        "final_loss": total_loss.item(),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_psnr(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute PSNR over masked region in sRGB space."""
    mask_3d = mask.unsqueeze(-1).expand_as(gt)
    gt_masked = gt[mask_3d].reshape(-1, 3)
    pred_masked = pred[mask_3d].reshape(-1, 3)
    mse = torch.mean((gt_masked - pred_masked) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return -10.0 * np.log10(mse)


# ── Figure ─────────────────────────────────────────────────────────────────────


def create_publication_figure(results, obj_names, env_names, output_path):
    """Create a publication-quality comparison figure.

    Layout: rows = env maps, columns = [env map thumb, (GT, RENI++) per object]
    """
    n_envs = len(env_names)
    n_objs = len(obj_names)

    # Each object gets 2 columns (GT, Pred) + 1 for env map thumbnail
    n_cols = 1 + 2 * n_objs
    width_ratios = [1.8] + [1, 1] * n_objs

    fig, axes = plt.subplots(
        n_envs, n_cols,
        figsize=(2.0 * n_cols, 2.5 * n_envs + 0.5),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.02, "hspace": 0.15},
    )

    if n_envs == 1:
        axes = axes[np.newaxis, :]

    for i, env_name in enumerate(env_names):
        for j, obj_name in enumerate(obj_names):
            r = results[env_name][obj_name]
            q = r["q"]

            gt_disp = linear_to_sRGB(r["gt_render"], q=q).cpu().clamp(0, 1).numpy()
            pred_disp = linear_to_sRGB(r["pred_render"], q=q).cpu().clamp(0, 1).numpy()
            psnr = r["psnr"]

            # GT column
            col_gt = 1 + 2 * j
            axes[i, col_gt].imshow(gt_disp)
            axes[i, col_gt].axis("off")
            if i == 0:
                axes[i, col_gt].set_title("GT", fontsize=8)

            # Pred column
            col_pred = 2 + 2 * j
            axes[i, col_pred].imshow(pred_disp)
            axes[i, col_pred].axis("off")
            if i == 0:
                axes[i, col_pred].set_title("RENI++", fontsize=8)
            # PSNR annotation
            axes[i, col_pred].text(
                0.5, -0.02, f"{psnr:.1f} dB",
                transform=axes[i, col_pred].transAxes,
                ha="center", va="top", fontsize=6, color="black",
            )

        # Env map thumbnail
        env_disp = linear_to_sRGB(r["gt_envmap"], use_quantile=True).cpu().clamp(0, 1).numpy()
        axes[i, 0].imshow(env_disp)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Environment", fontsize=8)

    # Object name labels across top
    for j, obj_name in enumerate(obj_names):
        col_center = 1.5 + 2 * j
        fig.text(
            (col_center + 0.5) / (n_cols + 0.5), 0.98,
            obj_name, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved publication figure: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    ckpt_path = Path("outputs/reni/reni_plus_plus_models/latent_dim_100")
    ckpt_step = 50000
    data_dir = project_root / "data" / "RENI_HDR"
    normal_dir = data_dir / "3d_models" / "normal_maps"
    output_dir = project_root / "outputs" / "reflection_quality_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RENI decoder
    print("Loading RENI decoder...")
    reni_field = load_pretrained_reni_decoder(ckpt_path, ckpt_step, device)

    # Setup sampler and shader
    sampler = EquirectangularSamplerConfig(
        width=ILLUMINATION_WIDTH,
        apply_random_rotation=False,
        remove_lower_hemisphere=False,
    ).setup()
    shader = BlinnPhongShader()

    ray_samples = sampler.generate_direction_samples()
    light_directions = ray_samples.frustums.directions.to(device)
    light_directions = light_directions / torch.norm(light_directions, dim=-1, keepdim=True)

    view_directions = create_camera_and_view_directions(IMAGE_SIZE, device)

    # Preload normal maps
    print("Loading normal maps...")
    normals_cache = {}
    for obj_name, obj_cfg in OBJECTS.items():
        path = normal_dir / obj_cfg["file"]
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {obj_name}")
            print(f"  Run 'python notebooks/render_normal_maps.py' first to generate it.")
            continue
        normals, mask = load_normal_map(path, IMAGE_SIZE)
        normals_cache[obj_name] = (normals.to(device), mask.to(device))
        print(f"  Loaded {obj_name}: {normals.shape}")

    if not normals_cache:
        print("No normal maps found! Exiting.")
        return

    # Filter OBJECTS to only those with available normal maps
    active_objects = {k: v for k, v in OBJECTS.items() if k in normals_cache}
    obj_names = list(active_objects.keys())
    env_names = [e.replace(".exr", "") for e in ENV_MAPS]

    # Run optimization for each env map × object
    results = {}
    metrics = []

    for env_idx, env_file in enumerate(ENV_MAPS):
        env_name = env_file.replace(".exr", "")
        env_path = data_dir / "val" / env_file
        if not env_path.exists():
            print(f"WARNING: {env_path} not found — skipping")
            continue

        gt_envmap = load_environment_map(env_path, ILLUMINATION_HEIGHT, ILLUMINATION_WIDTH).to(device)
        print(f"\n{'='*60}")
        print(f"Environment map: {env_file} ({env_idx+1}/{len(ENV_MAPS)})")
        print(f"{'='*60}")

        results[env_name] = {}

        for obj_name, obj_cfg in active_objects.items():
            print(f"\n  Object: {obj_name} (specular={obj_cfg['specular']}, shininess={obj_cfg['shininess']})")
            normals, mask = normals_cache[obj_name]

            r = optimize_lighting(
                reni_field=reni_field,
                normals=normals,
                mask=mask,
                gt_envmap=gt_envmap,
                light_directions=light_directions,
                view_directions=view_directions,
                shader=shader,
                sampler=sampler,
                device=device,
                specular_term=obj_cfg["specular"],
                shininess=obj_cfg["shininess"],
            )

            # Compute PSNR
            with torch.no_grad():
                q = r["q"]
                gt_srgb = linear_to_sRGB(r["gt_render"], q=q).clamp(0, 1)
                pred_srgb = linear_to_sRGB(r["pred_render"], q=q).clamp(0, 1)
                psnr = compute_psnr(gt_srgb, pred_srgb, mask)

            r["psnr"] = psnr
            results[env_name][obj_name] = r
            print(f"  → PSNR: {psnr:.2f} dB, Loss: {r['final_loss']:.4f}")

            metrics.append({
                "env_map": env_name,
                "object": obj_name,
                "psnr": round(psnr, 2),
                "loss": round(r["final_loss"], 4),
            })

            # Save individual renders
            if SAVE_INDIVIDUAL:
                for label, img in [("gt", r["gt_render"]), ("pred", r["pred_render"])]:
                    disp = linear_to_sRGB(img, q=q).cpu().clamp(0, 1).numpy()
                    plt.imsave(
                        str(output_dir / f"{env_name}_{obj_name}_{label}.png"),
                        disp,
                    )

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Create publication figure
    active_env_names = [e for e in env_names if e in results]
    create_publication_figure(results, obj_names, active_env_names, output_dir / "reflection_quality_figure.pdf")
    create_publication_figure(results, obj_names, active_env_names, output_dir / "reflection_quality_figure.png")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for m in metrics:
        print(f"  {m['env_map']:>8s} | {m['object']:>8s} | PSNR: {m['psnr']:5.1f} dB")

    avg_psnr = np.mean([m["psnr"] for m in metrics])
    print(f"\n  Average PSNR: {avg_psnr:.2f} dB")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
