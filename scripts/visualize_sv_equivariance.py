#!/usr/bin/env python3
"""
Visualize rotation equivariance of the SV illumination field.

Creates a figure showing:
1. Original environment map from latent Z
2. Environment map from rotated latent RZ
3. Original environment map rotated in image space by R

The second and third should be identical, proving f(ω; RZ) = f(R⁻¹ω; Z).
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reni.illumination_fields.sv_illumination_field import (
    SphericalVoronoiField,
    SphericalVoronoiFieldConfig,
)


def rotation_matrix_y(angle: float) -> torch.Tensor:
    """Rotation matrix around Y-axis."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=torch.float32)


def render_envmap(field, z, height=128, width=256):
    """Render environment map from latent code."""
    device = z.device
    
    # Generate equirectangular coordinates
    theta = torch.linspace(0, math.pi, height, device=device)
    phi = torch.linspace(-math.pi, math.pi, width, device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian (Y-up)
    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(theta)
    z_coord = torch.sin(theta) * torch.cos(phi)
    directions = torch.stack([x, y, z_coord], dim=-1)  # [H, W, 3]
    
    # Decode and render
    sites, colors, temps = field.decode(z)
    
    # Render
    dirs_batch = directions.unsqueeze(0).reshape(1, -1, 3)  # [1, H*W, 3]
    rgb = field.render(dirs_batch, sites, colors, temps)
    rgb = rgb.reshape(height, width, 3)
    
    return rgb


def render_envmap_with_rotated_dirs(field, z, R_inv, height=128, width=256):
    """Render environment map querying at counter-rotated directions."""
    device = z.device
    
    # Generate equirectangular coordinates
    theta = torch.linspace(0, math.pi, height, device=device)
    phi = torch.linspace(-math.pi, math.pi, width, device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian (Y-up)
    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(theta)
    z_coord = torch.sin(theta) * torch.cos(phi)
    directions = torch.stack([x, y, z_coord], dim=-1)  # [H, W, 3]
    
    # Counter-rotate directions: query at R⁻¹ω instead of ω
    directions_rotated = torch.einsum('ij,...j->...i', R_inv, directions)
    
    # Decode and render
    sites, colors, temps = field.decode(z)
    
    # Render at rotated directions
    dirs_batch = directions_rotated.unsqueeze(0).reshape(1, -1, 3)
    rgb = field.render(dirs_batch, sites, colors, temps)
    rgb = rgb.reshape(height, width, 3)
    
    return rgb


def tonemap(x):
    """Simple Reinhard tonemapping."""
    x = x.detach().cpu().numpy()
    x = np.clip(x, 0, None)  # Ensure positive
    return np.clip(x / (1 + x), 0, 1)


def main():
    output_dir = 'outputs/sv_equivariance'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create field
    config = SphericalVoronoiFieldConfig(
        latent_dim=16,
        num_sites=64,
        geometry_hidden_dim=32,
        geometry_num_layers=2,
        appearance_hidden_dim=64,
        appearance_num_layers=2,
    )
    field = SphericalVoronoiField(config, num_train_data=1, num_eval_data=1)
    field.eval()
    
    # Random latent code with larger values for interesting patterns
    z = torch.randn(1, 16, 3) * 0.5
    
    # Rotation angle (90 degrees around Y-axis)
    angle = math.pi / 2
    R = rotation_matrix_y(angle)
    
    with torch.no_grad():
        # 1. Original environment map
        envmap_original = render_envmap(field, z)
        
        # 2. Environment map from rotated latent: f(ω; R·Z)
        z_rotated = torch.einsum('ij,...j->...i', R, z)
        envmap_from_rotated_latent = render_envmap(field, z_rotated)
        
        # 3. Original latent, query at counter-rotated directions: f(R⁻¹ω; Z)
        R_inv = R.T  # For orthogonal matrices, inverse = transpose
        envmap_counter_rotated = render_envmap_with_rotated_dirs(field, z, R_inv)
        
        # 4. Difference to show they match exactly
        diff = torch.abs(envmap_from_rotated_latent - envmap_counter_rotated)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    axes[0, 0].imshow(tonemap(envmap_original))
    axes[0, 0].set_title('Original: f(ω; Z)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tonemap(envmap_from_rotated_latent))
    axes[0, 1].set_title(f'From Rotated Latent: f(ω; R·Z)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(tonemap(envmap_counter_rotated))
    axes[1, 0].set_title(f'Counter-rotated Dirs: f(R⁻¹ω; Z)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Difference map (should be near zero)
    diff_vis = diff.cpu().numpy() * 10  # Amplify for visibility
    axes[1, 1].imshow(np.clip(diff_vis, 0, 1))
    axes[1, 1].set_title(f'Difference (10x amplified)\nMax diff: {diff.max():.2e}', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle('Rotation Equivariance Proof: f(ω; R·Z) = f(R⁻¹ω; Z)\n90° rotation around Y-axis (azimuth)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rotation_equivariance_proof.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved equivariance visualization to: {output_path}")
    print(f"Maximum difference: {diff.max():.2e}")
    print(f"Mean difference: {diff.mean():.2e}")
    
    if diff.max() < 1e-4:
        print("✓ Rotation equivariance verified!")
    else:
        print("✗ Difference larger than expected")
    
    return output_path


if __name__ == '__main__':
    main()
