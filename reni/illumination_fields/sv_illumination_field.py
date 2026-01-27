# Copyright 2025 The University of York. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rotation-Equivariant Spherical Voronoi Illumination Field.

Combines RENI++ rotation equivariance with Spherical Voronoi explicit representation.
Uses a two-branch decoder:
- Geometry Branch (Equivariant): VN-MLP predicts Voronoi sites
- Appearance Branch (Invariant): VN-Invariant → MLP predicts colors and temperatures
"""

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from jaxtyping import Float

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.base_config import InstantiateConfig

from reni.field_components.field_heads import RENIFieldHeadNames
from reni.field_components.vn_layers import VNLinear, VNReLU, VNInvariant, VNLayerNorm
from reni.field_components.sv_primitives import soft_spherical_voronoi_batched, fibonacci_sphere_sites
from reni.illumination_fields.base_spherical_field import BaseRENIField, BaseRENIFieldConfig


@dataclass
class SphericalVoronoiFieldConfig(BaseRENIFieldConfig):
    """Configuration for Spherical Voronoi Field instantiation."""
    
    _target: Type = field(default_factory=lambda: SphericalVoronoiField)
    """Target class to instantiate"""
    
    # Latent code configuration
    latent_dim: int = 36
    """Number of latent vectors (N in Z ∈ ℝ^{N×3})"""
    
    # SV representation
    num_sites: int = 100
    """Number of Voronoi sites"""
    num_channels: int = 3
    """Number of output channels (RGB)"""
    
    # Decoder architecture
    geometry_hidden_dim: int = 64
    """Hidden dimension for geometry branch"""
    geometry_num_layers: int = 2
    """Number of layers in geometry branch"""
    appearance_hidden_dim: int = 128
    """Hidden dimension for appearance branch"""
    appearance_num_layers: int = 3
    """Number of layers in appearance branch"""
    
    # Equivariance configuration
    equivariance: Literal["SO2", "SO3"] = "SO2"
    """Type of rotation equivariance"""
    axis_of_invariance: Literal["x", "y", "z"] = "y"
    """Axis for SO(2) invariance (typically Y for gravity)"""
    
    # Training configuration
    init_temperature: float = 10.0
    """Initial temperature value for sites"""
    trainable_scale: bool = False
    """Whether to learn a global intensity scale"""


class VNGeometryBranch(nn.Module):
    """
    Equivariant geometry branch: predicts Voronoi sites from latent code.
    
    Uses VN-MLP to transform latent vectors Z ∈ ℝ^{N×3} to sites S ∈ ℝ^{K×3}.
    Property: Φ_geo(RZ) = R · Φ_geo(Z)
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_sites: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_sites = num_sites
        
        # Build VN-MLP
        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers.extend([
                VNLinear(in_dim, hidden_dim),
                VNReLU(hidden_dim),
                VNLayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
        
        # Final projection to sites
        layers.append(VNLinear(in_dim, num_sites))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent code [batch, latent_dim, 3]
            
        Returns:
            sites: Voronoi sites [batch, num_sites, 3] (unit normalized)
        """
        sites = self.net(z)  # [batch, num_sites, 3]
        # Normalize to unit sphere
        sites = F.normalize(sites, dim=-1)
        return sites


class VNAppearanceBranch(nn.Module):
    """
    Invariant appearance branch: predicts colors and temperatures from latent code.
    
    Uses Gram matrix (inner products Z^T @ Z) to extract rotation-invariant features.
    Property: Φ_app(RZ) = Φ_app(Z) because (RZ)^T @ (RZ) = Z^T @ R^T @ R @ Z = Z^T @ Z
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_sites: int,
        num_channels: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dim_coor: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_sites = num_sites
        self.num_channels = num_channels
        self.dim_coor = dim_coor
        
        # Invariant feature dimension: 
        # - Gram matrix: latent_dim * latent_dim (symmetric, so could use upper triangle)
        # - For efficiency, we use the full Gram matrix flattened
        # - Also include vector norms for additional features
        invariant_dim = latent_dim * latent_dim + latent_dim
        
        # MLP for colors
        color_layers = []
        in_dim = invariant_dim
        for i in range(num_layers - 1):
            color_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        color_layers.append(nn.Linear(in_dim, num_sites * num_channels))
        self.color_net = nn.Sequential(*color_layers)
        
        # MLP for temperatures (log-parameterized)
        temp_layers = []
        in_dim = invariant_dim
        for i in range(num_layers - 1):
            temp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        temp_layers.append(nn.Linear(in_dim, num_sites))
        self.temp_net = nn.Sequential(*temp_layers)
        
        # Initialize temperature network to output reasonable values
        with torch.no_grad():
            self.temp_net[-1].bias.fill_(2.3)  # exp(2.3) ≈ 10
    
    def compute_invariant_features(self, z: Tensor) -> Tensor:
        """
        Compute rotation-invariant features from latent code.
        
        Args:
            z: Latent code [batch, latent_dim, 3]
            
        Returns:
            Invariant features [batch, invariant_dim]
        """
        # Gram matrix: Z^T @ Z = inner products between all pairs of latent vectors
        # This is invariant because (RZ)^T @ (RZ) = Z^T @ R^T @ R @ Z = Z^T @ Z
        gram = torch.einsum('bid,bjd->bij', z, z)  # [batch, latent_dim, latent_dim]
        gram_flat = gram.flatten(start_dim=1)  # [batch, latent_dim^2]
        
        # Also compute vector norms as additional invariant features
        norms = torch.norm(z, dim=-1)  # [batch, latent_dim]
        
        # Concatenate all invariant features
        invariant = torch.cat([gram_flat, norms], dim=-1)
        
        return invariant
    
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            z: Latent code [batch, latent_dim, 3]
            
        Returns:
            colors: Per-site colors [batch, num_sites, num_channels]
            temperatures: Per-site temperatures [batch, num_sites] (positive)
        """
        # Extract invariant features
        invariant = self.compute_invariant_features(z)
        
        # Predict colors (can be any value for HDR)
        colors = self.color_net(invariant)  # [batch, num_sites * num_channels]
        colors = colors.view(-1, self.num_sites, self.num_channels)
        
        # Predict temperatures (positive via exp)
        log_temps = self.temp_net(invariant)  # [batch, num_sites]
        temperatures = torch.exp(log_temps)
        
        return colors, temperatures


class SphericalVoronoiField(BaseRENIField):
    """
    Rotation-Equivariant Spherical Voronoi Illumination Field.
    
    Combines:
    - Vector Neuron equivariance for site prediction
    - VN-Invariant for color/temperature prediction
    - Explicit SV rendering for sharp features
    
    The rendering formula f(ω; S, C, τ) is naturally equivariant when sites
    are predicted by an equivariant network:
        f(ω; RZ) = f(R⁻¹ω; Z)
    """
    
    def __init__(
        self,
        config: SphericalVoronoiFieldConfig,
        num_train_data: Optional[int] = None,
        num_eval_data: Optional[int] = None,
        normalisations: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            config=config,
            num_train_data=num_train_data,
            num_eval_data=num_eval_data,
            normalisations=normalisations,
        )
        
        self.latent_dim = config.latent_dim
        self.num_sites = config.num_sites
        self.num_channels = config.num_channels
        
        # Coordinate dimension (3 for SO3, could be 2 for SO2)
        self.dim_coor = 3
        
        # Two-branch decoder
        self.geometry_branch = VNGeometryBranch(
            latent_dim=config.latent_dim,
            num_sites=config.num_sites,
            hidden_dim=config.geometry_hidden_dim,
            num_layers=config.geometry_num_layers,
        )
        
        self.appearance_branch = VNAppearanceBranch(
            latent_dim=config.latent_dim,
            num_sites=config.num_sites,
            num_channels=config.num_channels,
            hidden_dim=config.appearance_hidden_dim,
            num_layers=config.appearance_num_layers,
            dim_coor=self.dim_coor,
        )
        
        # Learnable latent codes for autodecoding
        if num_train_data is not None and num_train_data > 0:
            self.train_latents = nn.Parameter(
                torch.randn(num_train_data, config.latent_dim, 3) * 0.1
            )
        else:
            self.register_parameter('train_latents', None)
            
        if num_eval_data is not None and num_eval_data > 0:
            self.eval_latents = nn.Parameter(
                torch.randn(num_eval_data, config.latent_dim, 3) * 0.1
            )
        else:
            self.register_parameter('eval_latents', None)
        
        # Optional trainable scale
        if config.trainable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
    
    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to fix decoder weights."""
        prev_state = self.fixed_decoder
        self.fixed_decoder = True
        
        # Store original requires_grad states
        orig_geo = {n: p.requires_grad for n, p in self.geometry_branch.named_parameters()}
        orig_app = {n: p.requires_grad for n, p in self.appearance_branch.named_parameters()}
        
        # Freeze decoder
        for p in self.geometry_branch.parameters():
            p.requires_grad = False
        for p in self.appearance_branch.parameters():
            p.requires_grad = False
        
        try:
            yield
        finally:
            # Restore states
            self.fixed_decoder = prev_state
            for n, p in self.geometry_branch.named_parameters():
                p.requires_grad = orig_geo[n]
            for n, p in self.appearance_branch.named_parameters():
                p.requires_grad = orig_app[n]
    
    def decode(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decode latent code to SV parameters.
        
        Args:
            z: Latent code [batch, latent_dim, 3]
            
        Returns:
            sites: [batch, num_sites, 3]
            colors: [batch, num_sites, num_channels]
            temperatures: [batch, num_sites]
        """
        sites = self.geometry_branch(z)
        colors, temperatures = self.appearance_branch(z)
        return sites, colors, temperatures
    
    def render(
        self,
        directions: Tensor,
        sites: Tensor,
        colors: Tensor,
        temperatures: Tensor,
    ) -> Tensor:
        """
        Render environment map values at given directions.
        
        Args:
            directions: Query directions [batch, num_rays, 3]
            sites: Voronoi sites [batch, num_sites, 3]
            colors: Per-site colors [batch, num_sites, num_channels]
            temperatures: Per-site temperatures [batch, num_sites]
            
        Returns:
            rgb: Rendered values [batch, num_rays, num_channels]
        """
        return soft_spherical_voronoi_batched(
            directions=directions,
            sites=sites,
            colors=colors,
            temperatures=temperatures,
            normalize_sites=False,  # Already normalized by geometry branch
        )
    
    def get_latent(self, idx: Tensor) -> Tensor:
        """Get latent code for given indices."""
        if self.training and not self.fixed_decoder:
            return self.train_latents[idx]
        else:
            return self.eval_latents[idx]
    
    def reset_eval_latents(self):
        """Reset eval latents to random initialization."""
        if self.eval_latents is not None:
            self.eval_latents.data = torch.randn_like(self.eval_latents) * 0.1
    
    def get_outputs(
        self,
        ray_samples: RaySamples,
        rotation: Optional[Tensor] = None,
        latent_codes: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Dict[RENIFieldHeadNames, Tensor]:
        """
        Returns the outputs of the field.
        
        Args:
            ray_samples: [num_rays] with camera_indices
            rotation: [3, 3] rotation matrix (applied to latent code)
            latent_codes: [batch, latent_dim, 3] override latent codes
            scale: [batch] intensity scale
            
        Returns:
            Dict with RGB values
        """
        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays]
        directions = ray_samples.frustums.directions  # [num_rays, 3]
        
        # Get unique camera indices and their inverse mapping
        unique_indices, inverse = torch.unique(camera_indices, return_inverse=True)
        batch_size = unique_indices.shape[0]
        
        # Get latent codes
        if latent_codes is not None:
            z = latent_codes
        else:
            z = self.get_latent(unique_indices)  # [batch, latent_dim, 3]
        
        # Apply rotation to latent code (equivariant operation)
        if rotation is not None:
            # rotation: [3, 3] -> apply to each latent vector
            z = torch.einsum('ij,...j->...i', rotation, z)
        
        # Decode to SV parameters
        sites, colors, temperatures = self.decode(z)
        
        # Expand directions by camera index
        # Group rays by camera
        directions_batched = directions.new_zeros(batch_size, directions.shape[0], 3)
        for b, idx in enumerate(unique_indices):
            mask = camera_indices == idx
            ray_dirs = directions[mask]
            directions_batched[b, :ray_dirs.shape[0]] = ray_dirs
        
        # Render
        rgb_batched = self.render(directions_batched, sites, colors, temperatures)
        
        # Ungroup to original ray order
        rgb = rgb_batched.new_zeros(directions.shape[0], self.num_channels)
        for b, idx in enumerate(unique_indices):
            mask = camera_indices == idx
            rgb[mask] = rgb_batched[b, :mask.sum()]
        
        # Apply scale
        if scale is not None:
            rgb = rgb * scale
        else:
            rgb = rgb * self.scale
        
        # Unnormalize if needed
        rgb = self.unnormalise(rgb)
        
        return {RENIFieldHeadNames.RGB: rgb}
    
    def forward(
        self,
        ray_samples: RaySamples,
        rotation: Optional[Tensor] = None,
        latent_codes: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Dict[RENIFieldHeadNames, Tensor]:
        """Evaluates field for given ray samples."""
        return self.get_outputs(
            ray_samples=ray_samples,
            rotation=rotation,
            latent_codes=latent_codes,
            scale=scale,
        )
