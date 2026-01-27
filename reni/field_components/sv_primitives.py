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
Spherical Voronoi primitives for differentiable environment map representation.

Based on: "Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere"
https://arxiv.org/abs/2512.14180
"""

from typing import Optional, Tuple, Literal
import torch
from torch import Tensor
import torch.nn.functional as F
import math


# ============================================================================
# HDR/LDR Conversion Utilities
# ============================================================================

def hdr_to_ldr(
    img: Tensor,
    exposure: float = 1.0,
    gamma: float = 2.2,
) -> Tensor:
    """
    Convert HDR image to LDR using filmic tonemapping and gamma correction.

    Args:
        img: HDR image tensor, any shape with values >= 0
        exposure: Exposure multiplier before tonemapping
        gamma: Gamma correction value (typically 2.2 for sRGB)

    Returns:
        LDR image tensor with values in [0, 1]
    """
    # Filmic tonemapping: 1 - exp(-x * exposure)
    img = 1.0 - torch.exp(-img * exposure)
    # Gamma correction
    img = torch.pow(torch.clamp(img, 0, 1), 1.0 / gamma)
    return img


def ldr_to_hdr(
    img: Tensor,
    exposure: float = 1.0,
    gamma: float = 2.2,
) -> Tensor:
    """
    Approximate inverse of hdr_to_ldr (inverse tonemapping).

    Args:
        img: LDR image tensor with values in [0, 1]
        gamma: Gamma value used in forward conversion
        exposure: Exposure value used in forward conversion

    Returns:
        Approximate HDR image tensor
    """
    # Inverse gamma
    img = torch.pow(torch.clamp(img, 1e-6, 1.0 - 1e-6), gamma)
    # Inverse filmic: -log(1 - x) / exposure
    img = -torch.log(torch.clamp(1.0 - img, min=1e-6)) / exposure
    return img


def gaussian_blur_2d(
    img: Tensor,
    sigma: float,
    kernel_size: Optional[int] = None,
) -> Tensor:
    """
    Apply Gaussian blur to a 2D image.

    Args:
        img: Image tensor of shape [H, W, C]
        sigma: Standard deviation of the Gaussian kernel
        kernel_size: Size of the kernel (auto-computed if None)

    Returns:
        Blurred image tensor of shape [H, W, C]
    """
    if sigma <= 0:
        return img

    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Reshape for conv2d: [H, W, C] -> [1, C, H, W]
    img_4d = img.permute(2, 0, 1).unsqueeze(0)
    C = img_4d.shape[1]

    # Separable convolution (horizontal then vertical)
    kernel_h = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)

    pad_h = kernel_size // 2
    pad_v = kernel_size // 2

    # Apply horizontal blur
    img_4d = F.pad(img_4d, (pad_h, pad_h, 0, 0), mode='replicate')
    img_4d = F.conv2d(img_4d, kernel_h, groups=C)

    # Apply vertical blur
    img_4d = F.pad(img_4d, (0, 0, pad_v, pad_v), mode='replicate')
    img_4d = F.conv2d(img_4d, kernel_v, groups=C)

    # Reshape back: [1, C, H, W] -> [H, W, C]
    return img_4d.squeeze(0).permute(1, 2, 0)


def compute_solid_angle_weights(height: int, width: int, device: torch.device = None) -> Tensor:
    """
    Compute solid angle weights for equirectangular map pixels.

    Each pixel's weight is proportional to sin(theta) where theta is the
    polar angle from the pole. This accounts for the area distortion in
    equirectangular projection.

    Args:
        height: Image height
        width: Image width
        device: Target device

    Returns:
        Weight tensor of shape [H, W] normalized to mean of 1.0
    """
    # v goes from 0 to 1 (top to bottom)
    v = torch.linspace(0.5 / height, 1 - 0.5 / height, height, device=device)
    # theta (polar angle) goes from 0 to pi
    theta = v * math.pi
    # Weight by sin(theta)
    weights = torch.sin(theta)
    # Expand to [H, W]
    weights = weights.unsqueeze(1).expand(height, width)
    # Normalize to mean of 1.0
    weights = weights / weights.mean()
    return weights


def fibonacci_sphere_sites(num_sites: int, device: torch.device = None) -> Tensor:
    """
    Generate uniformly distributed points on a sphere using Fibonacci spiral.
    
    Args:
        num_sites: Number of points to generate
        device: Target device for the tensor
        
    Returns:
        Tensor of shape [num_sites, 3] with unit-norm direction vectors
    """
    indices = torch.arange(num_sites, dtype=torch.float32, device=device)
    
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    
    # Fibonacci spiral
    theta = 2 * math.pi * indices / phi  # Azimuth angle
    z = 1 - (2 * indices + 1) / num_sites  # Height (from -1 to 1)
    radius = torch.sqrt(1 - z * z)
    
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    
    sites = torch.stack([x, y, z], dim=-1)
    return sites


def soft_spherical_voronoi(
    directions: Tensor,
    sites: Tensor,
    colors: Tensor,
    temperatures: Tensor,
    normalize_sites: bool = True,
) -> Tensor:
    """
    Compute soft Spherical Voronoi representation for given query directions.
    
    The formula is:
        f(ω) = Σ_k w_k(ω) * c_k
    where:
        w_k(ω) = exp(τ_k * (s_k · ω)) / Σ_j exp(τ_j * (s_j · ω))
    
    Args:
        directions: Query directions, shape [..., 3], should be unit-normalized
        sites: Voronoi sites on unit sphere, shape [K, 3]
        colors: Per-site colors/values, shape [K, C] where C is typically 3 (RGB)
        temperatures: Per-site sharpness parameters, shape [K]
        normalize_sites: Whether to normalize sites to unit sphere
        
    Returns:
        Interpolated values at query directions, shape [..., C]
    """
    # Ensure sites are on unit sphere
    if normalize_sites:
        sites = F.normalize(sites, dim=-1)
    
    # Compute dot products between all directions and all sites
    # directions: [..., 3], sites: [K, 3]
    # result: [..., K]
    dot_products = torch.einsum('...d,kd->...k', directions, sites)
    
    # Scale by temperatures (log-space for numerical stability)
    # temperatures: [K] -> [..., K]
    scaled_dots = dot_products * temperatures.unsqueeze(0).expand_as(dot_products)
    
    # Softmax over sites dimension to get weights
    # Use log-sum-exp trick for numerical stability with large temperatures
    weights = F.softmax(scaled_dots, dim=-1)  # [..., K]
    
    # Weighted sum of colors
    # weights: [..., K], colors: [K, C]
    # result: [..., C]
    output = torch.einsum('...k,kc->...c', weights, colors)
    
    return output


def soft_spherical_voronoi_batched(
    directions: Tensor,
    sites: Tensor,
    colors: Tensor,
    temperatures: Tensor,
    normalize_sites: bool = True,
) -> Tensor:
    """
    Batched version of soft_spherical_voronoi where each sample has its own SV parameters.
    
    Args:
        directions: Query directions, shape [B, N, 3] (B batches, N rays per batch)
        sites: Voronoi sites, shape [B, K, 3]
        colors: Per-site colors, shape [B, K, C]
        temperatures: Per-site sharpness, shape [B, K]
        normalize_sites: Whether to normalize sites
        
    Returns:
        Interpolated values, shape [B, N, C]
    """
    if normalize_sites:
        sites = F.normalize(sites, dim=-1)
    
    # directions: [B, N, 3], sites: [B, K, 3]
    # dot_products: [B, N, K]
    dot_products = torch.einsum('bnd,bkd->bnk', directions, sites)
    
    # Scale by temperatures: [B, K] -> [B, 1, K]
    scaled_dots = dot_products * temperatures.unsqueeze(1)
    
    # Softmax weights
    weights = F.softmax(scaled_dots, dim=-1)  # [B, N, K]
    
    # Weighted sum: weights [B, N, K], colors [B, K, C] -> [B, N, C]
    output = torch.einsum('bnk,bkc->bnc', weights, colors)
    
    return output


def softplus_inv(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    return math.log(math.exp(x) - 1.0)


class SphericalVoronoiRepresentation(torch.nn.Module):
    """
    Learnable Spherical Voronoi representation for environment maps.

    Parameters:
        num_sites: Number of Voronoi sites (default: 100)
        num_channels: Number of color channels (default: 3 for RGB)
        init_temperature: Initial temperature/beta value (default: 10.0)
        shared_temperature: If True, use single shared beta (like original paper).
                           If False, use per-site temperatures.
        temperature_min: Minimum temperature value (only used if shared_temperature=True)
        temperature_max: Maximum temperature value (only used if shared_temperature=True)
        color_activation: Activation for colors: "none", "relu" (LDR), or "softplus" (HDR)
        device: Target device
    """

    def __init__(
        self,
        num_sites: int = 100,
        num_channels: int = 3,
        init_temperature: float = 10.0,
        shared_temperature: bool = False,
        temperature_min: float = 0.5,
        temperature_max: float = 512.0,
        color_activation: Literal["none", "relu", "softplus"] = "none",
        device: torch.device = None,
    ):
        super().__init__()
        self.num_sites = num_sites
        self.num_channels = num_channels
        self.shared_temperature = shared_temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.color_activation = color_activation

        # Initialize sites with Fibonacci spiral
        sites = fibonacci_sphere_sites(num_sites, device=device)
        self.sites = torch.nn.Parameter(sites)

        # Initialize colors based on activation
        if color_activation == "relu":
            # For LDR with ReLU: initialize around 0.5
            colors = torch.ones(num_sites, num_channels, device=device) * 0.5
        elif color_activation == "softplus":
            # For HDR with softplus: initialize in pre-activation space
            # softplus(x) ≈ x for x > 5, so init around 0.5 in output space
            # softplus^-1(0.5) ≈ log(exp(0.5) - 1) ≈ -0.47
            colors = torch.zeros(num_sites, num_channels, device=device)
        else:
            # No activation: small random values
            colors = torch.randn(num_sites, num_channels, device=device) * 0.1
        self.colors = torch.nn.Parameter(colors)

        # Initialize temperatures
        if shared_temperature:
            # Single shared beta, parameterized via softplus
            init_clamped = float(max(temperature_min, min(temperature_max, init_temperature)))
            beta_raw_init = softplus_inv(init_clamped)
            self._beta_raw = torch.nn.Parameter(
                torch.tensor([[[beta_raw_init]]], device=device)
            )
        else:
            # Per-site temperatures in log-space
            log_temp = torch.full((num_sites,), math.log(init_temperature), device=device)
            self.log_temperatures = torch.nn.Parameter(log_temp)

    @property
    def temperature_values(self) -> Tensor:
        """Get actual temperature values (handling parameterization)."""
        if self.shared_temperature:
            beta = F.softplus(self._beta_raw).squeeze()
            beta = torch.clamp(beta, self.temperature_min, self.temperature_max)
            # Expand to [num_sites] for compatibility
            return beta.expand(self.num_sites)
        else:
            return torch.exp(self.log_temperatures)

    def get_beta(self) -> Tensor:
        """Get the beta/temperature value (scalar if shared, tensor if per-site)."""
        if self.shared_temperature:
            beta = F.softplus(self._beta_raw)
            return torch.clamp(beta, self.temperature_min, self.temperature_max).squeeze()
        else:
            return torch.exp(self.log_temperatures)

    def get_normalized_sites(self) -> Tensor:
        """Get sites normalized to unit sphere."""
        return F.normalize(self.sites, dim=-1)

    def get_colors(self) -> Tensor:
        """Get colors, applying activation if specified."""
        if self.color_activation == "relu":
            return F.relu(self.colors)
        elif self.color_activation == "softplus":
            return F.softplus(self.colors)
        return self.colors

    def forward(self, directions: Tensor) -> Tensor:
        """
        Evaluate the SV representation at given directions.

        Args:
            directions: Query directions, shape [..., 3]

        Returns:
            RGB values at those directions, shape [..., C]
        """
        return soft_spherical_voronoi(
            directions=directions,
            sites=self.sites,
            colors=self.get_colors(),
            temperatures=self.temperature_values,
            normalize_sites=True,
        )

    def render_equirectangular(
        self,
        height: int = 256,
        width: int = 512,
    ) -> Tensor:
        """
        Render the SV representation as an equirectangular environment map.

        Args:
            height: Image height
            width: Image width

        Returns:
            Environment map tensor of shape [height, width, C]
        """
        device = self.sites.device

        # Generate equirectangular coordinates
        theta = torch.linspace(0, math.pi, height, device=device)      # Elevation [0, π]
        phi = torch.linspace(-math.pi, math.pi, width, device=device)  # Azimuth [-π, π]

        theta, phi = torch.meshgrid(theta, phi, indexing='ij')

        # Convert to Cartesian directions (Y-up convention)
        x = torch.sin(theta) * torch.sin(phi)
        y = torch.cos(theta)
        z = torch.sin(theta) * torch.cos(phi)

        directions = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

        return self.forward(directions)


def fit_sv_to_envmap(
    target_envmap: Tensor,
    num_sites: int = 100,
    num_iterations: int = 1000,
    lr: float = 0.01,
    mode: Literal["hdr", "ldr"] = "hdr",
    # LDR-specific options
    exposure: float = 1.0,
    gamma: float = 2.2,
    blur_sigma: float = 2.5,
    # HDR-specific options
    log_loss: bool = True,
    # Common options
    use_solid_angle_weights: bool = True,
    batch_size: Optional[int] = None,
    lr_sites: Optional[float] = None,
    lr_colors: Optional[float] = None,
    lr_temperature: Optional[float] = None,
    init_temperature: Optional[float] = None,
    shared_temperature: Optional[bool] = None,
    verbose: bool = True,
) -> Tuple[SphericalVoronoiRepresentation, dict]:
    """
    Fit a Spherical Voronoi representation to a target environment map.

    Args:
        target_envmap: Target environment map, shape [H, W, 3].
                       For mode="hdr": expects HDR values (linear, unbounded).
                       For mode="ldr": expects HDR values which will be converted to LDR.
        num_sites: Number of Voronoi sites
        num_iterations: Optimization iterations
        lr: Base learning rate (used if specific lr_* not provided)
        mode: "hdr" for HDR fitting with log-loss, "ldr" for LDR fitting matching
              the original Spherical Voronoi paper methodology
        exposure: Exposure for HDR->LDR conversion (only used if mode="ldr")
        gamma: Gamma for HDR->LDR conversion (only used if mode="ldr")
        blur_sigma: Gaussian blur sigma for target (only used if mode="ldr", 0 to disable)
        log_loss: Use log-space MSE for HDR mode (ignored in LDR mode)
        use_solid_angle_weights: Weight loss by solid angle (sin(theta))
        batch_size: If provided, use stochastic optimization with this batch size.
                    If None, use full-image optimization.
        lr_sites: Learning rate for site positions (default: 1e-4)
        lr_colors: Learning rate for colors (default: 1e-3)
        lr_temperature: Learning rate for temperature/beta (default: 1e-3)
        init_temperature: Initial temperature/beta value (default: 256 for LDR, 128 for HDR)
        shared_temperature: Use single shared beta vs per-site temperatures.
                           Default: True for LDR, True for HDR (can override)
        verbose: Print progress

    Returns:
        Tuple of (fitted SV representation, metrics dict)
    """
    device = target_envmap.device
    height, width = target_envmap.shape[:2]

    # Set mode-specific defaults
    if mode == "ldr":
        # LDR mode: match original paper settings
        lr_sites = lr_sites if lr_sites is not None else 1e-4
        lr_colors = lr_colors if lr_colors is not None else 1e-3
        lr_temperature = lr_temperature if lr_temperature is not None else 1e-3
        shared_temperature = shared_temperature if shared_temperature is not None else True
        color_activation = "relu"
        use_log_loss = False
        init_temperature = init_temperature if init_temperature is not None else 256.0

        # Preprocess target: HDR -> LDR with optional blur
        target_processed = hdr_to_ldr(target_envmap, exposure=exposure, gamma=gamma)
        if blur_sigma > 0:
            target_processed = gaussian_blur_2d(target_processed, sigma=blur_sigma)
    else:
        # HDR mode: improved settings
        lr_sites = lr_sites if lr_sites is not None else 1e-4
        lr_colors = lr_colors if lr_colors is not None else 1e-3
        lr_temperature = lr_temperature if lr_temperature is not None else 1e-3
        shared_temperature = shared_temperature if shared_temperature is not None else True
        color_activation = "softplus"  # Non-negative HDR values
        use_log_loss = log_loss
        init_temperature = init_temperature if init_temperature is not None else 128.0
        target_processed = target_envmap

    # Initialize SV representation
    sv = SphericalVoronoiRepresentation(
        num_sites=num_sites,
        num_channels=3,
        init_temperature=init_temperature,
        shared_temperature=shared_temperature,
        color_activation=color_activation,
        device=device,
    )

    # Smart initialization: sample colors from target at site locations
    with torch.no_grad():
        sites = sv.get_normalized_sites()  # [K, 3]

        # Convert sites (Cartesian) to equirectangular coordinates
        # Y-up convention: y = cos(theta), x = sin(theta)*sin(phi), z = sin(theta)*cos(phi)
        theta = torch.acos(torch.clamp(sites[:, 1], -1, 1))  # [0, pi] elevation
        phi = torch.atan2(sites[:, 0], sites[:, 2])  # [-pi, pi] azimuth

        # Convert to pixel coordinates
        u = (phi + math.pi) / (2 * math.pi) * (width - 1)  # [0, W-1]
        v = theta / math.pi * (height - 1)  # [0, H-1]

        # Bilinear sampling using grid_sample
        # grid_sample expects grid in [-1, 1] range
        grid_x = 2 * u / (width - 1) - 1  # [-1, 1]
        grid_y = 2 * v / (height - 1) - 1  # [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]

        # Reshape target for grid_sample: [H, W, C] -> [1, C, H, W]
        target_chw = target_processed.permute(2, 0, 1).unsqueeze(0)

        # Sample colors at site locations
        sampled_colors = F.grid_sample(
            target_chw, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # [1, C, 1, K]
        sampled_colors = sampled_colors.squeeze(0).squeeze(1).T  # [K, C]

        sv.colors.data = sampled_colors

    # Set up optimizer with per-parameter learning rates
    param_groups = [
        {"params": [sv.sites], "lr": lr_sites, "name": "sites"},
        {"params": [sv.colors], "lr": lr_colors, "name": "colors"},
    ]
    if shared_temperature:
        param_groups.append({"params": [sv._beta_raw], "lr": lr_temperature, "name": "beta"})
    else:
        param_groups.append({"params": [sv.log_temperatures], "lr": lr_temperature, "name": "temperatures"})

    optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    # Compute solid angle weights if needed
    if use_solid_angle_weights:
        weights = compute_solid_angle_weights(height, width, device=device)
    else:
        weights = torch.ones(height, width, device=device)

    # For HDR log-loss
    epsilon = 1e-6

    # Prepare for batch-based training if requested
    if batch_size is not None:
        # Generate all directions using same convention as render_equirectangular
        # theta: elevation [0, π], phi: azimuth [-π, π]
        theta = torch.linspace(0, math.pi, height, device=device)
        phi = torch.linspace(-math.pi, math.pi, width, device=device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

        # Y-up convention (matches render_equirectangular)
        x = torch.sin(theta_grid) * torch.sin(phi_grid)
        y = torch.cos(theta_grid)
        z = torch.sin(theta_grid) * torch.cos(phi_grid)

        all_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        all_colors = target_processed.reshape(-1, 3)
        all_weights = weights.reshape(-1)
        N = all_directions.shape[0]

    losses = []
    psnrs = []

    for i in range(num_iterations):
        optimizer.zero_grad()

        if batch_size is not None:
            # Stochastic batch-based training
            idx = torch.randint(0, N, (batch_size,), device=device)
            dirs_batch = all_directions[idx]
            cols_batch = all_colors[idx]
            w_batch = all_weights[idx]

            pred = sv(dirs_batch)

            if use_log_loss:
                pred_clamped = torch.clamp(pred, min=epsilon)
                target_clamped = torch.clamp(cols_batch, min=epsilon)
                diff = torch.log(pred_clamped) - torch.log(target_clamped)
                loss = (diff ** 2 * w_batch.unsqueeze(-1)).mean()
            else:
                loss = ((pred - cols_batch) ** 2 * w_batch.unsqueeze(-1)).mean()
        else:
            # Full-image training
            rendered = sv.render_equirectangular(height, width)

            if use_log_loss:
                rendered_clamped = torch.clamp(rendered, min=epsilon)
                target_clamped = torch.clamp(target_processed, min=epsilon)
                diff = torch.log(rendered_clamped) - torch.log(target_clamped)
                loss = (diff ** 2 * weights.unsqueeze(-1)).mean()
            else:
                loss = ((rendered - target_processed) ** 2 * weights.unsqueeze(-1)).mean()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (i + 1) % 100 == 0:
            # Compute PSNR on processed target
            with torch.no_grad():
                rendered = sv.render_equirectangular(height, width)
                if color_activation == "relu":
                    rendered = torch.clamp(rendered, 0, 1)
                mse = ((rendered - target_processed) ** 2 * weights.unsqueeze(-1)).mean()
                psnr = -10 * torch.log10(mse.clamp(min=1e-8))
                psnrs.append(psnr.item())

                beta_str = ""
                if shared_temperature:
                    beta_str = f", beta={sv.get_beta().item():.2f}"

                print(f"Iter {i+1}/{num_iterations}: Loss={loss.item():.6f}, PSNR={psnr.item():.2f} dB{beta_str}")

    # Final metrics
    with torch.no_grad():
        rendered = sv.render_equirectangular(height, width)
        if color_activation == "relu":
            rendered = torch.clamp(rendered, 0, 1)
        mse = ((rendered - target_processed) ** 2 * weights.unsqueeze(-1)).mean()
        psnr = -10 * torch.log10(mse.clamp(min=1e-8))

    metrics = {
        'final_loss': losses[-1],
        'final_psnr': psnr.item(),
        'losses': losses,
        'mode': mode,
        'color_activation': color_activation,
        'target_processed': target_processed,  # Return processed target for visualization
    }

    return sv, metrics
