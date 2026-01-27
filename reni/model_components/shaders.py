# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Shaders for rendering."""
from typing import Optional

from reni.utils.colourspace import linear_to_sRGB

from jaxtyping import Float
import torch
from torch import Tensor, nn


class LambertianShader(nn.Module):
    """Calculate Lambertian shading."""

    @classmethod
    def forward(
        cls,
        albedo: Float[Tensor, "*bs 3"],
        normals: Float[Tensor, "*bs 3"],
        light_directions: Float[Tensor, "*bs num_light_directions 3"],
        light_colors: Float[Tensor, "*bs num_light_directions 3"],
        detach_normals=True,
    ):
        """Calculate Lambertian shading.

        Args:
            albedo: Accumulated albedo along a ray.
            normals: Accumulated normals along a ray.
            light_directions: Directions of light sources [bs, num_light_directions, 3].
            light_colors: Colors of light sources [bs, num_light_directions, 3].
            mask: Mask for valid pixels [bs, 1].
            detach_normals: Detach normals from the computation graph when computing shading.

        Returns:
            Textureless Lambertian shading, Lambertian shading
        """
        if detach_normals:
            normals = normals.detach()

        # Ensure normals have the same shape as light_directions for broadcasting
        normals_expanded = normals.unsqueeze(1)

        # Compute dot product along last dimension [-1], result has shape [bs, num_light_directions]
        lambertian_per_light = torch.einsum("...i,...i->...", normals_expanded, light_directions).clamp(min=0.0)

        # Compute shading for each light, result has shape [bs, num_light_directions, 3]
        lambertian_colors = lambertian_per_light.unsqueeze(-1) * light_colors

        # Sum colors from all lights, result has shape [bs, 3]
        lambertian_color_sum = lambertian_colors.sum(1)

        shaded_albedo = albedo * lambertian_color_sum

        # lambertian_color_sum = linear_to_sRGB(lambertian_color_sum)
        # shaded_albedo = linear_to_sRGB(shaded_albedo)

        return lambertian_color_sum, shaded_albedo


class BlinnPhongShader(nn.Module):
    """Calculate Blinn-Phong shading.

    Optimized for memory efficiency using broadcasting instead of tensor expansion.
    Supports both expanded inputs (N, M, 3) and broadcast-compatible inputs (1, M, 3).
    """

    @classmethod
    def forward(
        cls,
        albedo: torch.Tensor,  # shape: (N, 3)
        normals: torch.Tensor,  # shape: (N, 3)
        light_directions: torch.Tensor,  # shape: (N, M, 3) or (1, M, 3) for broadcasting
        light_colors: torch.Tensor,  # shape: (N, M, 3) or (1, M, 3) for broadcasting
        specular: torch.Tensor,  # shape: (N, 3)
        shininess: torch.Tensor,  # shape: (N,)
        view_directions: torch.Tensor,  # shape: (N, 3)
        detach_normals: bool = False,
        normalize_directions: bool = False,  # Set True only if inputs aren't pre-normalized
    ):
        """Calculate Blinn-Phong shading.

        Args:
            albedo: Diffuse albedo per pixel (N, 3)
            normals: Surface normals per pixel (N, 3), should be unit length
            light_directions: Light directions (N, M, 3) or (1, M, 3) for memory-efficient broadcasting
            light_colors: Light colors/intensities (N, M, 3) or (1, M, 3)
            specular: Specular coefficient per pixel (N, 3)
            shininess: Specular exponent per pixel (N,)
            view_directions: View/camera directions per pixel (N, 3)
            detach_normals: If True, detach normals from computation graph
            normalize_directions: If True, normalize light_directions (skip if pre-normalized)

        Returns:
            Final shaded color per pixel (N, 3)
        """
        if detach_normals:
            normals = normals.detach()

        # Only normalize if explicitly requested (saves compute if pre-normalized)
        if normalize_directions:
            light_directions = light_directions / light_directions.norm(dim=-1, keepdim=True)

        # Expand normals for broadcasting: (N, 3) -> (N, 1, 3)
        normals_expanded = normals.unsqueeze(1)

        # Lambertian term: dot(N, L) clamped to [0, inf)
        # Broadcasting: (N, 1, 3) * (N or 1, M, 3) -> (N, M, 3) -> sum -> (N, M)
        lambertian_per_light = torch.einsum("...i,...i->...", normals_expanded, light_directions).clamp(min=0.0)

        # Weight by light colors and sum over all lights
        # (N, M, 1) * (N or 1, M, 3) -> (N, M, 3) -> sum(dim=1) -> (N, 3)
        lambertian_colors = lambertian_per_light.unsqueeze(-1) * light_colors
        del lambertian_per_light
        lambertian_colors_sum = lambertian_colors.sum(1)
        del lambertian_colors

        shaded_lambertian = albedo * lambertian_colors_sum
        del lambertian_colors_sum

        # Half-vector for Blinn-Phong: H = normalize(L + V)
        # view_directions: (N, 3) -> (N, 1, 3) for broadcasting with light_directions
        H = light_directions + view_directions.unsqueeze(1)
        H = H / (H.norm(dim=-1, keepdim=True) + 1e-8)  # Add epsilon for numerical stability

        # Specular term: dot(N, H)^shininess
        # shininess: (N,) -> (N, 1) for broadcasting
        specular_term_per_light = torch.einsum("...i,...i->...", normals_expanded, H).clamp(min=0.0)
        del H
        specular_term_per_light = specular_term_per_light ** shininess.unsqueeze(1)
        specular_term_per_light = specular_term_per_light.unsqueeze(-1)

        # Blinn-Phong normalization factor: (n+2) / (4 * (2 - exp(-n/2)))
        bp_specular_normalisation_factor = (shininess + 2) / (
            4 * (2 - torch.exp(-shininess / 2))
        )

        # Combine specular with light colors
        specular_colors = specular_term_per_light * light_colors
        del specular_term_per_light
        shaded_specular = specular * bp_specular_normalisation_factor.unsqueeze(-1) * specular_colors.sum(1)
        del specular_colors

        final_color = shaded_lambertian + shaded_specular

        # Clamp to minimum value to avoid pure black
        final_color = final_color.clamp(min=1e-3)

        return final_color


class BlinnPhongShaderChunked(nn.Module):
    """Memory-efficient Blinn-Phong shader that processes pixels in chunks.

    Use this for very large images or limited GPU memory.
    """

    def __init__(self, chunk_size: int = 4096):
        super().__init__()
        self.chunk_size = chunk_size
        self.shader = BlinnPhongShader()

    def forward(
        cls,
        albedo: torch.Tensor,  # shape: (N, 3)
        normals: torch.Tensor,  # shape: (N, 3)
        light_directions: torch.Tensor,  # shape: (1, M, 3) - shared across pixels
        light_colors: torch.Tensor,  # shape: (1, M, 3) - shared across pixels
        specular: torch.Tensor,  # shape: (N, 3)
        shininess: torch.Tensor,  # shape: (N,)
        view_directions: torch.Tensor,  # shape: (N, 3)
        detach_normals: bool = False,
        normalize_directions: bool = False,
    ):
        """Process in chunks to reduce peak memory usage."""
        N = normals.shape[0]
        chunk_size = cls.chunk_size

        results = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            chunk_result = BlinnPhongShader.forward(
                albedo=albedo[start:end],
                normals=normals[start:end],
                light_directions=light_directions,  # Shared, no slicing
                light_colors=light_colors,  # Shared, no slicing
                specular=specular[start:end],
                shininess=shininess[start:end],
                view_directions=view_directions[start:end],
                detach_normals=detach_normals,
                normalize_directions=normalize_directions,
            )
            results.append(chunk_result)

        return torch.cat(results, dim=0)

