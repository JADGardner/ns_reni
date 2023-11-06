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
    """Calculate Blinn-Phong shading."""

    @classmethod
    def forward(
        cls,
        albedo: torch.Tensor,  # shape: (*bs, 3)
        normals: torch.Tensor,  # shape: (*bs, 3)
        light_directions: torch.Tensor,  # shape: (*bs, num_light_directions, 3)
        light_colors: torch.Tensor,  # shape: (*bs, num_light_directions, 3)
        specular: torch.Tensor,  # shape: (*bs, 3)
        shininess: torch.Tensor,  # shape: (*bs,)
        view_directions: torch.Tensor,  # shape: (*bs, 3)
        detach_normals=False,
    ):
        """Calculate Blinn-Phong shading."""

        if detach_normals:
            normals = normals.detach()
        
        # ensure light directions, normals are both normalised
        light_directions = light_directions / light_directions.norm(dim=-1, keepdim=True)

        normals_expanded = normals.unsqueeze(1)

        lambertian_per_light = torch.einsum("...i,...i->...", normals_expanded, light_directions).clamp(min=0.0)

        lambertian_colors = lambertian_per_light.unsqueeze(-1) * light_colors

        shaded_lambertian = albedo * lambertian_colors.sum(1)

        H = (light_directions + view_directions.unsqueeze(1)) / 2.0
        H = H / H.norm(dim=-1, keepdim=True)

        specular_term_per_light = torch.einsum("...i,...i->...", normals_expanded, H).clamp(
            min=0.0
        ) ** shininess.unsqueeze(1)
        specular_term_per_light = specular_term_per_light.unsqueeze(-1)

        # Add normalization factor
        bp_specular_normalisation_factor = (shininess + 2) / (
            4 * (2 - torch.exp(-shininess / 2))
        )

        # Now combine them
        specular_colors = specular_term_per_light * light_colors
        shaded_specular = specular * bp_specular_normalisation_factor.unsqueeze(-1) * specular_colors.sum(1)

        final_color = shaded_lambertian + shaded_specular

        # set minimum to 1e-3
        final_color = final_color.clamp(min=1e-3)

        return final_color

