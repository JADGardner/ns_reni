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

from jaxtyping import Float
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
            detach_normals: Detach normals from the computation graph when computing shading.

        Returns:
            Textureless Lambertian shading, Lambertian shading
        """
        if detach_normals:
            normals = normals.detach()

        # Ensure normals have the same shape as light_directions for broadcasting
        normals_expanded = normals.unsqueeze(1)

        # Compute dot product along last dimension [-1], result has shape [bs, num_light_directions]
        lambertian_per_light = (normals_expanded @ light_directions).squeeze(-1).clamp(min=0)

        # Compute shading for each light, result has shape [bs, num_light_directions, 3]
        lambertian_colors = lambertian_per_light.unsqueeze(-1) * light_colors

        # Sum colors from all lights, result has shape [bs, 3]
        lambertian_color_sum = lambertian_colors.sum(1)

        shaded_albedo = albedo * lambertian_color_sum

        return lambertian_color_sum, shaded_albedo

