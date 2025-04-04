# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import contextlib
from typing import Generator, Optional

import nerfacc
import torch
from torch import nn, Tensor
from jaxtyping import Float
import torch.nn.functional as F

from reni.utils.colourspace import linear_to_sRGB

BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None

@contextlib.contextmanager
def background_color_override_context(mode: Float[Tensor, "3"]) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE  # pylint: disable=global-statement
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color


class RGBLambertianRenderer(nn.Module):
    """Renderer for RGB Lambertian field with visibility."""

    @classmethod
    def render_and_combine_rgb(
        cls,
        albedos: Float[Tensor, "*bs num_samples 3"],
        normals: Float[Tensor, "*bs num_samples 3"],
        light_directions: Float[Tensor, "*bs num_samples 3"],
        light_colors: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 3"],
        ray_indices: Optional[Float[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            albedo: Albedo for each sample [num_rays, num_samples, 3]
            normal: Normal for each sample [num_rays, num_samples, 3]
            light_directions: Light directions for each sample [num_rays * num_samples, num_light_directions, 3]
            light_colors: Light colors for each sample [num_rays * num_samples, num_light_directions, 3]
            weights: Weights for each sample [num_rays, num_samples, 1]
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """

        albedos = albedos.view(-1, 3)
        normals = normals.view(-1, 3)

        # compute dot product between normals [num_rays * samples_per_ray, 3] and light directions [num_rays * samples_per_ray, num_illumination_directions, 3]
        dot_prod = torch.einsum(
            "bi,bji->bj", normals, light_directions
        )  # [num_rays * samples_per_ray, num_reni_directions]

        # clamp dot product values to be between 0 and 1
        dot_prod = torch.clamp(dot_prod, 0, 1)

        # count the number of elements in dot product that are greater than 0
        count = torch.sum((dot_prod > 0).float(), dim=1, keepdim=True)

        # replace all 0 values with 1 to avoid division by 0
        count = torch.where(count > 0, count, torch.ones_like(count))

        dot_prod = dot_prod / count

        # compute final color by multiplying dot product with albedo color and light color
        color = torch.einsum("bi,bj,bji->bi", albedos, dot_prod, light_colors)  # [num_rays * samples_per_ray, 3]

        radiance = color.view(*weights.shape[:-1], 3)

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            comp_rgb = nerfacc.accumulate_along_rays(weights, ray_indices, radiance, num_rays)
            accumulated_weight = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
        else:
            comp_rgb = torch.sum(weights * radiance, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        comp_rgb = linear_to_sRGB(comp_rgb)

        background_color = torch.tensor([1.0, 1.0, 1.0], device=comp_rgb.device, dtype=comp_rgb.dtype)

        comp_rgb = comp_rgb + background_color.to(weights.device) * (1.0 - accumulated_weight)

        return comp_rgb

    def forward(
        self,
        albedos: Float[Tensor, "*bs num_samples 3"],
        normals: Float[Tensor, "*bs num_samples 3"],
        light_directions: Float[Tensor, "*bs num_samples 3"],
        light_colors: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 3"],
        ray_indices: Optional[Float[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            albedo: Albedo for each sample
            normal: Normal for each sample
            light_directions: Light directions for each sample
            light_colors: Light colors for each sample
            visibility: Visibility of illumination for each sample
            weights: Weights for each sample
            background_illumination: Background color if ray does not hit anything
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of rgb values.
        """

        rgb = self.render_and_combine_rgb(
            albedos=albedos,
            normals=normals,
            light_directions=light_directions,
            light_colors=light_colors,
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb
