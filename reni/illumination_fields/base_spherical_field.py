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
Base class for the graphs.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Type, Union

import torch
from torch import nn

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.rays import Frustums, RaySamples, RayBundle


# Field related configs
@dataclass
class SphericalIlluminationFieldConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SphericalIlluminationField)
    """target class to instantiate"""


class SphericalIlluminationField(nn.Module):
    """Base class for illumination fields."""

    config: SphericalIlluminationFieldConfig

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def get_outputs(self, unique_indices, inverse_indices, directions, illumination_type):
        """Computes and returns the colors. Returns output field values.

        Args:
            unique_indices: [rays_per_batch]
            inverse_indices: [rays_per_batch, samples_per_ray]
            directions: [num_directions, 3]
        """

    @abstractmethod
    def get_latents(self):
        """Returns the latents of the field."""

    def forward(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None]):
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]
        """
        unique_indices, inverse_indices = torch.unique(camera_indices, return_inverse=True)
        illumination_colours, illumination_directions = self.get_outputs(
            unique_indices, inverse_indices, directions, rotation, illumination_type
        )
        return illumination_colours, illumination_directions
