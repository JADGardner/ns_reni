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
from typing import Literal, Type, Union, Dict

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.rays import Frustums, RaySamples, RayBundle

from reni.field_components.field_heads import RENIFieldHeadNames

# Field related configs
@dataclass
class ConditionalSphericalFieldConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: ConditionalSphericalField)
    """target class to instantiate"""


class ConditionalSphericalField(nn.Module):
    """Base class for illumination fields."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def get_outputs(self,  ray_bundle: RayBundle, rotation: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field."""
        pass

    def forward(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]
        """
        return self.get_outputs(ray_bundle=ray_bundle, rotation=rotation)
