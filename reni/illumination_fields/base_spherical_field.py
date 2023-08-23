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
Base class for the Spherical Neural Fields.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type, Union, Dict, Any
import contextlib

import torch
from torch import nn
from torchtyping import TensorType

from reni.field_components.field_heads import RENIFieldHeadNames

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.rays import RaySamples

# Field related configs
@dataclass
class SphericalFieldConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SphericalField)
    """target class to instantiate"""


class SphericalField(nn.Module):
    """Base class for illumination fields."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def get_outputs(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.
        
        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
        """
        raise NotImplementedError

    def forward(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
        """
        return self.get_outputs(ray_samples=ray_samples, rotation=rotation)


# RENI-Like Fields related configs
@dataclass
class BaseRENIFieldConfig(SphericalFieldConfig):
    """Configuration for RENI-Like Field instantiation"""

    _target: Type = field(default_factory=lambda: BaseRENIFieldConfig)
    """target class to instantiate"""
    fixed_decoder: bool = False
    """Whether to fix the decoder weights"""


class BaseRENIField(SphericalField):
    """Base class for RENI-Like Fields, abstract classes to match RENI training and evaluation."""

    def __init__(
        self,
        config: BaseRENIFieldConfig,
        num_train_data: Union[int, None],
        num_eval_data: Union[int, None],
        normalisations: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.num_train_data = num_train_data
        self.num_eval_data = num_eval_data
        self.normalisations = normalisations

        assert "min_max" in self.normalisations
        assert "log_domain" in self.normalisations
        
        if normalisations["min_max"] is not None:
            self.register_buffer("min_max", torch.tensor(self.normalisations["min_max"]))

        self.register_buffer("log_domain", torch.tensor(self.normalisations["log_domain"]))

        self.fixed_decoder = config.fixed_decoder

    @abstractmethod
    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to fix RENI decoder weights.

        Example usage:
        ```
        with instance_of_BaseRENIField.hold_decoder_fixed():
            # do stuff
        ```
        """
        raise NotImplementedError
    
    @abstractmethod
    def unnormalise(self, rgb: torch.Tensor) -> torch.Tensor:
        """Undo normalisations applied to the HDR RGB values prior to fitting"""
        raise NotImplementedError

    @abstractmethod
    def get_outputs(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None] = None, latent_codes: Union[torch.Tensor, None] = None) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.
        
        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
            latent_codes: [latent_shape]
        """
        raise NotImplementedError

    def forward(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None] = None, latent_codes: Union[torch.Tensor, None] = None) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays]
            rotation: [3, 3]
            latent_codes: [latent_shape]
        """
        return self.get_outputs(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)


