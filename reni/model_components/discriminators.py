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
class BaseDiscriminatorConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: BaseDiscriminator)
    """target class to instantiate"""


class BaseDiscriminator(nn.Module):
    """Base class for RESGAN discriminators."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def get_outputs(self, image):
        """Returns the prediction of the discriminator."""
        raise NotImplementedError

    def forward(self, image):
        """Evaluates discriminator for a given image batch.

        Args:
            image: [batch_size, 3, H, W]
        """
        return self.get_outputs(image)


@dataclass
class CNNDiscriminatorConfig(BaseDiscriminatorConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: CNNDiscriminator)
    """target class to instantiate"""
    channels: int = 64
    """Number of channels in the first layer of the discriminator."""
    image_width: int = 256
    """Width of the input image."""


class CNNDiscriminator(nn.Module):
    """Base class for RESGAN discriminators."""

    def __init__(
        self,
        config: CNNDiscriminatorConfig,
    ) -> None:
        super().__init__()

        self.config = config
        W = self.config.image_width
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.config.channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.config.channels),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.config.channels, self.config.channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.config.channels * 2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.config.channels * 2, self.config.channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.config.channels * 4),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(self.config.channels * 4 * W//16 * W//8, 1)

    def get_outputs(self, image):
        """Returns the prediction of the discriminator."""
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten
        x = x.view(x.shape[0], -1)

        # Fully connected layer
        x = self.fc(x)

        # Use sigmoid to get a probability
        x = torch.sigmoid(x)
        
        return x


    def forward(self, image):
        """Evaluates discriminator for a given image batch.

        Args:
            image: [batch_size, 3, H, W]
        """
        return self.get_outputs(image)
