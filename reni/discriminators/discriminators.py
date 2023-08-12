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
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.rays import Frustums, RaySamples, RayBundle

from reni.field_components.field_heads import RENIFieldHeadNames
from reni.field_components.vn_layers import VNLinear, VNInvariant, VNTransformerEncoder


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
    def get_outputs(self, ray_samples: RaySamples, rgb: TensorType["batch_size", "num_rays", 3]):
        """Returns the prediction of the discriminator."""
        raise NotImplementedError

    def forward(self, ray_samples: RaySamples, rgb: TensorType["batch_size", "num_rays", 3]):
        """Evaluates discriminator for a given image batch.

        Args:
            ray_samples: [batch_size, num_rays]
            rgb: [batch_size, num_rays, 3]
        """
        return self.get_outputs(ray_samples, rgb)


@dataclass
class VNTransformerDiscriminatorConfig(BaseDiscriminatorConfig):
    _target: Type = field(default_factory=lambda: VNTransformerDiscriminator)
    """target class to instantiate"""
    hidden_dim: int = 64
    """hidden dimension of the transformer"""
    depth: int = 2
    """depth of the transformer encoder"""
    dim_head: int = 64
    """dimension of the transformer head"""
    heads: int = 4
    """number of heads in the transformer"""
    bias_epsilon: float = 1e-6
    """epsilon for the bias in the transformer"""
    l2_dist_attn: bool = False
    """whether to use l2 distance in the attention"""
    flash_attn: bool = False
    """whether to use flash attention in the attention"""
    invariance: Literal["None", "SO2", "SO3"] = "None"
    """invariance of the discriminator"""


class VNTransformerDiscriminator(nn.Module):
    def __init__(
        self,
        config: VNTransformerDiscriminatorConfig,
    ) -> None:
        super().__init__()
        self.config = config
        
        self.vn_proj_in = nn.Sequential(
            Rearrange('... c -> ... 1 c'),
            VNLinear(1, self.config.hidden_dim, bias_epsilon = self.config.bias_epsilon)
        )

        self.encoder = VNTransformerEncoder(
            dim = self.config.hidden_dim,
            depth = self.config.depth,
            dim_head = self.config.dim_head,
            heads = self.config.heads,
            bias_epsilon = self.config.bias_epsilon,
            dim_coor = 6, # 3 for directions, 3 for rgb
            l2_dist_attn = self.config.l2_dist_attn,
            flash_attn = self.config.flash_attn
        )

        self.vn_invariant = VNInvariant(self.config.hidden_dim) # make either so(0,2,3)

        self.out = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )


    def get_outputs(self, ray_samples: RaySamples, rgb: TensorType["batch_size", "num_rays", 3]):
        """Returns the prediction of the discriminator."""
        directions = ray_samples.frustums.directions # [batch_size, num_rays, 3]
        
        # set both directions and rgb to be long
        directions = directions.long()
        rgb = rgb.long()

        # early fusion just concatenates the two inputs
        x = torch.cat([directions, rgb], dim=-1) # [batch_size, num_rays, 6]
        x = self.vn_proj_in(x) # [batch_size, num_rays, hidden_dim, 6]
        x = self.encoder(x) # [batch_size, num_rays, hidden_dim, 6]
        x = self.vn_invariant(x) # [batch_size, num_rays, 3]
        x_invar = x
        x = self.out(x) # [batch_size, 1]

        return x, x_invar
        

    def forward(self, ray_samples: RaySamples, rgb: TensorType["batch_size", "num_rays", 3]):
        """Evaluates discriminator for a given image batch.

        Args:
            ray_samples: [batch_size, num_rays]
            rgb: [batch_size, num_rays, 3]
        """
        return self.get_outputs(ray_samples, rgb)
    
