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
from typing import Literal, Type, Union, Dict, Optional, List, Tuple

import torch
from torch import nn, Tensor
from jaxtyping import Float
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.rays import Frustums, RaySamples, RayBundle

from reni.field_components.field_heads import RENIFieldHeadNames
from reni.field_components.vn_layers import VNLinear, VNInvariant, VNTransformerEncoder, VNLayerNorm


@dataclass
@dataclass
class VariationalVNEncoderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VariationalVNEncoder)
    """target class to instantiate"""
    hidden_dim: int = 64
    """hidden dimension of the transformer"""
    depth: int = 2
    """depth of the transformer encoder"""
    dim_head: int = 64
    """dimension of the transformer head"""
    heads: int = 4
    """number of heads in the transformer"""
    bias_epsilon: float = 0.0
    """epsilon for the bias in the transformer"""
    l2_dist_attn: bool = False
    """whether to use l2 distance in the attention"""
    flash_attn: bool = False
    """whether to use flash attention in the attention"""
    invariance: Literal["None", "SO2", "SO3"] = "None"
    """invariance of the discriminator"""
    return_intermediate_components: bool = False
    """whether to return the intermediate components of the discriminator (for testing)"""
    output_dim: int = 1
    """output dimension of the discriminator"""
    output_activation: torch.nn.Module = nn.Sigmoid()
    """output activation of the discriminator"""
    fusion_strategy: Literal["early", "late"] = "early"
    """early fusion concat at the input, late fusion concat at the output"""
    num_input_rays: int = 8096
    """number of input rays"""


class VariationalVNEncoder(nn.Module):
    def __init__(
        self,
        config: VariationalVNEncoderConfig,
        num_input_rays: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_input_rays = num_input_rays
        self.output_dim = output_dim
        if num_input_rays is None:
            self.num_input_rays = self.config.num_input_rays
        if output_dim is None:
            self.output_dim = self.config.output_dim

        if self.config.fusion_strategy == "early":
            self.dim_coor = 6
        elif self.config.fusion_strategy == "late":
            self.dim_coor = 3
        
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
            dim_coor = self.dim_coor,
            l2_dist_attn = self.config.l2_dist_attn,
            flash_attn = self.config.flash_attn
        )

        self.layer_norm = VNLayerNorm(self.config.hidden_dim)

        if self.config.invariance == "None":
          self.vn_invariant = nn.Identity()
        else:
          self.vn_invariant = VNInvariant(self.config.hidden_dim, dim_coor=self.dim_coor)
        
        self.fc1 = nn.Linear(self.num_input_rays * 6, 512)
        # Project the intermediate representation to mean and log variance
        self.fc_mean = nn.Linear(512, self.output_dim)
        self.fc_logvar = nn.Linear(512, self.output_dim)

    def get_outputs(self, ray_samples: RaySamples, rgb: Float[Tensor, "batch_size, num_rays, 3"]):
        """Returns the prediction of the discriminator."""
        directions = ray_samples.frustums.directions # [batch_size, num_rays, 3]
        
        # set both directions and rgb to be long
        directions = directions
        rgb = rgb

        # early fusion just concatenates the two inputs
        if self.config.fusion_strategy == "early":
            x1 = torch.cat([directions, rgb], dim=-1) # [batch_size, num_rays, 6]
        else:
            x1 = directions
        x2 = self.vn_proj_in(x1) # [batch_size, num_rays, hidden_dim, self.dim_coor]
        x3 = self.layer_norm(self.encoder(x2)) # [batch_size, num_rays, hidden_dim, self.dim_coor]

        if self.config.invariance == "SO2":
            x3_z = x3[..., 2:3] # [batch_size, num_rays, hidden_dim, 1]
            x4_z = Reduce('b n f c -> b n c', 'mean')(x3_z) # [batch_size, num_rays, 1]
            x4 = self.vn_invariant(x3) # [batch_size, num_rays, self.dim_coor]
            # swap z component with x4_z
            x4[:, :, 2:3] = x4_z
        else:
            x4 = self.vn_invariant(x3) # [batch_size, num_rays, self.dim_coor]

        if self.config.fusion_strategy == "late":
            x4 = torch.cat([x4, rgb], dim=-1) # [batch_size, num_rays, 6]
        
        x5 = Rearrange('b n c -> b (n c)')(x4)

        x6 = nn.ReLU()(self.fc1(x5)) # [batch_size, 512]
        mean = self.fc_mean(x6) # [batch_size, output_dim]
        logvar = self.fc_logvar(x6) # [batch_size, output_dim]

        if self.config.return_intermediate_components:
            return (mean, logvar), (x1, x2, x3, x4)
        else:
            return mean, logvar
        

    def forward(self, ray_samples: RaySamples, rgb: Float[Tensor, "batch_size, num_rays, 3"]):
        """Evaluates discriminator for a given image batch.

        Args:
            ray_samples: [batch_size, num_rays]
            rgb: [batch_size, num_rays, 3]
        """
        return self.get_outputs(ray_samples, rgb)