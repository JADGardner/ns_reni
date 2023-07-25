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

"""RENI field"""

from typing import Literal, Type, Union, Optional, Dict, Union, Tuple
from dataclasses import dataclass, field
import wget
import zipfile
import os
import contextlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding, Encoding

from reni.illumination_fields.base_spherical_field import SphericalField, SphericalFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames

@dataclass
class EnvironmentMapFieldConfig(SphericalFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: EnvironmentMapField)
    """target class to instantiate"""
    path: Path = Path('path/to/environment_map/s.pt')
    """path to environment map"""
    resolution: Tuple[int, int] = (1024, 512)
    """resolution of environment map"""
    trainable: bool = False
    """whether to train the environment map or not"""
    apply_padding: bool = True
    """whether to apply padding to the environment map or not"""
    scale: float = 1.0
    """scale to apply to the environment map"""

class EnvironmentMapField(SphericalField):
    """A representation of distant illumination as an environment map."""

    def __init__(
        self,
        config: EnvironmentMapFieldConfig,
        num_train_data: int,
        num_eval_data: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_train_data = num_train_data
        self.num_eval_data = num_eval_data
        self.path = config.path
        self.resolution = config.resolution
        self.trainable = config.trainable
        self.apply_padding = config.apply_padding
        self.scale = config.scale

    def sample_envmaps(self, envmaps, directions):
        """Sample colors from the environment maps given a set of 3D directions.

        Args:
            envmaps: Environment maps of shape [unique_indices, 3, H, W].
            directions: Directions of shape [num_directions, 3].

        Returns:
            Sampled colors from environment maps.
        """
        num_directions = directions.shape[0]

        # Convert 3D directions to 2D coordinates in the environment map.
        # Note that we assume directions are already normalized to unit length.
        # We consider that the environment map's up is the positive Z direction.
        phi = torch.atan2(directions[:,1], directions[:,0])  # azimuthal angle
        theta = torch.acos(directions[:,2])  # polar angle

        # Convert spherical to pixel coordinates. We assume the environment map spans 360° horizontally and 180° vertically.
        u = (phi + math.pi) / (2 * math.pi)  # horizontal coordinate between 0 and 1
        v = theta / math.pi  # vertical coordinate between 0 and 1

        # Rescale and shift coordinates to match the grid_sample convention.
        if self.config.apply_padding:
            u = u * (envmaps.shape[-1] - 2 * self.padding) / envmaps.shape[-1]  # Remap u to consider padding
            u = u + self.padding / envmaps.shape[-1]  # Shift u to consider padding
            u = 2 * u - 1  # Convert u to range [-1, 1] for grid sampling.
        else:
            u = 2 * u - 1  # horizontal coordinate between -1 and 1
        v = 2 * v - 1  # vertical coordinate between -1 and 1

        # Repeat coordinates for each environment map.
        u = u[None, :].repeat(envmaps.shape[0], 1)  # [unique_indices, num_directions]
        v = v[None, :].repeat(envmaps.shape[0], 1)  # [unique_indices, num_directions]

        # Concatenate u and v coordinates to form the sampling grid.
        grid = torch.stack([u, v], dim=-1)  # [num_latent_codes, num_directions, 2]

        # Convert the grid to a 4D tensor of shape [B, H, W, 2] as expected by grid_sample.
        grid = grid.view(envmaps.shape[0], num_directions, 1, 2)

        # Sample colors from the environment maps using bilinear interpolation.
        colors = torch.nn.functional.grid_sample(envmaps, grid, align_corners=False, mode='bilinear') # [unique_indices, 3, num_directions, 1]

        return colors.squeeze(-1).permute(0, 2, 1) # [unique_indices, num_directions, 3]

    def get_outputs(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None], latent_codes: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]
            latent_codes: [latent_dim, 3]
        """
        # we want to batch over camera_indices as these correspond to unique latent codes
        camera_indices = ray_bundle.camera_indices.squeeze() # [num_rays]

        if latent_codes is None:
            latent_codes = self.sample_latent(camera_indices) # [num_rays, latent_dim, 3]
        else:
            latent_codes = latent_codes.repeat(ray_bundle.shape[0], 1, 1) # [num_rays, latent_dim, 3]

        if rotation is not None:
            latent_codes = torch.matmul(latent_codes, rotation)

        directions = ray_bundle.directions # [num_rays, 3] # each has unique latent code defined by camera index

        return outputs

    def forward(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None], latent_codes: Union[torch.Tensor, None] = None) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]

        Returns:
            Dict[RENIFieldHeadNames, TensorType]: A dictionary containing the outputs of the field.
        """
        return self.get_outputs(ray_bundle=ray_bundle, rotation=rotation, latent_codes=latent_codes)