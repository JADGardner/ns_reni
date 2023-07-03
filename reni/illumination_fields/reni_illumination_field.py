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

from typing import Literal, Type
from dataclasses import dataclass, field
import wget
import zipfile
import os

import numpy as np
import torch
from torch import nn

from reni_neus.illumination_fields.base_illumination_field import IlluminationField, IlluminationFieldConfig
from reni_neus.utils.utils import sRGB

def invariant_representation(
    Z, D, equivariance: Literal["None", "SO2", "SO3"] = "SO2", conditioning: Literal["FiLM", "Concat"] = "Concat"
):
    """Generates an invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)
        equivariance (str): Type of equivariance to use. Options are 'none', 'SO2', and 'SO3'
        conditioning (str): Type of conditioning to use. Options are 'Concat' and 'FiLM'

    Returns:
        torch.Tensor: Invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    if equivariance == "None":
        innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
        z_input = Z.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
        if conditioning == "FiLM":
            return innerprod, z_input
        if conditioning == "Concat":
            model_input = torch.cat((innerprod, z_input), 2)
            return model_input
        raise ValueError(f"Invalid conditioning type {conditioning}")

    if equivariance == "SO2":
        z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
        d_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
        # Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
        G = torch.bmm(z_xz, torch.transpose(z_xz, 1, 2))
        # Flatten G and replicate for all pixels, giving size B x npix x ndims^2
        z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
        # innerprod is size B x npix x ndims
        innerprod = torch.bmm(d_xz, torch.transpose(z_xz, 1, 2))
        d_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
        # Copy Z_y for every pixel to be size B x npix x ndims
        z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
        # Just the y component of D (B x npix x 1)
        d_y = D[:, :, 1].unsqueeze(2)
        if conditioning == "FiLM":
            model_input = torch.cat((d_xz_norm, d_y, innerprod), 2)  # [B, npix, 2 + ndims]
            conditioning_input = torch.cat((z_xz_invar, z_y), 2)  # [B, npix, ndims^2 + ndims]
            return model_input, conditioning_input
        if conditioning == "Concat":
            # model_input is size B x npix x 2 x ndims + ndims^2 + 2
            model_input = torch.cat((innerprod, z_xz_invar, d_xz_norm, z_y, d_y), 2)
            return model_input
        raise ValueError(f"Invalid conditioning type {conditioning}")

    if equivariance == "SO3":
        G = Z @ torch.transpose(Z, 1, 2)
        innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
        z_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
        if conditioning == "FiLM":
            return innerprod, z_invar
        if conditioning == "Concat":
            return torch.cat((innerprod, z_invar), 2)
        raise ValueError(f"Invalid conditioning type {conditioning}")


class RENIFieldConfig(SphericalIlluminationFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SphericalIlluminationField)
    """target class to instantiate"""
    equivariance: Literal["None", "SO2", "SO3"] = "SO2"
    """Type of equivariance to use"""
    invariant_function: Literal["GramMatrix", "VN"] = "GramMatrix"
    """Type of invariant function to use"""
    conditioning: Literal["FiLM", "Concat"] = "Concat"
    """Type of conditioning to use"""
    positional_encoding: Literal["None", "NeRF"] = "NeRF"
    """Type of positional encoding to use"""
    latent_dim: int = 36
    """Dimensionality of latent code"""
    hidden_layers: int = 3
    """Number of hidden layers"""
    hidden_features: int = 128
    """Number of hidden features"""
    mapping_layers: int = 3
    """Number of mapping layers"""
    mapping_features: int = 128
    """Number of mapping features"""
    out_features: int = 3 # RGB
    """Number of output features"""
    last_layer_linear: bool = False
    """Whether to use a linear layer as the last layer"""
    output_activation: str = "sigmoid"
    """Activation function for output layer"""
    first_omega_0: float = 30.0
    """Omega_0 for first layer"""
    hidden_omega_0: float = 30.0
    """Omega_0 for hidden layers"""

class RENIField(SphericalIlluminationField):
    """Base class for illumination fields."""

    config: RENIFieldConfig

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.equivariance = self.config.equivariance
        self.conditioning = self.config.conditioning
        self.latent_dim = self.config.latent_dim
        self.hidden_layers = self.config.hidden_layers
        self.hidden_features = self.config.hidden_features
        self.mapping_layers = self.config.mapping_layers
        self.mapping_features = self.config.mapping_features
        self.out_features = self.config.out_features
        self.last_layer_linear = self.config.last_layer_linear
        self.output_activation = self.config.output_activation
        self.first_omega_0 = self.config.first_omega_0
        self.hidden_omega_0 = self.config.hidden_omega_0

    def get_outputs(self, ray_bundle, rotation):
        """Returns the outputs of the field."""
        pass

    def get_latents(self):
        """Returns the latents of the field."""

    def forward(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None]):
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]
        """
        return get_outputs(self, ray_bundle, rotation)
