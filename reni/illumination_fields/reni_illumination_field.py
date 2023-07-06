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

from typing import Literal, Type, Union, Optional, Dict, Union
from dataclasses import dataclass, field
import wget
import zipfile
import os

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding, Encoding

from reni.illumination_fields.base_spherical_field import ConditionalSphericalField, ConditionalSphericalFieldConfig
from reni.field_components.siren import Siren
from reni.field_components.activations import ExpLayer
from reni.field_components.field_heads import RENIFieldHeadNames

def invariant_grammatrix_representation(
    Z, D, equivariance: Literal["None", "SO2", "SO3"] = "SO2", 
    conditioning: Literal["FiLM", "Concat"] = "Concat",
    axis_of_invariance: int = 1,
    posiiton_encoding: Union[Encoding, None] = None,
):
    """Generates an invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (num_rays x latent_dim x 3)
        D (torch.Tensor): Direction coordinates (num_rays x 3)
        equivariance (str): Type of equivariance to use. Options are 'none', 'SO2', and 'SO3'
        conditioning (str): Type of conditioning to use. Options are 'Concat' and 'FiLM'
        axis_of_invariance (int): The axis of rotation invariance. Should be 0 (x-axis), 1 (y-axis), or 2 (z-axis).
            Default is 1 (y-axis).
    Returns:
        torch.Tensor: Invariant representation 
    """
    assert 0 <= axis_of_invariance < 3, 'axis_of_invariance should be 0, 1, or 2.'
    other_axes = [i for i in range(3) if i != axis_of_invariance]


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
        # Select components along axes orthogonal to the axis of invariance
        z_other = torch.stack((Z[:, :, other_axes[0]], Z[:, :, other_axes[1]]), -1) # [num_rays, latent_dim, 2]
        d_other = torch.stack((D[:, other_axes[0]], D[:, other_axes[1]]), -1).unsqueeze(1) # [num_rays, 2]
        d_other = d_other.expand_as(z_other) # size becomes [10, 5, 2]

        # Invariant representation of Z, gram matrix G=Z*Z' is size num_rays x latent_dim x latent_dim
        G = torch.bmm(z_other, torch.transpose(z_other, 1, 2))

        # Flatten G to be size num_rays x latent_dim^2
        z_other_invar = G.flatten(start_dim=1)

        # Innerprod is size num_rays x latent_dim
        innerprod = (z_other * d_other).sum(dim=-1) # [num_rays, latent_dim]

        # Compute norm along the axes orthogonal to the axis of invariance
        d_other_norm = torch.sqrt(D[::, other_axes[0]] ** 2 + D[:, other_axes[1]] ** 2).unsqueeze(-1) # [num_rays, 1]

        # Get invariant component of Z along the axis of invariance 
        z_invar = Z[:, :, axis_of_invariance] # [num_rays, latent_dim]

        # Get invariant component of D along the axis of invariance
        d_invar = D[:, axis_of_invariance].unsqueeze(-1) # [num_rays, 1]

        if conditioning == "FiLM":
            model_input = torch.cat((d_other_norm, d_invar, innerprod), 1)  # [num_rays, 2 + ndims]
            conditioning_input = torch.cat((z_other_invar, z_invar), 1)  # [num_rays, ndims^2 + ndims]
            return model_input, conditioning_input
        if conditioning == "Concat":
            # model_input is size [num_rays, 2 x ndims + ndims^2 + 2]
            model_input = torch.cat((innerprod, z_other_invar, d_other_norm, z_invar, d_invar), 1)
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

def invariant_vn_representation(
    Z, D, equivariance: Literal["None", "SO2", "SO3"] = "SO2",
    conditioning: Literal["FiLM", "Concat"] = "Concat",
    axis_of_invariance: int = 1
):
    return None

@dataclass
class RENIFieldConfig(ConditionalSphericalFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: RENIField)
    """target class to instantiate"""
    equivariance: Literal["None", "SO2", "SO3"] = "SO2"
    """Type of equivariance to use"""
    invariant_function: Literal["GramMatrix", "VN"] = "GramMatrix"
    """Type of invariant function to use"""
    conditioning: Literal["FiLM", "Concat"] = "Concat"
    """Type of conditioning to use"""
    positional_encoding: Literal["None", "NeRF"] = "None"
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
    output_activation: Literal["sigmoid", "tanh", "relu", "exp", "None"] = "exp"
    """Activation function for output layer"""
    first_omega_0: float = 30.0
    """Omega_0 for first layer"""
    hidden_omega_0: float = 30.0
    """Omega_0 for hidden layers"""
    fixed_decoder: bool = False
    """Whether to fix the decoder weights"""
    split_head: bool = False
    """Whether to split the head into three separate heads, HDR, LDR and BlendWeight"""

class RENIField(ConditionalSphericalField):
    """Base class for illumination fields."""

    def __init__(
        self,
        config: RENIFieldConfig,
        num_train_data: int,
        num_eval_data: int,
    ) -> None:
        super().__init__()
        self.config = config
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
        
        self.num_train_data = num_train_data
        self.num_eval_data = num_eval_data

        train_mu, train_logvar = self.init_latent_codes(self.num_train_data)
        eval_mu, eval_logvar = self.init_latent_codes(self.num_eval_data)

        self.register_parameter("train_mu", train_mu)
        self.register_parameter("train_logvar", train_logvar)
        self.register_parameter("eval_mu", eval_mu)
        self.register_parameter("eval_logvar", eval_logvar)

        self.invariant_function = invariant_grammatrix_representation if self.config.invariant_function == "GramMatrix" else invariant_vn_representation

        self.network = self.setup_network() # ModuleDict {'siren': siren [mlp], 'hdr_head': Union[mlp, None], 'ldr_head': Union[mlp, None], 'mixing_head': Union[mlp, None]}
        
        if self.config.fixed_decoder:
            for _, value in self.network.items():
                for param in value.parameters():
                    param.requires_grad = False

    class fixed_decoder:
        """Context manager to fix the decoder weights

        Example usage:
        ```
        with instance_of_RENIField.fixed_decoder(instance_of_RENIField):
            # do stuff
        
        """
        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            self.prev_state = {k: p.requires_grad for k, p in self.outer.named_parameters() if 'siren' in k}
            for name, param in self.outer.named_parameters():
                if 'siren' in name:
                    param.requires_grad_(False)
            return self

        def __exit__(self, type, value, traceback):
            for name, param in self.outer.named_parameters():
                if 'siren' in name:
                    param.requires_grad_(self.prev_state[name])

    def setup_network(self):
        # TODO handle FiLM
        input_dim = 2 * self.latent_dim + self.latent_dim**2 + 2
        if self.config.positional_encoding == "NeRF":
            self.positional_encoding = NeRFEncoding(in_dim=input_dim, num_frequencies=2, min_freq_exp=0.0, max_freq_exp=2.0, include_input=True)
            input_dim = self.positional_encoding.get_out_dim()

        output_activation = None
        if self.config.output_activation == "exp":
            output_activation = ExpLayer()
        elif self.config.output_activation == "sigmoid":
            output_activation = nn.Sigmoid()
        elif self.config.output_activation == "tanh":
            output_activation = nn.Tanh()
        elif self.config.output_activation == "relu":
            output_activation = nn.ReLU()
        
        hdr_head = None
        ldr_head = None
        mixing_head = None
        if not self.config.split_head:
            siren = Siren(in_dim=input_dim,
                          hidden_layers=self.hidden_layers,
                          hidden_features=self.hidden_features,
                          out_dim=self.out_features,
                          outermost_linear=self.last_layer_linear,
                          first_omega_0=self.first_omega_0,
                          hidden_omega_0=self.hidden_omega_0,
                          out_activation=output_activation)
            network = nn.ModuleDict({'siren': siren})
        else:
            # we want a single siren with output_dim of hidden_features and then three heads with in_dim of hidden_features and out_dim of 3, 3 and 1 respectively
            siren = Siren(in_dim=input_dim,
                          hidden_layers=3,
                          hidden_features=self.hidden_features,
                          out_dim=self.hidden_features,
                          outermost_linear=False,
                          first_omega_0=self.first_omega_0,
                          hidden_omega_0=self.hidden_omega_0,
                          out_activation=None)
            
            hdr_head = Siren(in_dim=self.hidden_features,
                              hidden_layers=3,
                              hidden_features=self.hidden_features,
                              out_dim=3,
                              outermost_linear=False,
                              first_omega_0=self.first_omega_0,
                              hidden_omega_0=self.hidden_omega_0,
                              out_activation=ExpLayer())
            
            ldr_head = Siren(in_dim=self.hidden_features,
                              hidden_layers=3,
                              hidden_features=self.hidden_features,
                              out_dim=3,
                              outermost_linear=False,
                              first_omega_0=self.first_omega_0,
                              hidden_omega_0=self.hidden_omega_0,
                              out_activation=nn.ReLU())
            
            mixing_head = Siren(in_dim=self.hidden_features,
                                hidden_layers=3,
                                hidden_features=self.hidden_features,
                                out_dim=1,
                                outermost_linear=False,
                                first_omega_0=self.first_omega_0,
                                hidden_omega_0=self.hidden_omega_0,
                                out_activation=nn.Sigmoid())
            
            network = nn.ModuleDict({
                'siren': siren,  # Assuming 'siren' is a PyTorch model (nn.Module instance)
                'hdr_head': hdr_head,  # Assuming 'hdr_head' is a PyTorch model (nn.Module instance)
                'ldr_head': ldr_head,  # Assuming 'ldr_head' is a PyTorch model (nn.Module instance)
                'mixing_head': mixing_head,  # Assuming 'mixing_head' is a PyTorch model (nn.Module instance)
            })

        return network

    def sample_latent(self, idx):
        """Sample the latent code at a given index

        Args:
        idx (int): Index of the latent variable to sample

        Returns:
        tuple (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing the sampled latent variable, the mean of the latent variable and the log variance of the latent variable
        """

        if self.training:
            mu = self.train_mu[idx, :, :]
            log_var = self.train_logvar[idx, :, :]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = self.train_mu[idx, :, :]
            mu = self.train_mu[idx, :, :]
            log_var = self.train_logvar[idx, :, :]

        return sample, mu, log_var

    def init_latent_codes(self, num_latents: int):
        """Initializes the latent codes
        
        """
        log_var = torch.nn.Parameter(torch.normal(-5, 1, size=(num_latents, self.latent_dim, 3)))
        
        if self.config.fixed_decoder:
            log_var.requires_grad = False
            mu = torch.nn.Parameter(torch.zeros(num_latents, self.latent_dim, 3))
        else:
            mu = torch.nn.Parameter(torch.randn(num_latents, self.latent_dim, 3))    
        
        return mu, log_var

    def get_outputs(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]
        """
        # we want to batch over camera_indices as these correspond to unique latent codes
        camera_indices = ray_bundle.camera_indices.squeeze() # [num_rays]

        latent_codes, mu, log_var = self.sample_latent(camera_indices) # [num_rays, latent_dim, 3]

        if rotation is not None:
            latent_codes = torch.matmul(latent_codes, rotation)

        directions = ray_bundle.directions # [num_rays, 3] # each has unique latent code defined by camera index

        model_input = self.invariant_function(latent_codes, directions) # [num_rays, 3]

        if self.config.positional_encoding == "NeRF":
            model_input = self.positional_encoding(model_input)

        outputs = {}

        if self.config.split_head:
            base_output = self.network["siren"](model_input) # [num_rays, hidden_features]
            hdr_output = self.network["hdr_head"](base_output) # [num_rays, 3]
            ldr_output = self.network["ldr_head"](base_output) # [num_rays, 3]
            mixing_output = self.network["mixing_head"](base_output) # [num_rays, 1]
            model_outputs = hdr_output * mixing_output + ldr_output * (1 - mixing_output) # [num_rays, 3]
            outputs[RENIFieldHeadNames.HDR] = hdr_output
            outputs[RENIFieldHeadNames.LDR] = ldr_output
            outputs[RENIFieldHeadNames.MIXING] = mixing_output
        else:
            model_outputs = self.network["siren"](model_input) # [num_rays, 3]

        outputs[RENIFieldHeadNames.RGB] = model_outputs
        outputs[RENIFieldHeadNames.MU] = mu
        outputs[RENIFieldHeadNames.LOG_VAR] = log_var

        return outputs

    def get_latents(self):
        """Returns the latents of the field."""

    def forward(self, ray_bundle: RayBundle, rotation: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays, 3]
            rotation: [3, 3]
        """
        return self.get_outputs(ray_bundle=ray_bundle, rotation=rotation)
