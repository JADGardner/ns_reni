# Copyright 2023 The University of York. All rights reserved.
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

from typing import Literal, Type, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import contextlib

import torch
from torch import nn
from torchtyping import TensorType
from einops.layers.torch import Rearrange

from reni.illumination_fields.base_spherical_field import BaseRENIField, BaseRENIFieldConfig
from reni.field_components.siren import Siren
from reni.field_components.film_siren import FiLMSiren
from reni.field_components.activations import ExpLayer
from reni.field_components.transformer_decoder import Decoder
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.field_components.vn_layers import VNInvariant, VNLinear

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding


@dataclass
class RENIFieldConfig(BaseRENIFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: RENIField)
    """target class to instantiate"""
    equivariance: Literal["None", "SO2", "SO3"] = "SO2"
    """Type of equivariance to use"""
    axis_of_invariance: Literal["x", "y", "z"] = "y"
    """Which axis should SO2 equivariance be invariant to"""
    invariant_function: Literal["GramMatrix", "VN"] = "GramMatrix"
    """Type of invariant function to use"""
    conditioning: Literal["FiLM", "Concat", "Attention"] = "Concat"
    """Type of conditioning to use"""
    positional_encoding: Literal["NeRF"] = "NeRF"
    """Type of positional encoding to use"""
    encoded_input: Literal["None", "Directions", "Conditioning", "Both"] = "Directions"
    """Type of input to encode"""
    latent_dim: int = 36
    """Dimensionality of latent code, N for a latent code size of (N x 3)"""
    hidden_layers: int = 3
    """Number of hidden layers"""
    hidden_features: int = 128
    """Number of hidden features"""
    mapping_layers: int = 3
    """Number of mapping layers"""
    mapping_features: int = 128
    """Number of mapping features"""
    num_attention_heads: int = 8
    """Number of attention heads"""
    num_attention_layers: int = 3
    """Number of attention layers"""
    out_features: int = 3  # RGB
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
    trainable_scale: Union[bool, Literal["train", "eval", "both"]] = False
    """Whether to train the scale parameter"""
    old_implementation: bool = False
    """Whether to match implementation of old RENI, when using old checkpoints"""


class RENIField(BaseRENIField):
    """Base class for illumination fields."""

    def __init__(
        self,
        config: RENIFieldConfig,
        num_train_data: Optional[int] = None,
        num_eval_data: Optional[int] = None,
        normalisations: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            config=config, num_train_data=num_train_data, num_eval_data=num_eval_data, normalisations=normalisations
        )
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
        self.old_implementation = self.config.old_implementation
        self.axis_of_invariance = ["x", "y", "z"].index(self.config.axis_of_invariance)

        if self.num_train_data is not None:
            train_mu, train_logvar = self.init_latent_codes(self.num_train_data, "train")
            self.register_parameter("train_mu", train_mu)
            self.register_parameter("train_logvar", train_logvar)

            if self.config.trainable_scale in [True, "train", "both"]:
                self.train_scale = nn.Parameter(torch.ones(self.num_train_data))

        if self.num_eval_data is not None:
            eval_mu, eval_logvar = self.init_latent_codes(self.num_eval_data, "eval")
            self.register_parameter("eval_mu", eval_mu)
            self.register_parameter("eval_logvar", eval_logvar)

            if self.config.trainable_scale in [True, "eval", "both"]:
                self.eval_scale = nn.Parameter(torch.ones(self.num_eval_data))

        if self.config.invariant_function == "GramMatrix":
            self.invariant_function = self.gram_matrix_invariance
        else:
            self.vn_proj_in = nn.Sequential(
                Rearrange("... c -> ... 1 c"), VNLinear(dim_in=1, dim_out=1, bias_epsilon=0)
            )
            dim_coor = 2 if self.config.equivariance == "SO2" else 3
            self.vn_invar = VNInvariant(dim=1, dim_coor=dim_coor)
            self.invariant_function = self.vn_invariance

        self.network = self.setup_network()

        if self.fixed_decoder:
            for param in self.network.parameters():
                param.requires_grad = False

            if self.config.invariant_function == "VN":
                for param in self.vn_proj_in.parameters():
                    param.requires_grad = False
                for param in self.vn_invar.parameters():
                    param.requires_grad = False

    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to fix the decoder weights

        Example usage:
        ```
        with instance_of_RENIField.hold_decoder_fixed():
            # do stuff
        ```
        """
        prev_state_network = {name: p.requires_grad for name, p in self.network.named_parameters()}
        for param in self.network.parameters():
            param.requires_grad = False
        if self.config.invariant_function == "VN":
            prev_state_proj_in = {k: p.requires_grad for k, p in self.vn_proj_in.named_parameters()}
            prev_state_invar = {k: p.requires_grad for k, p in self.vn_invar.named_parameters()}
            for param in self.vn_proj_in.parameters():
                param.requires_grad = False
            for param in self.vn_invar.parameters():
                param.requires_grad = False
        if self.config.trainable_scale in [True, "train", "both"]:
            prev_state_scale = self.train_scale.requires_grad
            self.train_scale.requires_grad = False

        prev_decoder_state = self.fixed_decoder
        self.fixed_decoder = True
        try:
            yield
        finally:
            # Restore the previous requires_grad state
            for name, param in self.network.named_parameters():
                param.requires_grad = prev_state_network[name]
            if self.config.invariant_function == "VN":
                for name, param in self.vn_proj_in.named_parameters():
                    param.requires_grad_(prev_state_proj_in[name])
                for name, param in self.vn_invar.named_parameters():
                    param.requires_grad_(prev_state_invar[name])
            if self.config.trainable_scale in [True, "train", "both"]:
                self.train_scale.requires_grad_(prev_state_scale)
            self.fixed_decoder = prev_decoder_state

    def vn_invariance(self, Z, D, equivariance: Literal["None", "SO2", "SO3"] = "SO2", axis_of_invariance: int = 1):
        """Generates an invariant representation from latent code Z and direction coordinates D.

        Args:
            Z (torch.Tensor): Latent code (num_rays x latent_dim x 3)
            D (torch.Tensor): Direction coordinates (num_rays x 3)
            equivariance (str): Type of equivariance to use. Options are 'none', 'SO2', and 'SO3'
            axis_of_invariance (int): The axis of rotation invariance. Should be 0 (x-axis), 1 (y-axis), or 2 (z-axis).
                Default is 1 (y-axis).
        Returns:
            torch.Tensor: Invariant representation
        """
        assert 0 <= axis_of_invariance < 3, "axis_of_invariance should be 0, 1, or 2."
        other_axes = [i for i in range(3) if i != axis_of_invariance]

        if equivariance == "None":
            # get inner product between latent code and direction coordinates
            innerprod = torch.sum(Z * D.unsqueeze(1), dim=-1)  # [num_rays, latent_dim]
            z_input = Z.flatten(start_dim=1)  # [num_rays, latent_dim * 3]
            return innerprod, z_input

        if equivariance == "SO2":
            z_other = torch.stack((Z[:, :, other_axes[0]], Z[:, :, other_axes[1]]), -1)  # [num_rays, latent_dim, 2]
            d_other = torch.stack((D[:, other_axes[0]], D[:, other_axes[1]]), -1).unsqueeze(1)  # [num_rays, 2]
            d_other = d_other.expand_as(z_other)  # size becomes [num_rays, latent_dim, 2]

            z_other_emb = self.vn_proj_in(z_other)  # [num_rays, latent_dim, 1, 2]
            z_other_invar = self.vn_invar(z_other_emb)  # [num_rays, latent_dim, 2]

            # Get invariant component of Z along the axis of invariance
            z_invar = Z[:, :, axis_of_invariance].unsqueeze(-1)  # [num_rays, latent_dim, 1]

            # Innerproduct between projection of Z and D on the plane orthogonal to the axis of invariance.
            # This encodes the rotational information. This is rotation-equivariant to rotations of either Z
            # or D and is invariant to rotations of both Z and D.
            innerprod = (z_other * d_other).sum(dim=-1)  # [num_rays, latent_dim]

            # Compute norm along the axes orthogonal to the axis of invariance
            d_other_norm = torch.sqrt(D[::, other_axes[0]] ** 2 + D[:, other_axes[1]] ** 2).unsqueeze(
                -1
            )  # [num_rays, 1]

            # Get invariant component of D along the axis of invariance
            d_invar = D[:, axis_of_invariance].unsqueeze(-1)  # [num_rays, 1]

            directional_input = torch.cat((innerprod, d_invar, d_other_norm), 1)  # [num_rays, latent_dim + 2]
            conditioning_input = torch.cat((z_other_invar, z_invar), dim=-1).flatten(1)  # [num_rays, latent_dim * 3]

            return directional_input, conditioning_input

        if equivariance == "SO3":
            z = self.vn_proj_in(Z)  # [num_rays, latent_dim, 1, 3]
            z_invar = self.vn_invar(z)  # [num_rays, latent_dim, 3]
            conditioning_input = z_invar.flatten(1)  # [num_rays, latent_dim * 3]
            innerprod = torch.sum(Z * D.unsqueeze(1), dim=-1)  # [num_rays, latent_dim]
            return innerprod, z_invar

    def gram_matrix_invariance(
        self,
        Z,
        D,
        equivariance: Literal["None", "SO2", "SO3"] = "SO2",
        axis_of_invariance: int = 1,
    ):
        """Generates an invariant representation from latent code Z and direction coordinates D.

        Args:
            Z (torch.Tensor): Latent code (num_rays x latent_dim x 3)
            D (torch.Tensor): Direction coordinates (num_rays x 3)
            equivariance (str): Type of equivariance to use. Options are 'none', 'SO2', and 'SO3'
            axis_of_invariance (int): The axis of rotation invariance. Should be 0 (x-axis), 1 (y-axis), or 2 (z-axis).
                Default is 1 (y-axis).
        Returns:
            torch.Tensor: Invariant representation
        """
        assert 0 <= axis_of_invariance < 3, "axis_of_invariance should be 0, 1, or 2."
        other_axes = [i for i in range(3) if i != axis_of_invariance]

        if equivariance == "None":
            # get inner product between latent code and direction coordinates
            innerprod = torch.sum(Z * D.unsqueeze(1), dim=-1)  # [num_rays, latent_dim]
            z_input = Z.flatten(start_dim=1)  # [num_rays, latent_dim * 3]
            return innerprod, z_input

        if equivariance == "SO2":
            # Select components along axes orthogonal to the axis of invariance
            z_other = torch.stack((Z[:, :, other_axes[0]], Z[:, :, other_axes[1]]), -1)  # [num_rays, latent_dim, 2]
            d_other = torch.stack((D[:, other_axes[0]], D[:, other_axes[1]]), -1).unsqueeze(1)  # [num_rays, 2]
            d_other = d_other.expand_as(z_other)  # size becomes [num_rays, latent_dim, 2]

            # Invariant representation of Z, gram matrix G=Z*Z' is size num_rays x latent_dim x latent_dim
            G = torch.bmm(z_other, torch.transpose(z_other, 1, 2))

            # Flatten G to be size num_rays x latent_dim^2
            z_other_invar = G.flatten(start_dim=1)

            # Get invariant component of Z along the axis of invariance
            z_invar = Z[:, :, axis_of_invariance]  # [num_rays, latent_dim]

            # Innerprod is size num_rays x latent_dim
            innerprod = (z_other * d_other).sum(dim=-1)  # [num_rays, latent_dim]

            # Compute norm along the axes orthogonal to the axis of invariance
            d_other_norm = torch.sqrt(D[::, other_axes[0]] ** 2 + D[:, other_axes[1]] ** 2).unsqueeze(
                -1
            )  # [num_rays, 1]

            # Get invariant component of D along the axis of invariance
            d_invar = D[:, axis_of_invariance].unsqueeze(-1)  # [num_rays, 1]

            if not self.old_implementation:
                directional_input = torch.cat((innerprod, d_invar, d_other_norm), 1)  # [num_rays, latent_dim + 2]
                conditioning_input = torch.cat((z_other_invar, z_invar), 1)  # [num_rays, latent_dim^2 + latent_dim]
            else:
                # this is matching the previous implementation of RENI, needed if using old checkpoints
                return torch.cat((innerprod, z_other_invar, d_other_norm, z_invar, d_invar), 1)

            return directional_input, conditioning_input

        if equivariance == "SO3":
            G = Z @ torch.transpose(Z, 1, 2)  # [num_rays, latent_dim, latent_dim]
            innerprod = torch.sum(Z * D.unsqueeze(1), dim=-1)  # [num_rays, latent_dim]
            z_invar = G.flatten(start_dim=1)  # [num_rays, latent_dim^2]
            return innerprod, z_invar

    def setup_network(self):
        """Sets up the network architecture"""
        base_input_dims = {
            "VN": {
                "None": {"direction": self.latent_dim, "conditioning": self.latent_dim * 3},
                "SO2": {"direction": self.latent_dim + 2, "conditioning": self.latent_dim * 3},
                "SO3": {"direction": self.latent_dim, "conditioning": self.latent_dim * 3},
            },
            "GramMatrix": {
                "None": {"direction": self.latent_dim, "conditioning": self.latent_dim * 3},
                "SO2": {"direction": self.latent_dim + 2, "conditioning": self.latent_dim**2 + self.latent_dim},
                "SO3": {"direction": self.latent_dim, "conditioning": self.latent_dim**2},
            },
        }

        # Extract the necessary input dimensions
        input_types = ["direction", "conditioning"]
        input_dims = {
            key: base_input_dims[self.config.invariant_function][self.config.equivariance][key] for key in input_types
        }

        # Helper function to create NeRF encoding
        def create_nerf_encoding(in_dim):
            return NeRFEncoding(
                in_dim=in_dim, num_frequencies=2, min_freq_exp=0.0, max_freq_exp=2.0, include_input=True
            )

        # Dictionary-based encoding setup
        encoding_setup = {
            "None": [],
            "Conditioning": ["conditioning"],
            "Directions": ["direction"],
            "Both": ["direction", "conditioning"],
        }

        # Setting up the required encodings
        for input_type in encoding_setup.get(self.config.encoded_input, []):
            # create self.{input_type}_encoding and update input_dims
            setattr(self, f"{input_type}_encoding", create_nerf_encoding(input_dims[input_type]))
            input_dims[input_type] = getattr(self, f"{input_type}_encoding").get_out_dim()

        output_activation = None
        if self.config.output_activation == "exp":
            output_activation = ExpLayer()
        elif self.config.output_activation == "sigmoid":
            output_activation = nn.Sigmoid()
        elif self.config.output_activation == "tanh":
            output_activation = nn.Tanh()
        elif self.config.output_activation == "relu":
            output_activation = nn.ReLU()

        network = None
        if self.conditioning == "Concat":
            network = Siren(
                in_dim=input_dims["direction"] + input_dims["conditioning"],
                hidden_layers=self.hidden_layers,
                hidden_features=self.hidden_features,
                out_dim=self.out_features,
                outermost_linear=self.last_layer_linear,
                first_omega_0=self.first_omega_0,
                hidden_omega_0=self.hidden_omega_0,
                out_activation=output_activation,
            )
        elif self.conditioning == "FiLM":
            network = FiLMSiren(
                in_dim=input_dims["direction"],
                hidden_layers=self.hidden_layers,
                hidden_features=self.hidden_features,
                mapping_network_in_dim=input_dims["conditioning"],
                mapping_network_layers=self.mapping_layers,
                mapping_network_features=self.mapping_features,
                out_dim=self.out_features,
                outermost_linear=True,
                out_activation=output_activation,
            )
        elif self.conditioning == "Attention":
            # transformer where K, V is from conditioning input and Q is from pos encoded directional input
            network = Decoder(
                in_dim=input_dims["direction"],
                conditioning_input_dim=input_dims["conditioning"],
                hidden_features=self.config.hidden_features,
                num_heads=self.config.num_attention_heads,
                num_layers=self.config.num_attention_layers,
                out_activation=output_activation,
            )
        assert network is not None, "unknown conditioning type"
        return network

    def sample_latent(self, idx):
        """Sample the latent code at a given index

        Args:
        idx (int): Index of the latent variable to sample

        Returns:
        tuple (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing the sampled latent variable, the mean of the latent variable and the log variance of the latent variable
        """

        if self.training and not self.fixed_decoder:
            # use reparameterization trick
            mu = self.train_mu[idx, :, :]
            log_var = self.train_logvar[idx, :, :]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            # just sample from the mean
            sample = self.eval_mu[idx, :, :]
            mu = self.eval_mu[idx, :, :]
            log_var = self.eval_logvar[idx, :, :]

        return sample, mu, log_var

    def select_scale(self):
        """Selects the scale to use for the network"""

        scale = None
        if self.training and not self.fixed_decoder:
            if self.config.trainable_scale in [True, "train", "both"]:
                scale = self.train_scale
        else:
            if self.config.trainable_scale in [True, "eval", "both"]:
                scale = self.eval_scale

        return scale

    def init_latent_codes(self, num_latents: int, mode: Literal["train", "eval"]):
        """Initializes the latent codes"""
        log_var = torch.nn.Parameter(torch.normal(-5, 1, size=(num_latents, self.latent_dim, 3)))

        if mode == "eval":
            # init latents as all zeros for eval (the mean of the prior)
            log_var.requires_grad = False
            mu = torch.nn.Parameter(torch.zeros(num_latents, self.latent_dim, 3))
        else:
            mu = torch.nn.Parameter(torch.randn(num_latents, self.latent_dim, 3))

        return mu, log_var

    def reset_eval_latents(self):
        """Resets the eval latents"""
        eval_mu, eval_logvar = self.init_latent_codes(self.num_eval_data, "eval")
        eval_mu = eval_mu.type_as(self.eval_mu)
        eval_logvar = eval_logvar.type_as(self.eval_logvar)
        self.eval_mu.data = eval_mu.data
        self.eval_logvar.data = eval_logvar.data

        if self.config.trainable_scale in [True, "eval", "both"]:
            eval_scale = torch.ones(self.num_eval_data).type_as(self.eval_scale)
            self.eval_scale.data = eval_scale.data

    def apply_positional_encoding(self, directional_input, conditioning_input):
        # conditioning on just invariant directional input
        if self.config.encoded_input == "Conditioning":
            conditioning_input = self.conditioning_encoding(conditioning_input)  # [num_rays, embedding_dim]
        elif self.config.encoded_input == "Directions":
            directional_input = self.direction_encoding(directional_input)  # [num_rays, embedding_dim]
        elif self.config.encoded_input == "Both":
            directional_input = self.dirction_encoding(directional_input)
            conditioning_input = self.conditioning_encoding(conditioning_input)

        return directional_input, conditioning_input

    def get_outputs(
        self,
        ray_samples: RaySamples,
        rotation: Optional[torch.Tensor] = None,
        latent_codes: Optional[torch.Tensor] = None,
    ) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
            latent_codes: [num_rays, latent_dim, 3]
        """
        # we want to batch over camera_indices as these correspond to unique latent codes
        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays]

        if latent_codes is None:
            latent_codes, mu, log_var = self.sample_latent(camera_indices)  # [num_rays, latent_dim, 3]
        else:
            mu = None
            log_var = None

        if rotation is not None:
            latent_codes = torch.matmul(latent_codes, rotation)

        directions = (
            ray_samples.frustums.directions
        )  # [num_rays, 3] # each has unique latent code defined by camera index

        if not self.old_implementation:
            directional_input, conditioning_input = self.invariant_function(
                latent_codes, directions, equivariance=self.equivariance, axis_of_invariance=self.axis_of_invariance
            )  # [num_rays, 3]

            if self.config.positional_encoding == "NeRF":
                directional_input, conditioning_input = self.apply_positional_encoding(
                    directional_input, conditioning_input
                )

            if self.conditioning == "Concat":
                model_outputs = self.network(
                    torch.cat((directional_input, conditioning_input), dim=1)
                )  # returns -> [num_rays, 3]
            elif self.conditioning == "FiLM":
                model_outputs = self.network(directional_input, conditioning_input)  # returns -> [num_rays, 3]
            elif self.conditioning == "Attention":
                model_outputs = self.network(directional_input, conditioning_input)  # returns -> [num_rays, 3]
        else:
            # in the old implementation directions were sampled with y-up not z-up so need to swap y and z in directions
            directions = torch.stack((directions[:, 0], directions[:, 2], directions[:, 1]), -1)
            model_input = self.invariant_function(
                latent_codes, directions, equivariance=self.equivariance, axis_of_invariance=self.axis_of_invariance
            )  # [num_rays, 3]

            model_outputs = self.network(model_input)

        outputs = {}

        scale = self.select_scale()
        if scale is not None:
            scales = scale[camera_indices]  # [num_rays]
            scales = torch.exp(scales)  # [num_rays] exp to ensure positive

            if self.log_domain:
                model_outputs = model_outputs + torch.log(scales.unsqueeze(1))  # [num_rays, 3]
            else:
                model_outputs = model_outputs * scales.unsqueeze(1)  # [num_rays, 3]

        outputs[RENIFieldHeadNames.RGB] = model_outputs
        outputs[RENIFieldHeadNames.MU] = mu
        outputs[RENIFieldHeadNames.LOG_VAR] = log_var

        return outputs

    def forward(
        self,
        ray_samples: RaySamples,
        rotation: Optional[torch.Tensor] = None,
        latent_codes: Optional[torch.Tensor] = None,
    ) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
            latent_codes: [num_rays, latent_dim, 3]

        Returns:
            Dict[RENIFieldHeadNames, TensorType]: A dictionary containing the outputs of the field.
        """
        return self.get_outputs(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)
