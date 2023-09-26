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
from typing import Type, Union, Dict, Any, Tuple, Optional
import contextlib

import numpy as np
import torch
from torchtyping import TensorType

from reni.illumination_fields.base_spherical_field import BaseRENIField, BaseRENIFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames

from nerfstudio.cameras.rays import RaySamples


# Field related configs
@dataclass
class SphericalGaussianFieldConfig(BaseRENIFieldConfig):
    """Configuration for Spherical Gaussian instantiation"""

    _target: Type = field(default_factory=lambda: SphericalGaussianField)
    """target class to instantiate"""
    row_col_gaussian_dims: Tuple[int, int] = (10, 20)
    """number of gaussian components in row and column directions"""
    channel_dim: int = 3
    """number of channels in the field"""
    eval_train_equal: bool = True
    """whether to use the same latents for eval and train"""


class SphericalGaussianField(BaseRENIField):
    """Spherical Gaussian Illumination Field.
    https://github.com/lzqsd/SphericalGaussianOptimization
    """

    def __init__(
        self,
        config: SphericalGaussianFieldConfig,
        num_train_data: Union[int, None],
        num_eval_data: Union[int, None],
        normalisations: Dict[str, Any],
    ) -> None:
        super().__init__(
            config=config, num_train_data=num_train_data, num_eval_data=num_eval_data, normalisations=normalisations
        )

        self.sg_col, self.sg_row = self.config.row_col_gaussian_dims
        self.sg_num = int(self.sg_row * self.sg_col)

        phi_center = ((np.arange(self.sg_col) + 0.5) / self.sg_col - 0.5) * np.pi * 2
        theta_center = (np.arange(self.sg_row) + 0.5) / self.sg_row * np.pi / 2.0

        phi_center, theta_center = np.meshgrid(phi_center, theta_center)
        theta_center = theta_center.reshape(1, self.sg_num, 1).astype(np.float32)
        phi_center = phi_center.reshape(1, self.sg_num, 1).astype(np.float32)

        theta_center = torch.from_numpy(theta_center).requires_grad_(False)
        phi_center = torch.from_numpy(phi_center).requires_grad_(False)

        theta_center_train = theta_center.expand([self.num_train_data, self.sg_num, 1])
        phi_center_train = phi_center.expand([self.num_train_data, self.sg_num, 1])
        theta_center_eval = theta_center.expand([self.num_eval_data, self.sg_num, 1])
        phi_center_eval = phi_center.expand([self.num_eval_data, self.sg_num, 1])

        # register all as buffers
        self.register_buffer("theta_center_train", theta_center_train)
        self.register_buffer("phi_center_train", phi_center_train)
        self.register_buffer("theta_center_eval", theta_center_eval)
        self.register_buffer("phi_center_eval", phi_center_eval)

        theta_range = (np.pi / 2 / self.sg_row) * 1.5
        phi_range = (2 * np.pi / self.sg_col) * 1.5

        self.register_buffer("theta_range", torch.tensor(theta_range))
        self.register_buffer("phi_range", torch.tensor(phi_range))

        train_params = self.setup_param(self.num_train_data, self.sg_num, self.sg_row, self.config.channel_dim)
        self.register_parameter("train_params", train_params)
        eval_params = self.setup_param(self.num_eval_data, self.sg_num, self.sg_row, self.config.channel_dim)
        self.register_parameter("eval_params", eval_params)

    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to match RENI implementation.

        Example usage:
        ```
        with instance_of_SpheriacalGaussianField.hold_decoder_fixed():
            # do stuff
        ```
        """
        prev_decoder_state = self.fixed_decoder
        self.fixed_decoder = True
        try:
            yield
        finally:
            self.fixed_decoder = prev_decoder_state

    def setup_param(self, num_latents, sg_num, sg_row, channel_dim):
        weight = torch.zeros((num_latents, sg_num, channel_dim), dtype=torch.float32)
        theta = torch.zeros((num_latents, sg_num, 1), dtype=torch.float32)
        phi = torch.zeros((num_latents, sg_num, 1), dtype=torch.float32)
        lamb = torch.log(torch.ones(num_latents, sg_num, 1) * np.pi / sg_row)

        params = torch.cat([weight, theta, phi, lamb], dim=2)
        params = torch.nn.Parameter(params.view(num_latents, sg_num * 6))

        return params

    def reset_eval_latents(self):
        """Resets the eval latents"""
        eval_params = self.setup_param(self.num_eval_data, self.sg_num, self.sg_row, self.config.channel_dim).type_as(
            self.eval_params
        )
        self.eval_params.data = eval_params.data

    def copy_train_to_eval(self):
        """Copies the train latents to eval latents"""
        self.eval_params.data = self.train_params.data
        self.phi_center_eval.data = self.phi_center_train.data
        self.theta_center_eval.data = self.theta_center_train.data

    def deparameterize(self):
        """Get individual parameters from the concatenated parameters"""
        if self.training and not self.fixed_decoder:
            weight, theta, phi, lamb = torch.split(
                self.train_params.view(self.num_train_data, self.sg_num, 6), [3, 1, 1, 1], dim=2
            )
            theta_center = self.theta_center_train
            phi_center = self.phi_center_train
        else:
            weight, theta, phi, lamb = torch.split(
                self.eval_params.view(self.num_eval_data, self.sg_num, 6), [3, 1, 1, 1], dim=2
            )
            theta_center = self.theta_center_eval
            phi_center = self.phi_center_eval

        theta = self.theta_range * torch.tanh(theta) + theta_center
        phi = self.phi_range * torch.tanh(phi) + phi_center
        weight = torch.exp(weight)
        lamb = torch.exp(lamb)
        return theta, phi, weight, lamb

    def renderSG(self, directions, camera_indices, theta, phi, lamb, weight):
        """Get the rendered spherical gaussian values for the given directions

        Args:
            directions (torch.Tensor): [num_rays, 3] tensor of ray directions
            camera_indices (torch.Tensor): [num_rays] tensor of camera indices
            theta (torch.Tensor): [num_train/eval_data, sg_num, 1] tensor of Elevation values
            phi (torch.Tensor): [num_train/eval_data, sg_num, 1] tensor of Azimuth values
            lamb (torch.Tensor): [num_train/eval_data, sg_num, 1] tensor of Sharpness values
            weight (torch.Tensor): [num_train/eval_data, sg_num, 3] tensor of Amplitude values
        """

        theta_dir = torch.acos(directions[:, 2])  # [num_rays]
        phi_dir = torch.atan2(directions[:, 0], directions[:, 1])  # [num_rays]

        # Unsqueeze directions to match the shape of theta, phi, lamb and weight
        theta_dir = theta_dir.unsqueeze(-1).unsqueeze(-1)  # [num_rays, 1, 1]
        phi_dir = phi_dir.unsqueeze(-1).unsqueeze(-1)  # [num_rays, 1, 1]

        # Computing spherical gaussian as per the formula
        cos_angle = torch.sin(theta[camera_indices]) * torch.sin(theta_dir) * torch.cos(
            phi[camera_indices] - phi_dir
        ) + torch.cos(theta[camera_indices]) * torch.cos(
            theta_dir
        )  # [num_rays, sg_num, 1]

        # index select the weight for each ray
        rgb = weight[camera_indices] * torch.exp(lamb[camera_indices] * (cos_angle - 1))  # [num_rays, sg_num, 3]

        # Sum over all spherical gaussians for each ray
        rgb = torch.sum(rgb, dim=1)

        return rgb

    @abstractmethod
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
        """
        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays]

        directions = (
            ray_samples.frustums.directions
        )  # [num_rays, 3] # each has unique latent code defined by camera index

        if latent_codes is None:
            theta, phi, weight, lamb = self.deparameterize()
        else:
            raise NotImplementedError

        rgb = self.renderSG(directions, camera_indices, theta, phi, lamb, weight)  # [num_rays, 3]

        return {RENIFieldHeadNames.RGB: rgb}

    def forward(
        self,
        ray_samples: RaySamples,
        rotation: Optional[torch.Tensor] = None,
        latent_codes: Optional[torch.Tensor] = None,
    ) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_bundle: [num_rays]
            rotation: [3, 3]
        """
        return self.get_outputs(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)
