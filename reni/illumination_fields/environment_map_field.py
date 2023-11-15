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

"""Environment Map"""

from typing import Type, Union, Dict, Union, Tuple, Optional, Any
from dataclasses import dataclass, field
import contextlib
from pathlib import Path
import imageio
import math
import numpy.typing as npt

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples

from reni.illumination_fields.base_spherical_field import SphericalField, SphericalFieldConfig, BaseRENIFieldConfig, BaseRENIField
from reni.field_components.field_heads import RENIFieldHeadNames


@dataclass
class EnvironmentMapFieldConfig(BaseRENIFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: EnvironmentMapField)
    """target class to instantiate"""
    path: Path = Path("path/to/environment_map.exr")
    """path to environment map"""
    resolution: Tuple[int, int] = (512, 1024)
    """resolution of environment map"""
    trainable: bool = False
    """whether to train the environment map or not"""
    apply_padding: bool = True
    """whether to apply padding to the environment map or not"""


class EnvironmentMapField(BaseRENIField):
    """A representation of distant illumination as an environment map."""

    def __init__(
        self,
        config: EnvironmentMapFieldConfig,
        num_train_data: Union[int, None],
        num_eval_data: Union[int, None],
        normalisations: Dict[str, Any],
    ) -> None:
        super().__init__(
            config=config, num_train_data=num_train_data, num_eval_data=num_eval_data, normalisations=normalisations
        )
        self.config = config
        self.num_train_data = num_train_data
        self.num_eval_data = num_eval_data
        self.path = config.path
        self.resolution = [config.resolution[0], config.resolution[1]]
        self.trainable = config.trainable
        self.apply_padding = config.apply_padding
        self.fixed_decoder = config.fixed_decoder

        assert self.resolution[0] == self.resolution[1] // 2, "Environment map must have a 2:1 aspect ratio."

        if self.apply_padding:
            self.resolution[1] += 2  # add padding to the width dimension

        if self.config.trainable:
            self.train_envmaps = nn.Parameter(
                torch.randn(num_train_data, 3, self.resolution[0], self.resolution[1]), requires_grad=True
            )
            self.eval_envmaps = nn.Parameter(
                torch.randn(num_eval_data, 3, self.resolution[0], self.resolution[1]), requires_grad=True
            )
        else:
            # get the image from the path and set both train and eval envmaps to the same image
            image = self.get_numpy_image(self.path)
            image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]

            # Apply padding to the environment maps to avoid artifacts at the borders.
            if self.apply_padding:
                image = image.permute(0, 2, 3, 1)  # Change dimensions to [B, W, H, C]
                image = torch.nn.functional.pad(
                    image, (0, 0, 1, 1), mode="replicate"  # padding of 1 on either side of width dimension
                )
                image = image.permute(0, 3, 1, 2)  # Change dimensions back to [B, C, H, W+2]

            self.train_envmaps = image
            self.eval_envmaps = image

        # ensure envmap is only 3 channels
        self.train_envmaps = self.train_envmaps[:, :3, :, :]
        self.eval_envmaps = self.eval_envmaps[:, :3, :, :]

        # to match RENI training make train_mu and eval_mu pointers to self.train_envmaps, self.eval_envmaps
        self.train_mu = self.train_envmaps
        self.eval_mu = self.eval_envmaps

    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to fix the decoder weights

        Example usage:
        ```
        with instance_of_RENIField.hold_decoder_fixed():
            # do stuff
        ```
        """
        prev_decoder_state = self.fixed_decoder
        self.fixed_decoder = True
        try:
            yield
        finally:
            self.fixed_decoder = prev_decoder_state

    def get_numpy_image(self, image_path: Path) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image = imageio.imread(image_path).astype("float32")
        # make any inf values equal to max non inf
        image[image == np.inf] = np.nanmax(image[image != np.inf])
        # make any values less than zero equal to min non negative
        image[image <= 0] = np.nanmin(image[image > 0])
        assert np.all(np.isfinite(image)), "Image contains non finite values."
        assert np.all(image >= 0), "Image contains negative values."
        assert len(image.shape) == 3
        assert image.dtype == np.float32
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def cart_to_spherical(self, directions):
        """Converts cartesian coordinates to spherical coordinates."""
        theta = torch.acos(directions[:, 2])  # [num_rays]
        phi = torch.atan2(directions[:, 0], directions[:, 1])  # [num_rays]
        return theta, phi

    def angles_to_map_coords(self, theta, phi, H, W):
        """Converts spherical angles to image coordinates."""
        # normalize to [0, 1)
        u = phi / (2 * math.pi) % 1.0
        v = theta / math.pi

        # map to image coordinates
        x = (W * u).long()
        y = (H * v).long()

        if self.apply_padding:
            # Adjust x for padding
            x = x + 1  # shift by one column to account for padding

        # clamp for safety
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        return x, y

    def bilinear_interpolate(self, envmaps, camera_indices, x, y):
        """
        Performs bilinear interpolation on the image.

        Args:
            envmaps: [num_envmaps, 3, H, W]
            camera_indices: [num_rays]
            x: [num_rays]
            y: [num_rays]
        """
        H, W = envmaps.shape[-2:]
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, W - 2)
        x1 = torch.clamp(x1, 0, W - 1)
        y0 = torch.clamp(y0, 0, H - 2)
        y1 = torch.clamp(y1, 0, H - 1)

        # sample the envmap associated with each ray
        Ia = envmaps[camera_indices, :, y0, x0]
        Ib = envmaps[camera_indices, :, y1, x0]
        Ic = envmaps[camera_indices, :, y0, x1]
        Id = envmaps[camera_indices, :, y1, x1]

        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wc = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))

        return wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

    def get_outputs(
        self, ray_samples: RaySamples, rotation: Optional[torch.Tensor]= None, envmaps: Optional[torch.Tensor] = None
    ) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
        """
        # we want to batch over camera_indices as these correspond to unique latent codes
        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays]
        if not self.trainable:
            camera_indices = torch.zeros_like(camera_indices)  # [num_rays]

        if envmaps is not None:
            # we are using a single pre-provided environment map for all cameras
            envmaps = envmaps
        else:
            if self.trainable:
                # we are optimising envmap pixels for each camera
                if self.training and not self.fixed_decoder:
                    envmaps = self.train_envmaps
                else:
                    envmaps = self.eval_envmaps
            else:
                # we are using a single pre-provided environment map for all cameras
                envmaps = self.eval_envmaps  # [1, 3, H, W]

        directions = (
            ray_samples.frustums.directions
        )  # [num_rays, 3] # each has unique latent code defined by camera index

        if rotation is not None:
            # apply rotation to directions
            rotation = rotation.T
            directions = torch.matmul(ray_bundle.directions, rotation)  # [num_rays, 3]

        theta, phi = self.cart_to_spherical(directions)  # [num_rays], [num_rays]
        x, y = self.angles_to_map_coords(theta, phi, envmaps.shape[-2], envmaps.shape[-1])  # [num_rays], [num_rays]
        samples = self.bilinear_interpolate(envmaps, camera_indices, x, y)  # [num_rays, 3]

        outputs = {
            RENIFieldHeadNames.RGB: samples,
        }

        return outputs

    def forward(
        self, ray_samples: RaySamples, rotation: Optional[torch.Tensor]= None, envmaps: Optional[torch.Tensor] = None
    ) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
            envmaps: [num_envmaps, 3, H, W]

        Returns:
            Dict[RENIFieldHeadNames, TensorType]: A dictionary containing the outputs of the field.
        """
        return self.get_outputs(ray_samples=ray_samples, rotation=rotation, envmaps=envmaps)
