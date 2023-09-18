# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Code for sampling pixels.
"""

from typing import Dict, Optional, Union, Type

import torch
from jaxtyping import Int
from dataclasses import dataclass, field
from torch import Tensor

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig


@dataclass
class RENIEquirectangularPixelSamplerConfig(PixelSamplerConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: RENIEquirectangularPixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    full_image_per_batch: bool = False
    """Whether or not to sample the full image."""
    images_per_batch: int = 1
    """Number of images to sample per batch."""


class RENIEquirectangularPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(
        self,
        config: RENIEquirectangularPixelSamplerConfig,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.full_image_per_batch = kwargs.get("full_image_per_batch", self.config.full_image_per_batch)
        self.images_per_batch = kwargs.get("images_per_batch", self.config.images_per_batch)

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if self.full_image_per_batch:
            # Randomly select N images from the range [0, num_images)
            random_image_indices = torch.randint(0, num_images, (self.images_per_batch,), device=device)  # [N]
            # Create a grid of index values for height and width
            phi_values = torch.arange(0, image_height, device=device)  # [H]
            theta_values = torch.arange(0, image_width, device=device)  # [W]

            # Create a meshgrid to combine phi and theta for all points in the image
            phi_grid, theta_grid = torch.meshgrid(phi_values, theta_values)  # [H, W]

            # Repeat the grid for each image in the batch
            phi_grid = phi_grid.repeat(num_images, 1, 1)  # [N, H, W]
            theta_grid = theta_grid.repeat(num_images, 1, 1)  # [N, H, W]

            # Create a tensor for the selected random image indices
            image_indices = random_image_indices.view(-1, 1, 1)  # [N, 1, 1]
            image_indices = image_indices.repeat(1, image_height, image_width)  # [N, H, W]

            # Stack the random image indices, phi, and theta to create the final indices tensor
            indices = torch.stack((image_indices, phi_grid, theta_grid), dim=-1)  # [N, H, W, 3]

            indices = indices.view(-1, 3)
            indices = indices.long()

        else:
            indices = super().sample_method_equirectangular(
                batch_size, num_images, image_height, image_width, mask=mask, device=device
            )

        return indices

    def collate_image_dataset_batch(
        self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, indices: Optional[Tensor] = None
    ):
        """
        This has been modified to allow resampling with the a set of indices from the same cameras.
        This is used to sample a set of directions rotated around the vertical axis and then
        used in RENI as a gradient based loss where the gradient is approximated using finite-difference.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
            indices: indices to sample from
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if indices is None:
            if "mask" in batch:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        if self.full_image_per_batch:
            assert collated_batch["image"].shape[0] == image_height * image_width * self.images_per_batch
        else:
            assert collated_batch["image"].shape[0] == num_rays_per_batch

        # # Needed to correct the random indices to their actual camera idx locations.
        collated_batch["sampled_idxs"] = indices[:, 0].clone()  # for calling again with the same indices
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict, keep_full_image: Optional[bool] = None, indices: Optional[Tensor] = None):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if keep_full_image is None:
            keep_full_image = self.config.keep_full_image

        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=keep_full_image, indices=indices
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch
