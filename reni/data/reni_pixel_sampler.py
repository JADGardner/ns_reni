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

from typing import Dict, Optional, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler


class RENIEquirectangularPixelSampler(EquirectangularPixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, sample_full_image: bool = False, images_per_batch: int = 1):
        super().__init__(num_rays_per_batch, keep_full_image)
        self.sample_full_image = sample_full_image
        self.images_per_batch = images_per_batch

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if self.sample_full_image:
            # Randomly select N images from the range [0, num_images)
            random_image_indices = torch.randint(0, num_images, (self.images_per_batch,), device=device)
            # Creating a grid for phi and theta
            phi_values = torch.linspace(0, 1, image_height, device=device) # Uniformly distributed values for phi
            theta_values = torch.linspace(0, 1, image_width, device=device) # Uniformly distributed values for theta
            # Transforming phi values according to the PDF f(phi) = sin(phi) / 2
            phi_values = torch.acos(1 - 2 * phi_values) / torch.pi
            # Creating a meshgrid to combine phi and theta for all points in the image
            phi_grid, theta_grid = torch.meshgrid(phi_values, theta_values, indexing='ij')
            # Repeating the grid for each image in the batch
            phi_grid = phi_grid.repeat(self.images_per_batch, 1, 1)
            theta_grid = theta_grid.repeat(self.images_per_batch, 1, 1)
            # Creating a tensor for the selected random image indices
            image_indices = random_image_indices.view(-1, 1, 1)
            image_indices = image_indices.repeat(1, image_height, image_width)
            # Stacking the random image indices, phi, and theta to create the final indices tensor
            indices = torch.stack((image_indices, phi_grid, theta_grid), dim=-1)
            # Scaling by the actual dimensions, note that for the image index, we don't need to scale
            indices = indices * torch.tensor([1, image_height - 1, image_width - 1], device=device)
            indices = indices.view(-1, 3)
            indices = indices.long()
        else:
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)

        return indices
    
    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, indices: Optional[Tensor] = None):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if indices is None:
            if "mask" in batch:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        if self.sample_full_image:
            assert collated_batch["image"].shape[0] == image_height * image_width * self.images_per_batch
        else:
            assert collated_batch["image"].shape[0] == num_rays_per_batch

        # # Needed to correct the random indices to their actual camera idx locations.
        collated_batch["sampled_idxs"] = indices[:, 0].clone() # for calling again with the same indices
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
            keep_full_image = self.keep_full_image

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