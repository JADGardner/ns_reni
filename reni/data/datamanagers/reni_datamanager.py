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
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.pixel_samplers import (
    PixelSampler,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
    variable_res_collate,
    TDataset,
)
from nerfstudio.utils.rich_utils import CONSOLE

from reni.data.datasets.reni_dataset import RENIDataset


@dataclass
class RENIDataManagerConfig(VanillaDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: RENIDataManager)
    """Target class to instantiate."""


class RENIDataManager(VanillaDataManager):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RENIDataManagerConfig
    train_dataset: RENIDataset
    eval_dataset: RENIDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: RENIDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")

        self.num_train = len(self.train_dataset)
        self.num_eval = len(self.eval_dataset)

        self.image_height = self.train_dataset.metadata["image_height"]
        self.image_width = self.train_dataset.metadata["image_width"]

        # add batch_size to metadata
        self.train_dataset.metadata["batch_size"] = self.config.pixel_sampler.images_per_batch

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break

        super(VanillaDataManager, self).__init__()  # Call grandparent class constructor ignoring parent class

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
    
    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int, **kwargs) -> PixelSampler:
        """Infer pixel sampler to use."""
        return self.config.pixel_sampler.setup(is_equirectangular=True, num_rays_per_batch=num_rays_per_batch, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        # NOTE: Updated to RENI sampler where full image per batch is possible
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))
        self.train_ray_generator.image_coords = self.train_ray_generator.image_coords.to(self.device)

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_image_pixel_sampler = self._get_pixel_sampler(
            self.eval_dataset,
            self.config.eval_num_rays_per_batch,
            full_image_per_batch=True,
            images_per_batch=1,
        )
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        self.eval_ray_generator.image_coords = self.eval_ray_generator.image_coords.to(self.device)

        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def stack_ray_bundle(self, ray_bundle: RayBundle, num_rays: int) -> RayBundle:
        ray_bundle.directions = torch.stack(
            [ray_bundle.directions[:num_rays], ray_bundle.directions[num_rays:]], dim=0
        )  # [2, N, 3]
        ray_bundle.origins = torch.stack(
            [ray_bundle.origins[:num_rays], ray_bundle.origins[num_rays:]], dim=0
        )  # [2, N, 3]
        ray_bundle.camera_indices = torch.stack(
            [ray_bundle.camera_indices[:num_rays], ray_bundle.camera_indices[num_rays:]], dim=0
        )  # [2, N, 1]
        return ray_bundle

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)  # [len(train_dataset), H, W, 3]
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]  # [N, 3]
        ray_bundle = self.train_ray_generator(ray_indices)  # [N, 3]
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)  # [len(eval_dataset), H, W, 3]
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, idx: int) -> Tuple[int, RayBundle, Dict]:
        self.eval_count += 1
        idx = idx % len(self.eval_dataset)
        image_batch = self.eval_image_dataloader[idx]  # [H, W, 3]
        image_batch["image_idx"] = torch.tensor([image_batch["image_idx"]])
        image_batch["image"] = image_batch["image"].unsqueeze(0)
        if 'mask' in image_batch:
            image_batch["mask"] = image_batch["mask"].unsqueeze(0)
        assert self.eval_image_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_image_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        image_idx = int(ray_bundle.camera_indices[0, 0])
        return image_idx, ray_bundle, batch

    def create_train_dataset(self) -> RENIDataset:
        """Sets up the data loaders for training"""
        return RENIDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split="train",
        )

    def create_eval_dataset(self) -> RENIDataset:
        """Sets up the data loaders for evaluation"""
        return RENIDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            min_max=self.train_dataset.metadata["min_max"],
            split=self.test_split,
        )
