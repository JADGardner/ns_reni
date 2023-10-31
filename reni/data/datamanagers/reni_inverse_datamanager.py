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
from typing import Literal, Type, Union

import torch

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
    variable_res_collate,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator

from reni.data.datasets.reni_inverse_dataset import RENIInverseDataset


@dataclass
class RENIInverseDataManagerConfig(VanillaDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: RENIInverseDataManager)
    """Target class to instantiate."""
    normals_on_gpu: bool = True
    """Whether to put normals on GPU"""
    albedo_on_gpu: bool = True
    """Whether to put albedo on GPU"""
    specular_on_gpu: bool = True
    """Whether to put specular on GPU"""
    shininess_on_gpu: bool = True
    """Whether to put shininess on GPU"""

class RENIInverseDataManager(VanillaDataManager):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def __init__(
        self,
        config: RENIInverseDataManagerConfig,
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
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        
        # Training and eval are the same as RENI is pre-trained and we are optimising latents only
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=test_mode)

        # Training and eval are the same as RENI is pre-trained and we are optimising latents only
        self.train_dataset = self.create_dataset()
        self.eval_dataset = self.train_dataset
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device

        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")
        if self.config.normals_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("normal")
        if self.config.albedo_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("albedo")
        if self.config.specular_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("specular")
        if self.config.shininess_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("shininess")

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

    def create_dataset(self) -> RENIInverseDataset:
        """Sets up the data loaders for training"""
        return RENIInverseDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split='val'
        )
