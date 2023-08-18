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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from time import time
from typing import Optional, Type

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal


from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig, RENIDataManager
from reni.discriminators.discriminators import BaseDiscriminatorConfig

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline


@dataclass
class RESGANPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: RESGANPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = RENIDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    eval_latent_optimisation_source: Literal["none", "envmap", "image_half", "image_full"] = "image_half"
    """Source for latent optimisation during eval"""
    eval_latent_optimisation_epochs: int = 100
    """Number of epochs to optimise latent during eval"""
    eval_latent_optimisation_lr: float = 0.1
    """Learning rate for latent optimisation during eval"""
    gan_type: Literal["std", "wgan"] = "std",
    """Type of GAN to use"""
    discriminator_train_ratio: int = 5
    """Number of times to train discriminator for each time we train the generator"""


class RESGANPipeline(VanillaPipeline):
    """The pipeline class for Rotation Equivariant Spherical Generative Adverserial Networks.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: RESGANPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super(VanillaPipeline, self).__init__()  # Call grandparent class constructor ignoring parent class
        self.config = config
        self.test_mode = test_mode
        self.datamanager: RENIDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        if test_mode in ["val", "test"]:
            assert self.datamanager.eval_dataset is not None, "Missing validation dataset"

        num_train_data = self.datamanager.num_train
        num_eval_data = self.datamanager.num_eval
        self.batch_size = self.datamanager.config.number_of_images_per_batch

        metadata = self.datamanager.train_dataset.metadata
        metadata['gan_type'] = self.config.gan_type

        self._model = config.model.setup(
            scene_box=None,
            num_train_data=num_train_data,
            num_eval_data=num_eval_data,
            metadata=metadata,
        )
        self.model.to(device)

        self.world_size = world_size

        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.last_step_of_eval_optimisation = 0

    def forward(self):
        """Blank forward method
        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_train_loss_discriminator(self, step: int):
        # Train with all real batch
        # if we are using the scale_inv_loss we are using finite diff so batch will be [2, B*N, 3] otherwise [B*N, 3] This is handled in the model
        ray_bundle, batch = self.datamanager.next_train(step)
        label = torch.full((self.batch_size,), self.model.real_label).type_as(batch['image'])
        discriminator_outputs = self._model.forward_discriminator(ray_bundle, batch['image']) # use _model for data parallel
        loss_dict_real = self.model.get_gan_loss_dict(discriminator_outputs, {'gt_labels': label})
        # now train with all fake batch
        model_outputs = self._model(ray_bundle, batch) # use _model for data parallel
        label.fill_(self.model.fake_label)
        discriminator_outputs = self._model.forward_discriminator(ray_bundle, model_outputs['rgb'])
        loss_dict_fake = self.model.get_gan_loss_dict(discriminator_outputs, {'gt_labels': label})
        loss_dict = {key: loss_dict_real[key] + loss_dict_fake[key] for key in loss_dict_real}
        if self.config.gan_type == 'wgan':
            loss_dict['wgan_loss'] = -loss_dict['wgan_loss']       
        loss_dict = {key: loss_dict_real[key] + loss_dict_fake[key] for key in loss_dict_real}
        return model_outputs, loss_dict

    @profiler.time_function
    def get_train_loss_generator(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        # real label as we want to fool the discriminator
        label = torch.full((self.batch_size,), self.model.real_label).type_as(batch['image'])
        model_outputs = self._model(ray_bundle, batch) # use _model for data parallel
        discriminator_outputs = self._model.forward_discriminator(ray_bundle, model_outputs['rgb'])
        loss_dict = self.model.get_gan_loss_dict(discriminator_outputs, {'gt_labels': label})
        return model_outputs, loss_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.last_step_of_eval_optimisation != step:
            if self.model.config.training_regime != "vae":
                self.model.fit_eval_latents(self.datamanager)
            self.last_step_of_eval_optimisation = step
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.last_step_of_eval_optimisation != step:
            if self.model.config.training_regime != "vae":
                self.model.fit_eval_latents(self.datamanager)
            self.last_step_of_eval_optimisation = step
        image_idx, ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model(ray_bundle, batch)  
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = ray_bundle.directions.shape[-2] # as directions is either [2, N, 3] or [N, 3]
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        if self.last_step_of_eval_optimisation != step:
            if self.model.config.training_regime != "vae":
                self.model.fit_eval_latents(self.datamanager)
            self.last_step_of_eval_optimisation = step
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        # get all eval images

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for idx in range(num_images):
                _, ray_bundle, batch = self.datamanager.next_eval_image(idx=idx)
                num_rays = ray_bundle.directions.shape[-2]
                # time this the following line
                inner_start = time()
                outputs = self.model(ray_bundle, batch)    
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (num_rays)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict
