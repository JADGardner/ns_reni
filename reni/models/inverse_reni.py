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
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import ModelConfig, Model

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.model_components.illumination_samplers import IlluminationSamplerConfig
from reni.model_components.shaders import LambertianShader
from reni.utils.utils import find_nerfstudio_project_root
from reni.fields.nerfacto_reni import NerfactoFieldRENI
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.renderers import RGBLambertianRenderer


# Model related configs
@dataclass
class InverseRENIConfig(ModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: InverseRENI)
    """target class to instantiate"""
    illumination_field: SphericalFieldConfig = SphericalFieldConfig()
    """Illumination Field"""
    illumination_field_ckpt_path: Path = Path("/path/to/ckpt.pt")
    """Path of pretrained illumination field"""
    illumination_field_ckpt_step: int = 0
    """Step of pretrained illumination field"""
    illumination_sampler: IlluminationSamplerConfig = IlluminationSamplerConfig()
    """Illumination sampler to use"""


class InverseRENI(Model):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: InverseRENIConfig

    def __init__(
        self,
        config: InverseRENIConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__(config=config, scene_box=scene_box, num_train_data=num_train_data, **kwargs)

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()

        self.illumination_sampler = self.config.illumination_sampler.setup()

        self.illumination_field = self.config.illumination_field.setup(
            num_train_data=None,
            num_eval_data=self.num_train_data,
        )

        if isinstance(self.config.illumination_field, RENIFieldConfig):
            # # Now you can use this to construct paths:
            project_root = find_nerfstudio_project_root(Path(__file__))
            relative_path = (
                self.config.illumination_field_ckpt_path
                / "nerfstudio_models"
                / f"step-{self.config.illumination_field_ckpt_step:09d}.ckpt"
            )
            ckpt_path = project_root / relative_path

            if not ckpt_path.exists():
                raise ValueError(f"Could not find illumination field checkpoint at {ckpt_path}")

            ckpt = torch.load(str(ckpt_path))
            illumination_field_dict = {}
            match_str = "_model.field."
            ignore_strs = [
                "_model.field.train_logvar",
                "_model.field.eval_logvar",
                "_model.field.train_mu",
                "_model.field.eval_mu",
            ]
            for key in ckpt["pipeline"].keys():
                if key.startswith(match_str) and not any([ignore_str in key for ignore_str in ignore_strs]):
                    illumination_field_dict[key[len(match_str) :]] = ckpt["pipeline"][key]

            # load weights of the decoder
            self.illumination_field.load_state_dict(illumination_field_dict, strict=False)

        self.lambertian_shader = LambertianShader()
        self.labmertian_renderer = RGBLambertianRenderer()

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        param_groups = super().get_param_groups()
        param_groups["illumination_field"] = list(self.illumination_field.parameters())
        return param_groups

    def get_illumination_shader(self, camera_indices: torch.Tensor):
        """Generate samples and sample illumination field"""
        illumination_ray_samples = self.illumination_sampler()  # [num_illumination_directions, 3]
        illumination_ray_samples = illumination_ray_samples.to(self.device)
        camera_indices_for_uniqueness = camera_indices[:, 0]  # [num_rays, sampels_per_ray] -> [num_rays]
        unique_indices, inverse_indices = torch.unique(camera_indices_for_uniqueness, return_inverse=True)
        # unique_indices: [num_unique_camera_indices]
        # inverse_indices: [num_rays]
        num_unique_camera_indices = unique_indices.shape[0]
        num_illumination_directions = illumination_ray_samples.shape[0]

        # Sample from RENI we want to sample all light directions for each camera
        # so shape tensors as appropriate
        unique_indices = unique_indices.unsqueeze(1).expand(
            -1, num_illumination_directions
        )  # [num_unique_camera_indices, num_illumination_directions]
        directions = illumination_ray_samples.frustums.directions.unsqueeze(0).expand(
            num_unique_camera_indices, -1, -1
        )  # [num_unique_camera_indices, num_illumination_directions, 3]
        illumination_ray_samples.camera_indices = unique_indices.reshape(
            -1
        )  # [num_unique_camera_indices * num_illumination_directions]
        illumination_ray_samples.frustums.directions = directions.reshape(
            -1, 3
        )  # [num_unique_camera_indices * num_illumination_directions, 3]
        illuination_field_outputs = self.illumination_field.forward(
            illumination_ray_samples
        )  # [num_unique_camera_indices * num_illumination_directions, 3]
        hdr_illumination_colours = illuination_field_outputs[
            RENIFieldHeadNames.RGB
        ]  # [num_unique_camera_indices * num_illumination_directions, 3]
        hdr_illumination_colours = self.illumination_field.unnormalise(
            hdr_illumination_colours
        )  # [num_unique_camera_indices * num_illumination_directions, 3]
        hdr_illumination_colours = hdr_illumination_colours.reshape(
            num_unique_camera_indices, num_illumination_directions, 3
        )  # [num_unique_camera_indices, num_illumination_directions, 3]
        hdr_illumination_colours = hdr_illumination_colours[
            inverse_indices
        ]  # [num_rays, num_illumination_directions, 3]
        illumination_directions = directions[inverse_indices]  # [num_rays, num_illumination_directions, 3

        return hdr_illumination_colours, illumination_directions

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle, batch: Dict) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        
        light_colors, light_directions = self.get_illumination_shader(ray_bundle.camera_indices)

        lambertian_color_sum, rgb = self.lambertian_shader(
            albedo=albedo,
            normals=pred_normals,
            light_directions=light_directions,
            light_colors=light_colors,
            detach_normals=False,
        )

    def forward(self, ray_bundle: RayBundle, batch: Dict) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, batch)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, batch: Dict) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, batch)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        """Returns the RGBA image from the outputs of the model.

        Args:
            outputs: Outputs of the model.

        Returns:
            RGBA image.
        """
        accumulation_name = output_name.replace("rgb", "accumulation")
        if (
            not hasattr(self, "renderer_rgb")
            or not hasattr(self.renderer_rgb, "background_color")
            or accumulation_name not in outputs
        ):
            raise NotImplementedError(f"get_rgba_image is not implemented for model {self.__class__.__name__}")
        rgb = outputs[output_name]
        if self.renderer_rgb.background_color == "random":  # type: ignore
            acc = outputs[accumulation_name]
            if acc.dim() < rgb.dim():
                acc = acc.unsqueeze(-1)
            return torch.cat((rgb / acc.clamp(min=1e-10), acc), dim=-1)
        return torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
