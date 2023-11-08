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
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import ModelConfig, Model
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.utils import misc
from nerfstudio.utils import colormaps
from nerfstudio.data.scene_box import SceneBox

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.illumination_fields.sg_illumination_field import SphericalGaussianFieldConfig
from reni.illumination_fields.sh_illumination_field import SphericalHarmonicIlluminationFieldConfig
from reni.model_components.illumination_samplers import IlluminationSamplerConfig, EquirectangularSamplerConfig
from reni.utils.utils import find_nerfstudio_project_root
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.shaders import BlinnPhongShader
from reni.utils.colourspace import linear_to_sRGB


# Model related configs
@dataclass
class RENIInverseModelConfig(ModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: RENIInverseModel)
    """target class to instantiate"""
    illumination_field: SphericalFieldConfig = SphericalFieldConfig()
    """Illumination Field"""
    illumination_field_ckpt_path: Path = Path("/path/to/ckpt.pt")
    """Path of pretrained illumination field"""
    illumination_field_ckpt_step: int = 0
    """Step of pretrained illumination field"""
    illumination_sampler: IlluminationSamplerConfig = IlluminationSamplerConfig()
    """Illumination sampler to use"""
    loss_inclusions: Dict[str, Any] = to_immutable_dict(
        {
            "rgb_l1_loss": True,
            "rgb_l2_loss": False,
            "cosine_similarity_loss": True,
        }
    )
    """Losses to include in the loss dict"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_l1_loss": 1.0,
            "rgb_l2_loss": 1.0,
            "cosine_similarity_loss": 1.0,
        }
    )
    """Loss coefficients for each loss"""
    print_nan: bool = False
    """Whether to print nan values in loss dict"""


class RENIInverseModel(Model):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: RENIInverseModelConfig

    def __init__(
        self,
        config: RENIInverseModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__(config=config, scene_box=scene_box, num_train_data=num_train_data, **kwargs)

        self.metadata = kwargs["metadata"]

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        super().populate_modules()

      
        self.illumination_sampler = self.config.illumination_sampler.setup()
        self.equirectangular_sampler = EquirectangularSamplerConfig(width=128).setup()  # for displaying environment map

        if isinstance(self.config.illumination_field, RENIFieldConfig):
            self.illumination_field = self.config.illumination_field.setup(
                num_train_data=None,
                num_eval_data=None,
            )
            # use local latents as illumination field is just for decoder
            self.illumination_latents = torch.nn.Parameter(
                torch.zeros((self.num_train_data, self.illumination_field.latent_dim, 3))
            )
            self.scale = nn.Parameter(torch.ones(self.num_train_data))

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
        elif isinstance(self.config.illumination_field, SphericalHarmonicIlluminationFieldConfig):
            self.illumination_field = self.config.illumination_field.setup(
                num_train_data=1,
                num_eval_data=1,
                normalisations={"min_max": None, "log_domain": True}
            )
            self.illumination_latents = torch.nn.Parameter(
                torch.zeros((self.num_train_data, self.illumination_field.num_sh_coeffs, 3))
            )
            self.scale = None
        elif isinstance(self.config.illumination_field, SphericalGaussianFieldConfig):
            self.illumination_field = self.config.illumination_field.setup(
                num_train_data=1,
                num_eval_data=1,
                normalisations={"min_max": None, "log_domain": True}
            )
            self.illumination_latents = torch.nn.Parameter(
                torch.zeros((self.num_train_data, self.illumination_field.sg_num, 6))
            )
            self.scale = None

        self.blinn_phong_shader = BlinnPhongShader()

        # losses
        if self.config.loss_inclusions["rgb_l1_loss"]:
            self.l1_loss = nn.L1Loss()
        if self.config.loss_inclusions["rgb_l2_loss"]:
            self.l2_loss = nn.MSELoss()
        if self.config.loss_inclusions["cosine_similarity_loss"]:
            self.cosine_similarity = nn.CosineSimilarity(dim=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        param_groups = {}
        if self.scale is not None:
            param_groups["illumination_latents"] = [self.illumination_latents, self.scale]
        else:
            param_groups["illumination_latents"] = [self.illumination_latents]
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

        # get RENI latent codes for either train or eval
        illumination_latents, scale = self.illumination_latents, self.scale
        illumination_latents = illumination_latents[
            illumination_ray_samples.camera_indices
        ]  # [num_rays, latent_dim, 3]

        # if illumiantion field is RENI
        if isinstance(self.illumination_field, RENIField):
            scale = scale[illumination_ray_samples.camera_indices]  # [num_rays]

            illumination_field_outputs = self.illumination_field.forward(
                ray_samples=illumination_ray_samples, latent_codes=illumination_latents, scale=scale
            )  # [num_unique_camera_indices * num_illumination_directions, 3]
        else:
            illumination_field_outputs = self.illumination_field.forward(
                ray_samples=illumination_ray_samples, latent_codes=illumination_latents
            )
        hdr_illumination_colours = illumination_field_outputs[
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

        return hdr_illumination_colours, illumination_directions, unique_indices

    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        # TODO in eval ray bundle is subset from one camera and we need to choose the pixels from tha batch
        q = None
        if batch is not None:
            normals = batch["normal"].to(self.device) # [num_rays, 3]
            albedo = batch["albedo"].to(self.device) # [num_rays, 3]
            specular = batch["specular"].to(self.device) # [num_rays, 3]
            shininess = batch["shininess"].to(self.device) # [num_rays]

            view_directions = ray_bundle.directions.reshape(-1, 3) # num_rays x 3
            
            light_colours, light_directions, unique_indices = self.get_illumination_shader(ray_bundle.camera_indices) # [num_rays, num_illumination_directions, 3]

            rendered_pixels = self.blinn_phong_shader(albedo=albedo,
                                                      normals=normals,
                                                      light_directions=light_directions,
                                                      light_colors=light_colours,
                                                      specular=specular,
                                                      shininess=shininess,
                                                      view_directions=view_directions,
                                                      detach_normals=True)
                    
            outputs = {
                "rgb": rendered_pixels,
                "unique_indices": unique_indices,
            }
        else:
            outputs = {
                "rgb": torch.ones((ray_bundle.origins.shape[0], 3)).to(self.device),
            }

        return outputs

    def forward(self, ray_bundle: RayBundle, batch: Union[Dict, None] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        return self.get_outputs(ray_bundle, batch)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        # add the min max and mean of self.scale
        metrics_dict = {}

        if self.scale is not None:
            metrics_dict["scale_min"] = torch.exp(self.scale).min()
            metrics_dict["scale_max"] = torch.exp(self.scale).max()
            metrics_dict["scale_mean"] = torch.exp(self.scale).mean()

        return metrics_dict

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        rgb = outputs["rgb"]
        image = batch["image"].to(self.device)
        q = batch["q"].unsqueeze(-1).to(self.device)

        rgb = linear_to_sRGB(rgb, q=q, clamp=False)
        image = linear_to_sRGB(image, q=q, clamp=False)

        loss_dict = {}
        if self.config.loss_inclusions["rgb_l1_loss"]:
            loss_dict["rgb_l1_loss"] = self.l1_loss(rgb, image)
        if self.config.loss_inclusions["rgb_l2_loss"]:
            loss_dict["rgb_l2_loss"] = self.l2_loss(rgb, image)
        if self.config.loss_inclusions["cosine_similarity_loss"]:
            similarity = self.cosine_similarity(rgb, image)
            loss_dict["cosine_similarity_loss"] = 1.0 - similarity.mean()
        if self.config.loss_inclusions["prior_loss"]:
            loss_dict["prior_loss"] = torch.mean(torch.square(self.illumination_latents[outputs['unique_indices']]))

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, batch: Union[Dict, None] = None) -> Dict[str, torch.Tensor]:
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
            new_batch = None
            if batch is not None:
                new_batch = {}
                new_batch["image"] = batch["image"].reshape(-1, 3)[start_idx:end_idx]
                new_batch["normal"] = batch["normal"].reshape(-1, 3)[start_idx:end_idx]
                new_batch["albedo"] = batch["albedo"].reshape(-1, 3)[start_idx:end_idx]
                new_batch["specular"] = batch["specular"].reshape(-1, 3)[start_idx:end_idx]
                new_batch["shininess"] = batch["shininess"].reshape(-1)[start_idx:end_idx]
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, batch=new_batch)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)

        gt_rgb = linear_to_sRGB(gt_rgb, use_quantile=True)
        predicted_rgb = linear_to_sRGB(predicted_rgb, use_quantile=True)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        gt_envmap = self.metadata["environment_maps"][self.metadata["render_metadata"][batch["image_idx"]]["environment_map_idx"]] # H, W, 3
        gt_envmap = gt_envmap.to(self.device)
        gt_envmap = linear_to_sRGB(gt_envmap, use_quantile=True)  # N, 3

        with torch.no_grad():
            ray_samples = self.equirectangular_sampler.generate_direction_samples()
            ray_samples = ray_samples.to(self.device)
            ray_samples.camera_indices = torch.ones_like(ray_samples.camera_indices) * batch["image_idx"]
            # get RENI latent codes for either train or eval
            illumination_latents, scale = self.illumination_latents, self.scale
            illumination_latents = illumination_latents[
                ray_samples.camera_indices.squeeze()
            ]  # [num_rays, latent_dim, 3]
            # if illumiantion field is RENI
            if isinstance(self.illumination_field, RENIField):
                scale = scale[ray_samples.camera_indices.squeeze()]  # [num_rays]

                illumination_field_outputs = self.illumination_field.forward(
                    ray_samples=ray_samples, latent_codes=illumination_latents, scale=scale
                )  # [num_unique_camera_indices * num_illumination_directions, 3]
            else:
                illumination_field_outputs = self.illumination_field.forward(
                    ray_samples=ray_samples, latent_codes=illumination_latents
                )
            hdr_envmap = illumination_field_outputs[RENIFieldHeadNames.RGB]
            hdr_envmap = self.illumination_field.unnormalise(hdr_envmap)  # N, 3
            ldr_envmap = linear_to_sRGB(hdr_envmap, use_quantile=True)  # N, 3
            # reshape to H, W, 3
            height = self.equirectangular_sampler.height
            width = self.equirectangular_sampler.width
            ldr_envmap = ldr_envmap.reshape(height, width, 3)

            hdr_mean = torch.mean(hdr_envmap, dim=-1)
            hdr_mean = hdr_mean.reshape(height, width, 1)
            hdr_mean_log_heatmap = colormaps.apply_depth_colormap(
                hdr_mean,
                near_plane=hdr_mean.min(),
                far_plane=hdr_mean.max(),
            )

            combined_reni_envmap = torch.cat([gt_envmap, ldr_envmap, hdr_mean_log_heatmap], dim=1)

            images_dict["reni_envmap"] = combined_reni_envmap

        return metrics_dict, images_dict