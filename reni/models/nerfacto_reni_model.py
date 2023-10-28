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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
import matplotlib.cm as cm
import numpy as np
import functools
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.utils import colormaps
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.engine.optimizers import OptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.illumination_fields.reni_illumination_field import RENIField, RENIFieldConfig
from reni.model_components.illumination_samplers import IlluminationSamplerConfig, EquirectangularSamplerConfig
from reni.model_components.shaders import LambertianShader, BlinnPhongShader
from reni.utils.utils import find_nerfstudio_project_root
from reni.fields.nerfacto_reni_field import NerfactoFieldRENI
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.renderers import RGBLambertianRenderer
from reni.utils.colourspace import linear_to_sRGB

CONSOLE = Console(width=120)


@dataclass
class NerfactoRENIModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoRENIModel)
    illumination_field: SphericalFieldConfig = SphericalFieldConfig()
    """Illumination Field"""
    illumination_field_ckpt_path: Path = Path("/path/to/ckpt.pt")
    """Path of pretrained illumination field"""
    illumination_field_ckpt_step: int = 0
    """Step of pretrained illumination field"""
    illumination_sampler: IlluminationSamplerConfig = IlluminationSamplerConfig()
    """Illumination sampler to use"""
    predict_shininess: bool = False
    """Whether to predict shininess"""
    depth_loss_mult: float = 1e-3
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = True
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.0001
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""
    eval_latent_optimizer: Dict[str, Any] = to_immutable_dict(
        {
            "eval_latents": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    """Optimizer and scheduler for latent code optimisation"""


class NerfactoRENIModel(NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoRENIModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.num_eval_data = self.kwargs["num_eval_data"]
        self.fitting_eval_latents = False

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        # Overwrite nerfacto field with one that predicts albedo (and optionally specular)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NerfactoFieldRENI(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            predict_shininess=self.config.predict_shininess,
        )

        self.illumination_sampler = self.config.illumination_sampler.setup()
        self.equirectangular_sampler = EquirectangularSamplerConfig(width=128).setup()  # for displaying environment map

        self.illumination_field = self.config.illumination_field.setup(
            num_train_data=None,
            num_eval_data=None,
        )

        # use local latents as illumination field is just for decoder
        self.train_illumination_latents = torch.nn.Parameter(
            torch.zeros((self.num_train_data, self.illumination_field.latent_dim, 3))
        )
        self.train_scale = nn.Parameter(torch.ones(self.num_train_data))

        self.eval_illumination_latents = torch.nn.Parameter(
            torch.zeros((self.num_eval_data, self.illumination_field.latent_dim, 3))
        )
        self.eval_scale = nn.Parameter(torch.ones(self.num_eval_data))

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
        self.blinn_phong_shader = BlinnPhongShader()
        self.labmertian_renderer = RGBLambertianRenderer()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["illumination_field"] = [self.train_illumination_latents, self.train_scale]
        return param_groups

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma

    def get_illumination_field_latents(self):
        """Return the illumination field latents for the current mode."""
        if self.training and not self.fitting_eval_latents:
            illumination_field_latents = self.train_illumination_latents
            scale = self.train_scale
        else:
            illumination_field_latents = self.eval_illumination_latents
            scale = self.eval_scale

        return illumination_field_latents, scale

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
        illumination_latents, scale = self.get_illumination_field_latents()  # [num_latents, latent_dim, 3]
        illumination_latents = illumination_latents[
            illumination_ray_samples.camera_indices
        ]  # [num_rays, latent_dim, 3]
        scale = scale[illumination_ray_samples.camera_indices]  # [num_rays]

        illuination_field_outputs = self.illumination_field.forward(
            ray_samples=illumination_ray_samples, latent_codes=illumination_latents, scale=scale
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

    def forward(self, ray_bundle: RayBundle, batch: Optional[Dict] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, batch)

    def get_outputs(self, ray_bundle: RayBundle, batch: Optional[Dict] = None):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=True)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[RENIFieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        albedo = self.renderer_rgb(
            rgb=field_outputs[RENIFieldHeadNames.ALBEDO],
            weights=weights,
            background_color=torch.tensor([1.0, 1.0, 1.0]),
        )
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        normals = self.renderer_normals(normals=field_outputs[RENIFieldHeadNames.NORMALS], weights=weights)

        if self.config.predict_shininess:
            shininess = self.renderer_rgb(
                rgb=field_outputs[RENIFieldHeadNames.SHININESS].repeat(1, 1, 3),
                weights=weights,
                background_color=torch.tensor([0.0, 0.0, 0.0]),
            )
            specular = 1.0 - albedo

        light_colors, light_directions = self.get_illumination_shader(ray_bundle.camera_indices)

        _, rgb = self.blinn_phong_shader(
            albedo=albedo,
            normals=normals,
            light_directions=light_directions,
            light_colors=light_colors,
            specular=specular,
            shininess=shininess[..., 0],
            view_directions=ray_samples.frustums.directions[:, 0, :],
            detach_normals=False,
        )

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normals,
            "albedo": albedo,
        }

        if self.config.predict_shininess:
            outputs["shininess"] = shininess

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            pred_normals = self.renderer_normals(field_outputs[RENIFieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["pred_normal"] = pred_normals
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(),
                field_outputs[RENIFieldHeadNames.NORMALS],
                ray_bundle.directions,
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[RENIFieldHeadNames.NORMALS].detach(),
                field_outputs[RENIFieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            if self.config.depth_loss_type in (DepthLossType.DS_NERF, DepthLossType.URF):
                metrics_dict["depth_loss"] = 0.0
                sigma = self._get_sigma().to(self.device)
                termination_depth = batch["depth"].to(self.device)
                for i in range(len(outputs["weights_list"])):
                    metrics_dict["depth_loss"] += depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=termination_depth,
                        predicted_depth=outputs["depth"],
                        sigma=sigma,
                        directions_norm=outputs["directions_norm"],
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    ) / len(outputs["weights_list"])
            elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["expected_depth"], batch["depth_image"].to(self.device)
                )
            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")

        metrics_dict["shininess_min"] = torch.min(outputs["shininess"])
        metrics_dict["shininess_max"] = torch.max(outputs["shininess"])
        metrics_dict["shininess_mean"] = torch.mean(outputs["shininess"])
        metrics_dict["albedo_min"] = torch.min(outputs["albedo"])
        metrics_dict["albedo_max"] = torch.max(outputs["albedo"])
        metrics_dict["albedo_mean"] = torch.mean(outputs["albedo"])
        metrics_dict["depth_min"] = torch.min(outputs["depth"])
        metrics_dict["depth_max"] = torch.max(outputs["depth"])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_image = outputs["rgb"]

        # if "fg_mask" in batch:
        #     image = image * batch["fg_mask"].to(self.device)
        #     pred_image = pred_image * batch["fg_mask"].to(self.device)

        loss_dict["rgb_loss"] = self.rgb_loss(image, pred_image)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

            if "rendered_normal_loss" in outputs:
                loss_dict["normal_loss"] = outputs["rendered_normal_loss"]

            if "fg_mask" in batch:
                # binary cross entropy loss for foreground mask
                # against outputs['accumulation']
                loss_dict["fg_mask_loss"] = F.binary_cross_entropy_with_logits(
                    outputs["accumulation"], batch["fg_mask"].expand_as(outputs["accumulation"]).to(self.device)
                )

            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, batch: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
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
            outputs = self.forward(ray_bundle=ray_bundle, batch=batch)
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
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        # normal = (normal + 1.0) / 2.0
        # pred_normal = (pred_normal + 1.0) / 2.0
        if "normal" in batch:
            # normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            normal_gt = batch["normal"].to(self.device)
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        if "pred_normal" in outputs:
            pred_normal = outputs["pred_normal"]
            combined_normal = torch.cat([combined_normal, pred_normal], dim=1)

        combined_normal = colormaps.apply_colormap(
            combined_normal, colormap_options=colormaps.ColormapOptions(colormap="turbo", normalize=True)
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if "fg_mask" in batch:
            combined_acc = torch.cat([acc, batch["fg_mask"].to(self.device).expand_as(acc)], dim=1)
        else:
            combined_acc = torch.cat([acc], dim=1)

        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            colormap_options=colormaps.ColormapOptions(colormap="turbo", normalize=True),
        )
        if "depth" in batch:
            depth_gt = colormaps.apply_depth_colormap(
                batch["depth"].to(self.device),
                accumulation=torch.where(batch["depth"].to(self.device) != torch.inf, 1, 0),
                colormap_options=colormaps.ColormapOptions(colormap="turbo", normalize=True),
            )
            combined_depth = torch.cat([depth_gt, depth], dim=1)
        else:
            combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        # lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        if "albedo" in outputs:
            images_dict["albedo"] = outputs["albedo"]

        if "shininess" in outputs:
            images_dict["shininess"] = outputs["shininess"]

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        with torch.no_grad():
            ray_samples = self.equirectangular_sampler.generate_direction_samples()
            ray_samples = ray_samples.to(self.device)
            ray_samples.camera_indices = torch.ones_like(ray_samples.camera_indices) * batch["image_idx"]
            latents = self.eval_illumination_latents[ray_samples.camera_indices.squeeze()]
            scale = self.eval_scale[ray_samples.camera_indices.squeeze()]
            illumination_field_outputs = self.illumination_field(ray_samples, latent_codes=latents, scale=scale)
            hdr_envmap = illumination_field_outputs[RENIFieldHeadNames.RGB]
            hdr_envmap = self.illumination_field.unnormalise(hdr_envmap)  # N, 3
            ldr_envmap = linear_to_sRGB(hdr_envmap)  # N, 3
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

            combined_reni_envmap = torch.cat([ldr_envmap, hdr_mean_log_heatmap], dim=1)

            images_dict["reni_envmap"] = combined_reni_envmap

        return metrics_dict, images_dict

    def fit_latent_codes_for_eval(self, datamanager: DataManager):
        """Fit evaluation latent codes to session envmaps so that illumination is correct."""

        # Make sure we are using eval RENI inside self.forward()
        self.fitting_eval_latents = True

        param_group = {"eval_latents": [self.eval_illumination_latents, self.eval_scale]}
        optimizer = Optimizers(self.config.eval_latent_optimizer, param_group)
        steps = optimizer.config["eval_latents"]["scheduler"].max_steps

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[blue]Loss: {task.fields[loss]}"),
            TextColumn("[green]LR: {task.fields[lr]}"),
        ) as progress:
            task = progress.add_task("[green]Optimising eval latents... ", total=steps, loss="", lr="")

            # this is likely already set by config, but just in case
            # ensures only latents (and scale if used) are optimised
            with self.illumination_field.hold_decoder_fixed():
                # Reset latents to zeros for fitting
                self.eval_illumination_latents.data.zero_()
                # Reset scale to 1.0 for fitting
                self.eval_scale.data.fill_(1.0)

                for step in range(steps):
                    ray_bundle, batch = datamanager.next_eval(step)

                    model_outputs = self.forward(ray_bundle=ray_bundle)
                    metrics_dict = self.get_metrics_dict(model_outputs, batch)
                    loss_dict = self.get_loss_dict(model_outputs, batch, metrics_dict)
                    loss = functools.reduce(torch.add, loss_dict.values())
                    optimizer.zero_grad_all()
                    loss.backward()
                    optimizer.optimizer_step("eval_latents")
                    optimizer.scheduler_step("eval_latents")

                    progress.update(
                        task,
                        advance=1,
                        loss=f"{loss.item():.4f}",
                        lr=f"{optimizer.schedulers['eval_latents'].get_last_lr()[0]:.8f}",
                    )

        # No longer using eval RENI
        self.fitting_eval_latents = False
