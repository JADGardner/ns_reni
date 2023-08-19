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
from typing import Dict, List, Type
from pathlib import Path
import torch
from torch.nn import Parameter

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.illumination_fields.reni_illumination_field import RENIField
from reni.model_components.illumination_samplers import IlluminationSamplerConfig
from reni.model_components.shaders import LambertianShader
from reni.utils.utils import find_nerfstudio_project_root
from reni.fields.nerfacto_reni import NerfactoFieldRENI
from reni.field_components.field_heads import RENIFieldHeadNames

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel


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
    predict_specular: bool = False
    """Whether to predict specular"""


class NerfactoRENIModel(NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoRENIModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

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
            predict_specular=self.config.predict_specular,
        )

        self.illumination_sampler = self.config.illumination_sampler.setup()

        # TODO Get from checkpoint
        normalisations = {"min_max": None, "log_domain": True}

        self.illumination_field = self.config.illumination_field.setup(
            num_train_data=None, num_eval_data=self.num_train_data, normalisations=normalisations
        )

        if isinstance(self.illumination_field, RENIField):
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
            match_str = "_model.field.network."
            for key in ckpt["pipeline"].keys():
                if key.startswith(match_str):
                    illumination_field_dict[key[len(match_str) :]] = ckpt["pipeline"][key]
            # load weights of the decoder
            self.illumination_field.network.load_state_dict(illumination_field_dict)

        self.lambertian_shader = LambertianShader()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["illumination_field"] = list(self.illumination_field.parameters())
        return param_groups

    def get_illumination(self, camera_indices: torch.Tensor):
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

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[RENIFieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        albedo = self.renderer_rgb(rgb=field_outputs[RENIFieldHeadNames.ALBEDO], weights=weights)
        if self.config.predict_specular:
            specular = self.renderer_rgb(rgb=field_outputs[RENIFieldHeadNames.SPECULAR], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        normals = self.renderer_normals(normals=field_outputs[RENIFieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[RENIFieldHeadNames.PRED_NORMALS], weights=weights)

        light_colors, light_directions = self.get_illumination(ray_bundle.camera_indices)

        lambertian_color_sum, shaded_albedo = self.lambertian_shader(
            albedo=albedo,
            normals=normals,
            light_directions=light_directions,
            light_colors=light_colors,
            detach_normals=False,
        )
        outputs = {
            "rgb": shaded_albedo,
            "accumulation": accumulation,
            "depth": depth,
            "normals": normals,
            "pred_normals": pred_normals,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[RENIFieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[RENIFieldHeadNames.NORMALS].detach(),
                field_outputs[RENIFieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
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
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
