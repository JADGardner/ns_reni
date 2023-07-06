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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import KLDivLoss

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.losses import WeightedMSE, KLD
from reni.utils.colourspace import linear_to_sRGB

@dataclass
class RENIModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: RENIModel)
    field: RENIFieldConfig = RENIFieldConfig()
    """Field configuration"""


class RENIModel(Model):
    """Rotation-Equivariant Neural Illumination Model

    Args:
        config: Model config
    """

    config: RENIModelConfig

    def __init__(
        self,
        config: RENIModelConfig,
        num_eval_data: int,
        **kwargs,
    ) -> None:
        self.num_eval_data = num_eval_data
        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        self.field = self.config.field.setup(num_train_data=self.num_train_data, num_eval_data=self.num_eval_data)

        # losses
        self.rgb_loss = WeightedMSE()
        self.kld_loss = KLD(Z_dims=self.field.latent_dim)

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["field"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        field_outputs = self.field.forward(ray_bundle, None)

        outputs = {
            "rgb": field_outputs[RENIFieldHeadNames.RGB],
            "mu": field_outputs[RENIFieldHeadNames.MU],
            "log_var": field_outputs[RENIFieldHeadNames.LOG_VAR],
            "hdr": field_outputs[RENIFieldHeadNames.HDR],
            "ldr": field_outputs[RENIFieldHeadNames.LDR],
            "mixing": field_outputs[RENIFieldHeadNames.MIXING],
        }

        return outputs
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        
        device = outputs["rgb"].device
        image = batch["image"].to(device)
                
        psnr = self.psnr(image, outputs["rgb"])

        metrics_dict = {"psnr": psnr}
        return metrics_dict


    def get_loss_dict(self, outputs, batch, ray_bundle, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        ldr_image = linear_to_sRGB(image)
        # ldr_rgb = linear_to_sRGB(outputs["rgb"])

        sineweight = self.get_sineweighting(ray_bundle.directions)
        sineweight = sineweight.unsqueeze(-1).expand_as(outputs["rgb"])

        rgb_hdr_loss = self.rgb_loss(outputs["hdr"], image, sineweight)
        rgb_ldr_loss = self.rgb_loss(outputs["ldr"], ldr_image, sineweight)
        kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])

        loss_dict = {"rgb_hdr_loss": rgb_hdr_loss, "rgb_ldr_loss": rgb_ldr_loss, "kld_loss": kld_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        
        image = linear_to_sRGB(image)
        rgb = linear_to_sRGB(rgb)

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        # lpips = self.lpips(image, rgb)
        assert isinstance(ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            # "lpips": float(lpips),
        }

        mixing = outputs["mixing"] # [H, W, C]


        images_dict = {"img": combined_rgb, "mixing": mixing}

        return metrics_dict, images_dict

    def get_sineweighting(self, directions):
        """
          Returns a sineweight based on the vertical Z axis for a set of directions.
          Assumes directions are normalized.
        """
        assert directions.shape[1] == 3
        assert len(directions.shape) == 2

        # Extract the z coordinates
        z_coordinates = directions[:, 2]
        
        # The inclination angle (θ) is the angle from the positive z-axis, calculated as arccos(z)
        theta = torch.acos(z_coordinates)
        
        # Calculate sineweight
        sineweight = torch.sin(theta)
        
        return sineweight

    def fit_eval_latents(self, ray_bundle, batch):
        
        with self.field.fixed_decoder(self.field):
            pass