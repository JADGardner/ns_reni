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
from typing import Any, Dict, List, Tuple, Type, Literal
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
import functools

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
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.losses import WeightedMSE, KLD
from reni.utils.colourspace import linear_to_sRGB

CONSOLE = Console(width=120)

@dataclass
class RENIModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: RENIModel)
    field: RENIFieldConfig = RENIFieldConfig()
    """Field configuration"""
    training_regime: Literal["autodecoder", "gan"] = "autodecoder"
    """Type of training, either as an autodecoder or generative adversarial network"""


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
        self.metadata = kwargs["metadata"]
        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        self.field = self.config.field.setup(num_train_data=self.num_train_data, num_eval_data=self.num_eval_data, min_max=self.metadata['min_max'])

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
        }

        if self.field.split_head:
            outputs["hdr"] = field_outputs[RENIFieldHeadNames.HDR]
            outputs["ldr"] = field_outputs[RENIFieldHeadNames.LDR]
            outputs["mixing"] = field_outputs[RENIFieldHeadNames.MIXING]

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

        sineweight = self.get_sineweighting(ray_bundle.directions)
        sineweight = sineweight.unsqueeze(-1).expand_as(outputs["rgb"])

        loss_dict = {}

        if self.field.split_head:
            ldr_image = linear_to_sRGB(image)
            rgb_hdr_loss = self.rgb_loss(outputs["hdr"], image, sineweight)
            rgb_ldr_loss = self.rgb_loss(outputs["ldr"], ldr_image, sineweight)
            loss_dict['rgb_hdr_loss'] = rgb_hdr_loss
            loss_dict['rgb_ldr_loss'] = rgb_ldr_loss
        else:
            rgb_hdr_loss = self.rgb_loss(outputs["rgb"], image, sineweight)
            loss_dict['rgb_hdr_loss'] = rgb_hdr_loss

        kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])
        loss_dict["kld_loss"] = kld_loss

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]

        if self.metadata["min_max_normalize"]:
            min_val, max_val = self.metadata["min_max"]
            # need to unnormalize the image from between -1 and 1
            image = 0.5 * (image + 1) * (max_val - min_val) + min_val
            rgb = 0.5 * (rgb + 1) * (max_val - min_val) + min_val
        
        if self.metadata['convert_to_log_domain']:
            # undo log domain conversion
            image = torch.exp(image)
            rgb = torch.exp(rgb)

        # i.e. we are not already in LDR space
        if not self.metadata['convert_to_ldr']:
            # convert from linear HDR to sRGB for viewing
            image = linear_to_sRGB(image)
            rgb = linear_to_sRGB(rgb)

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        # lpips = self.lpips(image, rgb) # TODO Needs -1 to 1 range
        assert isinstance(ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            # "lpips": float(lpips),
        }

        images_dict = {}

        if self.field.split_head:
          mixing = outputs["mixing"] # [H, W, C]
          images_dict["mixing"] = mixing

        images_dict["img"] = combined_rgb

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

    def fit_eval_latents(self, datamanager: VanillaDataManager):
        """Fit eval latents"""
        steps = 5000
        lr_start = 0.1
        lr_end = 0.0001

        opt = torch.optim.Adam(self.field.parameters(), lr=lr_start)
        
        # create exponential learning rate scheduler decaying 
        # from lr_start to lr_end over the course of steps
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(lr_end/lr_start)**(1/steps))

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[blue]Loss: {task.fields[extra]}"),
        ) as progress:
            task = progress.add_task("[green]Optimising eval latents... ", total=steps, extra="")

            with self.field.hold_decoder_fixed():
                self.field.reset_eval_latents()

                for step in range(steps):
                    ray_bundle, batch = datamanager.next_eval(step)
                    model_outputs = self(ray_bundle)  # train distributed data parallel model if world_size > 1
                    loss_dict = self.get_loss_dict(model_outputs, batch, ray_bundle)
                    # add all losses together
                    loss = functools.reduce(torch.add, loss_dict.values())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    lr_scheduler.step()

                    progress.update(task, advance=1, extra=f"{loss.item():.4f}")
