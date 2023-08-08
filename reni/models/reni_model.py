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
import matplotlib.colors

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import KLDivLoss

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
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

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.losses import WeightedMSE, KLD, WeightedCosineSimilarity, ScaleInvariantLogLoss, ScaleInvariantGradientMatchingLoss
from reni.utils.colourspace import linear_to_sRGB
from reni.discriminators.discriminators import BaseDiscriminatorConfig

CONSOLE = Console(width=120)

@dataclass
class RENIModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: RENIModel)
    field: SphericalFieldConfig = SphericalFieldConfig()
    """Field configuration"""
    include_sine_weighting: bool = True
    """Whether to include sine weighting in the loss"""
    training_regime: Literal["autodecoder", "gan"] = "autodecoder"
    """Type of training, either as an autodecoder or generative adversarial network"""
    loss_inclusions: Dict[str, bool] = to_immutable_dict({
        'mse_loss': True,
        'kld_loss': True,
        'cosine_similarity_loss': False,
        'scale_inv_loss': False,
        'scale_inv_grad_loss': False
    })
    """Which losses to include in the training"""
    discriminator: BaseDiscriminatorConfig = BaseDiscriminatorConfig()
    """Discriminator configuration"""
    eval_optimisation_params: Dict[str, Any] = to_immutable_dict({
        "num_steps": 5000,
        "lr_start": 0.1,
        "lr_end": 0.0001,
    })


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

        normalisations = {'min_max': self.metadata['min_max'],
                          'log_domain': self.metadata['convert_to_log_domain']}
        
        self.field = self.config.field.setup(num_train_data=self.num_train_data, num_eval_data=self.num_eval_data, normalisations=normalisations)

        if self.config.training_regime == 'gan':
            self.discriminator = self.config.discriminator.setup()

        # losses
        if self.config.loss_inclusions['mse_loss']:
            self.mse_loss = WeightedMSE()
        if self.config.loss_inclusions['kld_loss']:
            self.kld_loss = KLD(Z_dims=self.field.latent_dim)
        if self.config.loss_inclusions['cosine_similarity_loss']:
            self.cosine_similarity = WeightedCosineSimilarity()
        if self.config.loss_inclusions['scale_inv_loss']:
            self.scale_invariant_loss = ScaleInvariantLogLoss()
        if self.config.loss_inclusions['scale_inv_grad_loss']:
            self.scale_invariant_grad_loss = ScaleInvariantGradientMatchingLoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def create_ray_samples(self, ray_bundle: RayBundle) -> RaySamples:
        """Create ray samples from a ray bundle"""

        ray_samples = RaySamples(frustums=Frustums(origins=ray_bundle.origins,
                                                   directions=ray_bundle.directions,
                                                   starts=torch.zeros_like(ray_bundle.origins[:, 0]),
                                                   ends=torch.ones_like(ray_bundle.origins[:, 0]),
                                                   pixel_area=torch.ones_like(ray_bundle.origins[:, 0])),
                                 camera_indices=ray_bundle.camera_indices)
        
        return ray_samples

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["field"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        if self.config.loss_inclusions['scale_inv_grad_loss']:
            ray_bundle.directions.requires_grad = True
        
        ray_samples = self.create_ray_samples(ray_bundle)

        rotation=None
        latent_codes=None # if auto-decoder training regime latents are trainable params of the field
        if self.config.training_regime == 'gan':
            # sample from a uniform distribution
            latent_codes = torch.randn(1, self.field.latent_dim, 3).type_as(ray_bundle.origins)
        
        field_outputs = self.field.forward(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)

        outputs = {
            "rgb": field_outputs[RENIFieldHeadNames.RGB],
        }

        if RENIFieldHeadNames.MU in field_outputs:
            outputs["mu"] = field_outputs[RENIFieldHeadNames.MU]
        if RENIFieldHeadNames.LOG_VAR in field_outputs:
            outputs["log_var"] = field_outputs[RENIFieldHeadNames.LOG_VAR]

        return outputs
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        rgb = outputs["rgb"]

        if self.scale_invariant_loss:
            # estimate scale using least squares
            scale = (image * rgb).sum() / (rgb * rgb).sum()
            rgb = scale * rgb

        if self.metadata["min_max_normalize"]:
            min_val, max_val = self.metadata["min_max"]
            # need to unnormalize the image from between -1 and 1
            image = 0.5 * (image + 1) * (max_val - min_val) + min_val
            rgb = 0.5 * (rgb + 1) * (max_val - min_val) + min_val
        
        if self.metadata['convert_to_log_domain']:
            # undo log domain conversion
            image = torch.exp(image)
            rgb = torch.exp(rgb)
                
        psnr = self.psnr(image, rgb)

        metrics_dict = {"psnr": psnr}
        return metrics_dict


    def get_loss_dict(self, outputs, batch, ray_bundle, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        if self.config.include_sine_weighting:
            sineweight = self.get_sineweighting(ray_bundle.directions)
            sineweight = sineweight.unsqueeze(-1).expand_as(outputs["rgb"])
        else:
            sineweight = torch.ones_like(outputs["rgb"])

        loss_dict = {}

        if self.config.training_regime == 'autodecoder':
            if self.config.loss_inclusions['mse_loss']:
                mse_loss = self.mse_loss(outputs["rgb"], image, sineweight)
                loss_dict['mse_loss'] = mse_loss

            if self.config.loss_inclusions['kld_loss']:
                kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])
                loss_dict['kld_loss'] = kld_loss

            if self.config.loss_inclusions['cosine_similarity_loss']:
                cosine_similarity_loss = self.cosine_similarity(outputs["rgb"], image, sineweight)
                loss_dict['cosine_similarity_loss'] = cosine_similarity_loss

            if self.config.loss_inclusions['scale_inv_loss']:
                scale_inv_loss = self.scale_invariant_loss(outputs["rgb"], image, sineweight)
                loss_dict['scale_inv_loss'] = scale_inv_loss

            if self.config.loss_inclusions['scale_inv_grad_loss']:
                scale_inv_grad_loss = self.scale_invariant_grad_loss(outputs["rgb"], image, batch['HW'])
                loss_dict['scale_inv_grad_loss'] = scale_inv_grad_loss

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ray_bundle: RayBundle
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]

        if self.scale_invariant_loss:
            # estimate scale using least squares
            scale = (image * rgb).sum() / (rgb * rgb).sum()
            rgb = scale * rgb

        if self.metadata["min_max_normalize"]:
            min_val, max_val = self.metadata["min_max"]
            # need to unnormalize the image from between -1 and 1
            image = 0.5 * (image + 1) * (max_val - min_val) + min_val
            rgb = 0.5 * (rgb + 1) * (max_val - min_val) + min_val

        if self.metadata['convert_to_log_domain']:
            # undo log domain conversion
            image = torch.exp(image)
            rgb = torch.exp(rgb)
        
        # converting to grayscale by taking the mean across the color dimension
        image_gray = torch.mean(image, dim=-1)
        rgb_gray = torch.mean(rgb, dim=-1)

        gt_min, gt_max = torch.min(image_gray), torch.max(image_gray)

        # Creating the LogNorm object
        log_norm = matplotlib.colors.LogNorm(vmin=gt_min, vmax=gt_max)

        # Applying log normalization and creating tensors
        image_gray_log = torch.Tensor(log_norm(image_gray.cpu().detach().numpy())).to(image.device)
        rgb_gray_log = torch.Tensor(log_norm(rgb_gray.cpu().detach().numpy())).to(image.device)

        # Adding a new dimension for channel to image_gray_log and rgb_gray_log tensors
        image_gray_log = image_gray_log.unsqueeze(-1)
        rgb_gray_log = rgb_gray_log.unsqueeze(-1)

        # concatenating images
        combined_log_heatmap = torch.cat([image_gray_log, rgb_gray_log], dim=1)

        # create difference image
        difference = torch.abs(image - rgb)

        # i.e. we are not already in LDR space
        if not self.metadata['convert_to_ldr']:
            # convert from linear HDR to sRGB for viewing
            image_ldr = linear_to_sRGB(image, use_quantile=True)
            rgb_ldr = linear_to_sRGB(rgb, use_quantile=True)

        combined_rgb = torch.cat([image_ldr, rgb_ldr], dim=1)

        images_dict = {}

        images_dict["img"] = combined_rgb
        images_dict["heatmap"] = combined_log_heatmap
        images_dict['difference'] = difference

        # COMPUTE METRICS
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # TODO: since we are now sampling every pixel in the image, should we use sinewighting when computing metrics?
        # and is this valid?
        ray_bundle.directions = ray_bundle.directions.reshape(-1, 3)
        sineweight = self.get_sineweighting(ray_bundle.directions) # shape [H*W]
        # reshape to [1, C, H, W]
        H, W = image.shape[-2:]
        sineweight = sineweight.reshape(1, 1, H, W).repeat(1, 3, 1, 1)
        image = image * sineweight
        rgb = rgb * sineweight

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        # for lpips we need to convert to 0 to 1 using image.min() and image.max()
        image = (image - image.min()) / (image.max() - image.min())
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        lpips = self.lpips(image, rgb)
        assert isinstance(ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips),
        }

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

        # Return sineweight
        return torch.sin(theta)

    def fit_eval_latents(self, datamanager: VanillaDataManager):
        """Fit eval latents"""
        steps = self.config.eval_optimisation_params["num_steps"]
        lr_start = self.config.eval_optimisation_params["lr_start"]
        lr_end = self.config.eval_optimisation_params["lr_end"]

        opt = torch.optim.Adam([self.field.eval_mu], lr=lr_start)
        
        # create exponential learning rate scheduler decaying 
        # from lr_start to lr_end over the course of steps
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(lr_end/lr_start)**(1/steps))

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[blue]Loss: {task.fields[loss]}"),
            TextColumn("[green]LR: {task.fields[lr]}"),
        ) as progress:
            task = progress.add_task("[green]Optimising eval latents... ", total=steps, loss="", lr="")

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

                    progress.update(task, advance=1, loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.8f}")
