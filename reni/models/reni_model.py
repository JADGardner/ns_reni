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
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Type, Literal, Optional
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
import functools

import torch
import torch.nn as nn
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import KLDivLoss

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.optimizers import OptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import SchedulerConfig

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.losses import KLD, ScaleInvariantLogLoss, ScaleInvariantGradientMatchingLoss
from reni.utils.colourspace import linear_to_sRGB
from reni.discriminators.discriminators import BaseDiscriminatorConfig
from reni.field_components.vn_encoder import VariationalVNEncoderConfig

CONSOLE = Console(width=120)


@dataclass
class RENIModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: RENIModel)
    field: SphericalFieldConfig = SphericalFieldConfig()
    """Field configuration"""
    include_sine_weighting: bool = True
    """Whether to include sine weighting in the loss"""
    training_regime: Literal["vae", "autodecoder", "gan"] = "autodecoder"
    """Type of training, either as an vae, (variational)autodecoder or generative adversarial network"""
    loss_inclusions: Dict[str, bool] = to_immutable_dict(
        {
            "log_mse_loss": False,
            "hdr_mse_loss": False,
            "ldr_mse_loss": False,
            "kld_loss": False,
            "cosine_similarity_loss": False,
            "scale_inv_loss": False,
            "scale_inv_grad_loss": False,
            "bce_loss": False,
            "wgan_loss": False,
        }
    )
    """Which losses to include in the training"""
    discriminator: BaseDiscriminatorConfig = BaseDiscriminatorConfig()
    """Discriminator configuration"""
    encoder: VariationalVNEncoderConfig = VariationalVNEncoderConfig()
    """Rotation-Equivariant Encoder configuration"""
    eval_latent_optimizer: Dict[str, Any] = to_immutable_dict(
        {
            "eval_latents": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    """Optimizer and scheduler for latent code optimisation"""


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

        normalisations = {"min_max": self.metadata["min_max"], "log_domain": self.metadata["convert_to_log_domain"]}

        self.field = self.config.field.setup(
            num_train_data=self.num_train_data, num_eval_data=self.num_eval_data, normalisations=normalisations
        )

        if self.config.training_regime == "gan":
            self.discriminator = self.config.discriminator.setup(
                height=self.metadata["image_height"],
                width=self.metadata["image_width"],
                gan_type=self.metadata["gan_type"],
            )
            self.batch_size = self.metadata["batch_size"]
            self.rays_per_image = self.metadata["image_height"] * self.metadata["image_width"]
            self.real_label = 1
            self.fake_label = 0
            self.fixed_noise = torch.randn((1, self.field.latent_dim, 3))

        if self.config.training_regime == "vae":
            self.batch_size = self.metadata["batch_size"]
            self.rays_per_image = self.metadata["image_height"] * self.metadata["image_width"]
            self.encoder = self.config.encoder.setup(
                num_input_rays=self.rays_per_image, output_dim=self.field.latent_dim * 3
            )

        # losses
        if self.config.loss_inclusions["log_mse_loss"]:
            self.log_mse_loss = nn.MSELoss()
        if self.config.loss_inclusions["hdr_mse_loss"]:
            self.hdr_mse_loss = nn.MSELoss()
        if self.config.loss_inclusions["ldr_mse_loss"]:
            self.ldr_mse_loss = nn.MSELoss()
        if self.config.loss_inclusions["kld_loss"]:
            self.kld_loss = KLD(Z_dims=self.field.latent_dim)
        if self.config.loss_inclusions["cosine_similarity_loss"]:
            self.cosine_similarity = nn.CosineSimilarity(dim=1)
        if self.config.loss_inclusions["scale_inv_loss"]:
            self.scale_invariant_loss = ScaleInvariantLogLoss()
        if self.config.loss_inclusions["scale_inv_grad_loss"]:
            self.scale_invariant_grad_loss = ScaleInvariantGradientMatchingLoss()
        if self.config.loss_inclusions["bce_loss"]:
            self.bce_loss = nn.BCELoss()
        if self.config.loss_inclusions["wgan_loss"]:
            pass  # NOTE just a flag

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def forward(self, ray_bundle: RayBundle, batch: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        return self.get_outputs(ray_bundle, batch)

    def forward_discriminator(self, ray_bundle: RayBundle, image_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Foward on the discriminator"""
        if len(ray_bundle.directions.shape) == 3:  # [2, num_rays, 3]
            assert self.config.loss_inclusions["scale_inv_grad_loss"]
            assert ray_bundle.directions.shape[0] == 2
            # then we are using the finite diff gradient matching loss
            # and the ray_bundle is of shape [2, num_rays, 3]
            # we the second half of the rays are just the directions rolled by 1
            ray_samples = self.create_ray_samples(
                ray_bundle.origins[0], ray_bundle.directions[0], ray_bundle.camera_indices[0]
            )
            images = image_batch[0].reshape(self.batch_size, -1, 3)  # [batch_size, num_rays, 3]
            images_rolled = image_batch[1].reshape(self.batch_size, -1, 3)  # [batch_size, num_rays, 3]
            image_batch = images_rolled - images  # finite difference for scale invariant gan input
        else:
            ray_samples = self.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)
            ray_samples.frustums.directions = ray_samples.frustums.directions.reshape(
                self.batch_size, -1, 3
            )  # [batch_size, num_rays, 3]
            image_batch = image_batch.reshape(self.batch_size, -1, 3)  # [batch_size, num_rays, 3]

        image_batch = image_batch.to(self.device)
        return {"predictions": self.discriminator(ray_samples, image_batch)}

    def create_ray_samples(self, origins, directions, camera_indices) -> RaySamples:
        """Create ray samples from a ray bundle"""

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=torch.zeros_like(origins[:, 0]),
                ends=torch.ones_like(origins[:, 0]),
                pixel_area=torch.ones_like(origins[:, 0]),
            ),
            camera_indices=camera_indices,
        )

        return ray_samples

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["field"] = list(self.field.parameters())
        if self.config.training_regime == "vae":
            param_groups["encoder"] = list(self.encoder.parameters())
        if self.config.training_regime == "gan":
            param_groups["discriminator"] = list(self.discriminator.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, batch: Optional[dict] = None):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        if len(ray_bundle.directions.shape) == 3:  # [2, num_rays, 3]
            assert self.config.loss_inclusions["scale_inv_grad_loss"]
            assert ray_bundle.directions.shape[0] == 2
            # then we are using the finite diff gradient matching loss
            # and the ray_bundle is of shape [2, num_rays, 3]
            # we the second half of the rays are just the directions rolled by 1
            ray_samples = self.create_ray_samples(
                ray_bundle.origins[0], ray_bundle.directions[0], ray_bundle.camera_indices[0]
            )
        else:
            ray_samples = self.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)

        rotation = None
        latent_codes = None  # if auto-decoder training regime latents are trainable params of the field
        if self.training and self.config.training_regime == "gan":
            # sample from a uniform distribution
            latent_codes = torch.randn(self.batch_size, self.field.latent_dim, 3).type_as(ray_bundle.origins)
            latent_codes = (
                latent_codes.unsqueeze(1).expand(-1, self.rays_per_image, -1, -1).reshape(-1, self.field.latent_dim, 3)
            )  # [num_rays, latent_dim, 3]
        if self.config.training_regime == "vae":
            # encoder will expect rays batched as images not just any random set of rays with related latent codes like the
            # decoder can handle, so we need to reshape the rays into batches correctly
            # currently ray_samples.frustums.directions will be [num_rays, 3] where num_rays is [num_images * rays_per_image, 3]
            ray_samples.frustums.directions = ray_samples.frustums.directions.view(
                self.batch_size, self.rays_per_image, 3
            )
            rgb = batch["image"].view(self.batch_size, self.rays_per_image, 3).to(self.device)
            mu, log_var = self.encoder.forward(
                ray_samples=ray_samples, rgb=rgb
            )  # shapes [num_images, self.field.latent_dim * 3]
            # set ray_samples shape back for RENI which expects [num_rays, 3]
            ray_samples.frustums.directions = ray_samples.frustums.directions.view(-1, 3)

            # reparameterisation trick
            if self.training:
                log_var = torch.clamp(
                    log_var, min=-3, max=3
                )  # TODO make this a config option and check if this even makes sense
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                latent_codes = mu + (eps * std)  # [num_images, self.field.latent_dim * 3]
            else:
                latent_codes = mu  # [num_images, self.field.latent_dim * 3]

            # now reshape to match RENI latent code [num_rays, latent_dim, 3]
            latent_codes = latent_codes.view(self.batch_size, self.field.latent_dim, 3).unsqueeze(
                1
            )  # [num_images, 1, latent_dim, 3]
            latent_codes = latent_codes.expand(-1, self.rays_per_image, -1, -1)
            latent_codes = latent_codes.contiguous().view(
                self.batch_size * self.rays_per_image, self.field.latent_dim, 3
            )

            mu = mu.view(-1, self.field.latent_dim, 3)
            log_var = log_var.view(-1, self.field.latent_dim, 3)

        field_outputs = self.field.forward(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)

        outputs = {
            "rgb": field_outputs[RENIFieldHeadNames.RGB],
        }

        if self.config.training_regime == "vae":
            outputs["mu"] = mu
            outputs["log_var"] = log_var
        else:
            if RENIFieldHeadNames.MU in field_outputs:
                outputs["mu"] = field_outputs[RENIFieldHeadNames.MU]
            if RENIFieldHeadNames.LOG_VAR in field_outputs:
                outputs["log_var"] = field_outputs[RENIFieldHeadNames.LOG_VAR]

        # now generate rgb_rolled for finite diff gradient matching
        if self.config.loss_inclusions["scale_inv_grad_loss"]:
            ray_samples = self.create_ray_samples(
                ray_bundle.origins[1], ray_bundle.directions[1], ray_bundle.camera_indices[1]
            )
            field_outputs = self.field.forward(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)
            outputs["rgb_rolled"] = field_outputs[RENIFieldHeadNames.RGB]

        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        if len(batch["image"].shape) == 3:  # [2, num_rays, 3]
            assert self.config.loss_inclusions["scale_inv_grad_loss"]
            assert batch["image"].shape[0] == 2
            batch = {key: value[0] for key, value in batch.items()}  # [num_rays, ...]

        device = outputs["rgb"].device
        gt_image = batch["image"].to(device)
        pred_image = outputs["rgb"]

        if self.config.loss_inclusions["scale_inv_loss"]:
            # estimate scale using least squares
            scale = (gt_image * pred_image).sum() / (pred_image * pred_image).sum()
            pred_image = scale * pred_image

        gt_image = self.field.unnormalise(gt_image)
        pred_image = self.field.unnormalise(pred_image)

        psnr = self.psnr(preds=pred_image, target=gt_image)

        metrics_dict = {"psnr": psnr}
        return metrics_dict

    def get_gan_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        device = outputs["predictions"].device
        batch["gt_labels"] = batch["gt_labels"].to(device)

        loss_dict = {}

        if self.config.loss_inclusions["bce_loss"]:
            bce_loss = self.bce_loss(outputs["predictions"], batch["gt_labels"])
            loss_dict["bce_loss"] = bce_loss
        if self.config.loss_inclusions["wgan_loss"]:
            if batch["gt_labels"][0] == self.real_label:
                wgan_loss = torch.mean(outputs["predictions"])
            else:
                wgan_loss = -torch.mean(outputs["predictions"])
            loss_dict["wgan_loss"] = wgan_loss

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device

        if len(batch["image"].shape) == 3:  # [2, num_rays, 3]
            assert self.config.loss_inclusions["scale_inv_grad_loss"]
            assert batch["image"].shape[0] == 2
            batch_rolled = {key: value[1] for key, value in batch.items()}  # [num_rays, ...]
            batch = {key: value[0] for key, value in batch.items()}  # [num_rays, ...]
            batch_rolled["image"] = batch_rolled["image"].to(device)

        batch["image"] = batch["image"].to(device)

        loss_dict = {}

        # Unlike original RENI implementation, the sineweighting
        # is implemented by the ray sampling so no need to modify losses
        if self.config.loss_inclusions["log_mse_loss"]:
            log_mse_loss = self.log_mse_loss(outputs["rgb"], batch["image"])
            loss_dict["log_mse_loss"] = log_mse_loss

        if self.config.loss_inclusions["hdr_mse_loss"]:
            hdr_mse_loss = self.hdr_mse_loss(torch.exp(outputs["rgb"]), torch.exp(batch["image"]))
            loss_dict["hdr_mse_loss"] = hdr_mse_loss

        if self.config.loss_inclusions["ldr_mse_loss"]:
            ldr_mse_loss = self.ldr_mse_loss(outputs["rgb"], batch["image"])
            loss_dict["ldr_mse_loss"] = ldr_mse_loss

        if self.config.loss_inclusions["kld_loss"]:
            kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])
            loss_dict["kld_loss"] = kld_loss

        if self.config.loss_inclusions["cosine_similarity_loss"]:
            similarity = self.cosine_similarity(outputs["rgb"], batch["image"])
            cosine_similarity_loss = 1.0 - similarity.mean()
            loss_dict["cosine_similarity_loss"] = cosine_similarity_loss

        if self.config.loss_inclusions["scale_inv_loss"]:
            scale_inv_loss = self.scale_invariant_loss(outputs["rgb"], batch["image"])
            loss_dict["scale_inv_loss"] = scale_inv_loss

        if self.config.loss_inclusions["scale_inv_grad_loss"]:
            scale_inv_grad_loss = self.scale_invariant_grad_loss(
                outputs["rgb"], outputs["rgb_rolled"], batch["image"], batch_rolled["image"]
            )
            loss_dict["scale_inv_grad_loss"] = scale_inv_grad_loss

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        device = outputs["rgb"].device

        if len(batch["image"].shape) == 3:  # [2, num_rays, 3]
            assert self.config.loss_inclusions["scale_inv_grad_loss"]
            assert batch["image"].shape[0] == 2
            batch_rolled = {key: value[1] for key, value in batch.items()}  # [num_rays, ...]
            batch = {key: value[0] for key, value in batch.items()}  # [num_rays, ...]
            batch_rolled["image"] = batch_rolled["image"].to(device)

        batch["image"] = batch["image"].to(device)

        gt_image = batch["image"]  # [num_rays, 3]
        pred_image = outputs["rgb"]  # [num_rays, 3]

        # reshape to [H, W, 3]
        gt_image = gt_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)
        pred_image = pred_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)

        if self.config.loss_inclusions["scale_inv_loss"]:
            # estimate scale using least squares
            scale = (gt_image * pred_image).sum() / (pred_image * pred_image).sum()
            pred_image = scale * pred_image

        gt_image = self.field.unnormalise(gt_image)
        pred_image = self.field.unnormalise(pred_image)

        # if self.metadata["min_max_normalize"] is not None:
        #     min_val, max_val = self.metadata["min_max"]
        #     # need to unnormalize the image from between -1 and 1
        #     gt_image = 0.5 * (gt_image + 1) * (max_val - min_val) + min_val
        #     pred_image = 0.5 * (pred_image + 1) * (max_val - min_val) + min_val

        # if self.metadata['convert_to_log_domain']:
        #     # undo log domain conversion
        #     gt_image = torch.exp(gt_image)
        #     pred_image = torch.exp(pred_image)

        # converting to grayscale by taking the mean across the color dimension
        gt_image_gray = torch.mean(gt_image, dim=-1)
        pred_image_gray = torch.mean(pred_image, dim=-1)

        # reshape to H, W
        gt_image_gray = gt_image_gray.reshape(self.metadata["image_height"], self.metadata["image_width"], 1)
        pred_image_gray = pred_image_gray.reshape(self.metadata["image_height"], self.metadata["image_width"], 1)

        gt_min, gt_max = torch.min(gt_image_gray), torch.max(gt_image_gray)

        combined_log_heatmap = torch.cat([gt_image_gray, pred_image_gray], dim=1)

        combined_log_heatmap = colormaps.apply_depth_colormap(
            combined_log_heatmap,
            near_plane=gt_min,
            far_plane=gt_max,
        )

        # create difference image
        difference = torch.abs(gt_image - pred_image)

        # i.e. we are not already in LDR space
        if not self.metadata["convert_to_ldr"]:
            # convert from linear HDR to sRGB for viewing
            gt_image_ldr = linear_to_sRGB(gt_image, use_quantile=True)
            pred_image_ldr = linear_to_sRGB(pred_image, use_quantile=True)
        else:
            gt_image_ldr = gt_image
            pred_image_ldr = pred_image

        combined_rgb = torch.cat([gt_image_ldr, pred_image_ldr], dim=1)

        images_dict = {}

        images_dict["img"] = combined_rgb
        images_dict["heatmap"] = combined_log_heatmap
        images_dict["difference"] = difference

        if self.config.loss_inclusions["scale_inv_grad_loss"]:
            pred_rolled = outputs["rgb_rolled"]
            rolled_gt_image = batch_rolled["image"]
            pred_rolled = pred_rolled.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)
            rolled_gt_image = rolled_gt_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)
            finite_diff_gt = torch.abs(gt_image - rolled_gt_image)
            finite_diff_pred = torch.abs(pred_image - pred_rolled)
            finite_diff_gt = torch.exp(finite_diff_gt)
            finite_diff_pred = torch.exp(finite_diff_pred)
            finite_diff_gt = linear_to_sRGB(finite_diff_gt, use_quantile=True)
            finite_diff_pred = linear_to_sRGB(finite_diff_pred, use_quantile=True)
            combined_finite_diff = torch.cat([finite_diff_gt, finite_diff_pred], dim=1)
            images_dict["finite_diff"] = combined_finite_diff

        # COMPUTE METRICS
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_image = gt_image.unsqueeze(0).permute(0, 3, 1, 2)
        pred_image = pred_image.unsqueeze(0).permute(0, 3, 1, 2)
        gt_image_ldr = gt_image_ldr.unsqueeze(0).permute(0, 3, 1, 2)
        pred_image_ldr = pred_image_ldr.unsqueeze(0).permute(0, 3, 1, 2)

        metrics_dict = {}

        metrics_dict["psnr_hdr"] = self.psnr(preds=pred_image, target=gt_image)
        metrics_dict["ssim_hdr"] = self.ssim(preds=pred_image, target=gt_image)
        # for lpips we need to convert to 0 to 1 using image.min() and image.max()
        gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min())
        pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min())
        metrics_dict["lpips_hdr"] = self.lpips(pred_image, gt_image)

        # if we are not already learning in LDR space
        if not self.metadata["convert_to_ldr"]:
            metrics_dict["psnr_ldr"] = self.psnr(preds=pred_image_ldr, target=gt_image_ldr)
            metrics_dict["ssim_ldr"] = self.ssim(preds=pred_image_ldr, target=gt_image_ldr)
            metrics_dict["lpips_ldr"] = self.lpips(pred_image_ldr, gt_image_ldr)

        return metrics_dict, images_dict
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, batch: Optional[dict] = None) -> Dict[str, torch.Tensor]:
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

    def get_sineweighting(self, directions: torch.Tensor):
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

        param_group = {"eval_latents": [self.field.eval_mu]}
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

            with self.field.hold_decoder_fixed():
                self.field.reset_eval_latents()

                for step in range(steps):
                    ray_bundle, batch = datamanager.next_eval(step)
                    model_outputs = self(ray_bundle)
                    loss_dict = self.get_loss_dict(model_outputs, batch, ray_bundle)
                    # add all losses together
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
