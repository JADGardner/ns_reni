"""
Implementation of RENI++.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Type, Literal, Optional, Union
import functools
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn

import torch
import torch.nn as nn
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from reni.illumination_fields.base_spherical_field import SphericalFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.model_components.losses import KLD, ScaleInvariantLogLoss
from reni.utils.colourspace import linear_to_sRGB

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.optimizers import OptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import SchedulerConfig


CONSOLE = Console(width=120)


@dataclass
class RENIModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: RENIModel)
    field: SphericalFieldConfig = SphericalFieldConfig()
    """Field configuration"""
    loss_inclusions: Dict[str, Union[bool, Literal["train", "eval", "both"]]] = to_immutable_dict(
        {
            "log_mse_loss": False,
            "hdr_mse_loss": False,
            "ldr_mse_loss": False,
            "kld_loss": False,
            "cosine_similarity_loss": False,
            "scale_inv_loss": False,
        }
    )
    """Which losses to include in the training"""
    eval_latent_optimizer: Dict[str, Any] = to_immutable_dict(
        {
            "eval_latents": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    """Optimizer and scheduler for latent code optimisation"""
    eval_latent_batch_type: Literal['full_image', 'ray_samples'] = 'ray_samples'
    """Whether to use full images or ray samples from all images for eval latent optimisation"""


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
        self.fitting_eval_latents = False
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

        # losses
        if self.config.loss_inclusions["log_mse_loss"] in [True, "train", "eval", "both"]:
            self.log_mse_loss = nn.MSELoss()
        if self.config.loss_inclusions["hdr_mse_loss"] in [True, "train", "eval", "both"]:
            self.hdr_mse_loss = nn.MSELoss()
        if self.config.loss_inclusions["ldr_mse_loss"] in [True, "train", "eval", "both"]:
            self.ldr_mse_loss = nn.MSELoss()
        if self.config.loss_inclusions["kld_loss"] in [True, "train", "eval", "both"]:
            self.kld_loss = KLD(Z_dims=self.field.latent_dim)
        if self.config.loss_inclusions["cosine_similarity_loss"] in [True, "train", "eval", "both"]:
            self.cosine_similarity = nn.CosineSimilarity(dim=1)
        if self.config.loss_inclusions["scale_inv_loss"] in [True, "train", "eval", "both"]:
            self.scale_invariant_loss = ScaleInvariantLogLoss()
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
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, batch: Optional[dict] = None):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        ray_samples = self.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)

        field_outputs = self.field.forward(ray_samples=ray_samples)

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

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device

        batch["image"] = batch["image"].to(device)

        loss_dict = {}

        # Unlike original RENI implementation, the sineweighting
        # is implemented by the ray sampling so no need to modify losses
        if not self.fitting_eval_latents:
            if self.config.loss_inclusions["log_mse_loss"] in [True, "train", "both"]:
                log_mse_loss = self.log_mse_loss(outputs["rgb"], batch["image"])
                loss_dict["log_mse_loss"] = log_mse_loss

            if self.config.loss_inclusions["hdr_mse_loss"] in [True, "train", "both"]:
                hdr_mse_loss = self.hdr_mse_loss(torch.exp(outputs["rgb"]), torch.exp(batch["image"]))
                loss_dict["hdr_mse_loss"] = hdr_mse_loss

            if self.config.loss_inclusions["ldr_mse_loss"] in [True, "train", "both"]:
                ldr_mse_loss = self.ldr_mse_loss(outputs["rgb"], batch["image"])
                loss_dict["ldr_mse_loss"] = ldr_mse_loss

            if self.config.loss_inclusions["kld_loss"] in [True, "train", "both"]:
                kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])
                loss_dict["kld_loss"] = kld_loss

            if self.config.loss_inclusions["cosine_similarity_loss"] in [True, "train", "both"]:
                similarity = self.cosine_similarity(outputs["rgb"], batch["image"])
                cosine_similarity_loss = 1.0 - similarity.mean()
                loss_dict["cosine_similarity_loss"] = cosine_similarity_loss

            if self.config.loss_inclusions["scale_inv_loss"] in [True, "train", "both"]:
                scale_inv_loss = self.scale_invariant_loss(outputs["rgb"], batch["image"])
                loss_dict["scale_inv_loss"] = scale_inv_loss
        else:
            if self.config.loss_inclusions["log_mse_loss"] in [True, "eval", "both"]:
                log_mse_loss = self.log_mse_loss(outputs["rgb"], batch["image"])
                loss_dict["log_mse_loss"] = log_mse_loss

            if self.config.loss_inclusions["hdr_mse_loss"] in [True, "eval", "both"]:
                hdr_mse_loss = self.hdr_mse_loss(torch.exp(outputs["rgb"]), torch.exp(batch["image"]))
                loss_dict["hdr_mse_loss"] = hdr_mse_loss

            if self.config.loss_inclusions["ldr_mse_loss"] in [True, "eval", "both"]:
                ldr_mse_loss = self.ldr_mse_loss(outputs["rgb"], batch["image"])
                loss_dict["ldr_mse_loss"] = ldr_mse_loss

            if self.config.loss_inclusions["kld_loss"] in [True, "eval", "both"]:
                kld_loss = self.kld_loss(outputs["mu"], outputs["log_var"])
                loss_dict["kld_loss"] = kld_loss

            if self.config.loss_inclusions["cosine_similarity_loss"] in [True, "eval", "both"]:
                similarity = self.cosine_similarity(outputs["rgb"], batch["image"])
                cosine_similarity_loss = 1.0 - similarity.mean()
                loss_dict["cosine_similarity_loss"] = cosine_similarity_loss

            if self.config.loss_inclusions["scale_inv_loss"] in [True, "eval", "both"]:
                scale_inv_loss = self.scale_invariant_loss(outputs["rgb"], batch["image"])
                loss_dict["scale_inv_loss"] = scale_inv_loss

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        device = outputs["rgb"].device

        batch["image"] = batch["image"].to(device)

        if self.metadata["fit_val_in_ldr"]:
            gt_image = self.metadata["hdr_val_images"][batch["indices"][0, 0]].reshape(-1, 3)  # [num_rays, 3]
            gt_image = gt_image.to(device)
        else:
            gt_image = batch["image"]  # [num_rays, 3]

        pred_image = outputs["rgb"]  # [num_rays, 3]

        # reshape to [H, W, 3]
        gt_image = gt_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)
        pred_image = pred_image.reshape(self.metadata["image_height"], self.metadata["image_width"], 3)

        if self.config.loss_inclusions["scale_inv_loss"] in [True, "eval", "both"]:
            # estimate scale using least squares
            scale = (gt_image * pred_image).sum() / (pred_image * pred_image).sum()
            pred_image = scale * pred_image

        gt_image = self.field.unnormalise(gt_image)
        pred_image = self.field.unnormalise(pred_image)

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

        if "mask" in batch:
            mask = batch["mask"].reshape(self.metadata["image_height"], self.metadata["image_width"], 1).expand_as(
                gt_image_ldr
            ).to(device) # [H, W, 3]
            # we should mask gt_image_ldr to show only the pixels that were used in the loss
            masked_gt_image_ldr = gt_image_ldr * mask
            combined_rgb = torch.cat([gt_image_ldr, masked_gt_image_ldr, pred_image_ldr], dim=1)
        else:
            combined_rgb = torch.cat([gt_image_ldr, pred_image_ldr], dim=1)

        images_dict = {}

        images_dict["img"] = combined_rgb
        images_dict["heatmap"] = combined_log_heatmap
        images_dict["difference"] = difference

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
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, batch: Optional[dict] = None
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

    def fit_eval_latents(self, datamanager: VanillaDataManager):
        """Fit eval latents"""

        self.fitting_eval_latents = True

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
                    if self.config.eval_latent_batch_type == 'ray_samples':
                        ray_bundle, batch = datamanager.next_eval(step)
                    else:
                        _, ray_bundle, batch = datamanager.next_eval_image(step)
                    model_outputs = self(ray_bundle)
                    if self.metadata["fit_val_in_ldr"]:
                        model_outputs["rgb"] = linear_to_sRGB(self.field.unnormalise(model_outputs["rgb"]))
                    loss_dict = self.get_loss_dict(model_outputs, batch, ray_bundle)
                    # add losses together
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

        self.fitting_eval_latents = False
