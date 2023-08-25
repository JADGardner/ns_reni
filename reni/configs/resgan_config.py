"""
RENI configuration file.
"""
from pathlib import Path

from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
from reni.models.reni_model import RENIModelConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.engine.resgan_trainer import RESGANTrainerConfig
from reni.pipelines.resgan_pipeline import RESGANPipelineConfig
from reni.data.reni_pixel_sampler import RENIEquirectangularPixelSamplerConfig
from reni.discriminators.discriminators import (
    CNNDiscriminatorConfig,
    VNTransformerDiscriminatorConfig,
)  # pylint: disable=unused-import

from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, ExponentialDecaySchedulerConfig

RESGANField = MethodSpecification(
    config=RESGANTrainerConfig(
        method_name="resgan",
        experiment_name="resgan",
        machine=MachineConfig(),
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=RESGANPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR_AUG"),
                    download_data=False,
                    train_subset_size=None,
                    val_subset_size=None,
                    convert_to_ldr=True,
                    convert_to_log_domain=False,
                    min_max_normalize=None,  # in e^min = 0.0111, e^max = 8103.08
                    use_validation_as_train=False,
                ),
                pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                    num_rays_per_batch=8192,
                    full_image_per_batch=True,
                    images_per_batch=4,
                    is_equirectangular=True,
                ),
                images_on_gpu=True,
                train_num_rays_per_batch=8192,
            ),
            model=RENIModelConfig(
                field=RENIFieldConfig(
                    conditioning="Attention",
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z",  # Nerfstudio world space is z-up
                    positional_encoding="NeRF",
                    encoded_input="Directions",  # "InvarDirection", "Directions", "Conditioning", "Both"l
                    latent_dim=100,  # N for a latent code size of (N x 3)
                    hidden_features=128,
                    hidden_layers=9,
                    mapping_layers=5,
                    mapping_features=128,
                    num_attention_heads=8,
                    num_attention_layers=6,
                    output_activation="None",
                    last_layer_linear=True,
                ),
                # discriminator=CNNDiscriminatorConfig(
                #     num_layers=5,
                #     initial_filters=64,
                # ),
                discriminator=VNTransformerDiscriminatorConfig(
                    hidden_dim=32,
                    depth=2,
                    dim_head=32,
                    heads=2,
                    l2_dist_attn=True,
                    invariance="SO2",
                    fusion_strategy="late",
                ),
                eval_latent_optimizer={
                    "eval_latents": {
                        "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
                        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=2500),
                    },
                },
                loss_coefficients={
                    "bce_loss": 1.0,
                    "wgan_loss": 1.0,
                },
                loss_inclusions={
                    "bce_loss": False,  # for GAN
                    "wgan_loss": True,  # for WGAN
                },
                include_sine_weighting=False,  # This is already done by the equirectangular pixel sampler
                training_regime="gan",
            ),
            gan_type="wgan",
            discriminator_train_ratio=1,
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),  # 1e-3 for Attention, 1e-5 for Other
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
            "encoder": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
            "discriminator": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Rotation-Equivariant Natural Illumination Field.",
)
