"""
RENI configuration file.
"""
from pathlib import Path

from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
from reni.models.reni_model import RENIModelConfig
from reni.pipelines.reni_pipeline import RENIPipelineConfig
from reni.pipelines.sh_sg_pipeline import SHSGPipelineConfig
from reni.illumination_fields.sg_illumination_field import SphericalGaussianFieldConfig
from reni.illumination_fields.sh_illumination_field import SphericalHarmonicIlluminationFieldConfig
from reni.illumination_fields.environment_map_field import EnvironmentMapFieldConfig
from reni.data.reni_pixel_sampler import RENIEquirectangularPixelSamplerConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig


SHField = MethodSpecification(
    config=TrainerConfig(
        method_name="sh-illumination-field",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=SHSGPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR"),
                    download_data=False,
                    train_subset_size=None,
                    val_subset_size=None,
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None,  # Tuple[float, float] | Literal['min_max', 'quantile'] | None (Tuple should be in log domain if log_domain=True)
                    use_validation_as_train=True,  # SH and SG have no prior, just fit to val data
                    augment_with_mirror=False,
                ),
                pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                    full_image_per_batch=True,
                    images_per_batch=1,
                    is_equirectangular=True,
                ),
                images_on_gpu=True,
                train_num_rays_per_batch=8192,  # if not full_image_per_batch
                eval_num_rays_per_batch=8192,  # if not full_image_per_batch
            ),
            model=RENIModelConfig(
                field=SphericalHarmonicIlluminationFieldConfig(
                    spherical_harmonic_order=9,
                ),
                loss_coefficients={
                    "log_mse_loss": 1.0,
                    "hdr_mse_loss": 1.0,
                    "ldr_mse_loss": 1.0,
                    "cosine_similarity_loss": 1.0,
                    "kld_loss": 0.00001,
                    "scale_inv_loss": 1.0,
                    "scale_inv_grad_loss": 1.0,
                },
                loss_inclusions={
                    "log_mse_loss": True,
                    "hdr_mse_loss": False,
                    "ldr_mse_loss": False,
                    "cosine_similarity_loss": False,
                    "kld_loss": False,
                    "scale_inv_loss": False,
                    "scale_inv_grad_loss": False,
                    "bce_loss": False,  # For RESGAN, leave False in this config
                    "wgan_loss": False,  # For RESGAN, leave False in this config
                },
                include_sine_weighting=False,  # This is already handled by the equirectangular pixel sampler
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Directional Distance Field.",
)


SGField = MethodSpecification(
    config=TrainerConfig(
        method_name="sg-illumination-field",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=SHSGPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR"),
                    download_data=False,
                    train_subset_size=None,
                    val_subset_size=None,
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None,  # Tuple[float, float] | Literal['min_max', 'quantile'] | None (Tuple should be in log domain if log_domain=True)
                    use_validation_as_train=True,  # SH and SG have no prior, just fit to val data
                    augment_with_mirror=False,
                ),
                pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                    full_image_per_batch=True,
                    images_per_batch=1,
                    is_equirectangular=True,
                ),
                images_on_gpu=True,
                train_num_rays_per_batch=8192,  # if not full_image_per_batch
                eval_num_rays_per_batch=8192,  # if not full_image_per_batch
            ),
            model=RENIModelConfig(
                field=SphericalGaussianFieldConfig(row_col_gaussian_dims=(5, 10), channel_dim=3),
                loss_coefficients={
                    "log_mse_loss": 1.0,
                    "hdr_mse_loss": 1.0,
                    "ldr_mse_loss": 1.0,
                    "cosine_similarity_loss": 1.0,
                    "kld_loss": 0.00001,
                    "scale_inv_loss": 1.0,
                    "scale_inv_grad_loss": 1.0,
                },
                loss_inclusions={
                    "log_mse_loss": True,
                    "hdr_mse_loss": False,
                    "ldr_mse_loss": False,
                    "cosine_similarity_loss": False,
                    "kld_loss": False,
                    "scale_inv_loss": False,
                    "scale_inv_grad_loss": False,
                    "bce_loss": False,  # For RESGAN, leave False in this config
                    "wgan_loss": False,  # For RESGAN, leave False in this config
                },
                include_sine_weighting=False,  # This is already handled by the equirectangular pixel sampler
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Directional Distance Field.",
)


EnvMapField = MethodSpecification(
    config=TrainerConfig(
        method_name="envrionment-map-field",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=RENIPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR"),
                    download_data=False,
                    train_subset_size=None,
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None,  # in e^min = 0.0111, e^max = 8103.08
                ),
                pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                    num_rays_per_batch=8192,
                    full_image_per_batch=False,
                    images_per_batch=2,
                    is_equirectangular=True,
                ),
                train_num_rays_per_batch=8192,
            ),
            model=RENIModelConfig(
                field=EnvironmentMapFieldConfig(
                    path=Path("path/to/environment_map.exr"),
                    resolution=(512, 1024),
                    trainable=False,
                    apply_padding=True,
                ),
                loss_coefficients={
                    "mse_loss": 10.0,
                    "cosine_similarity_loss": 1.0,
                    "kld_loss": 0.00001,
                    "scale_inv_loss": 1.0,
                    "scale_inv_grad_loss": 1.0,
                },
                loss_inclusions={
                    "mse_loss": False,
                    "cosine_similarity_loss": True,
                    "kld_loss": True,
                    "scale_inv_loss": True,
                    "scale_inv_grad_loss": False,
                },
                include_sine_weighting=False,
                training_regime="autodecoder",
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Directional Distance Field.",
)
