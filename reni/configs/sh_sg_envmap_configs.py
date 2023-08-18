"""
RENI configuration file.
"""
from pathlib import Path

from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
from reni.models.reni_model import RENIModelConfig
from reni.pipelines.reni_pipeline import RENIPipelineConfig
from reni.illumination_fields.sg_illumination_field import SphericalGaussianFieldConfig
from reni.illumination_fields.sh_illumination_field import SphericalHarmonicIlluminationFieldConfig
from reni.illumination_fields.environment_map_field import EnvironmentMapFieldConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
)

SHField = MethodSpecification(
    config=TrainerConfig(
        method_name="sh-illumination-field",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=RENIPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR_AUG"),
                    download_data=False,
                    train_subset_size=None,
                    val_subset_size=None,
                    convert_to_ldr=True,
                    convert_to_log_domain=False,
                    min_max_normalize=None, # in e^min = 0.0111, e^max = 8103.08
                    use_validation_as_train=False,
                ),
                train_num_rays_per_batch=8192,
                full_image_per_batch=False, # overwrites train_num_rays_per_batch
                number_of_images_per_batch=1, # overwrites train_num_rays_per_batch
            ),
            model=RENIModelConfig(
                field=SphericalHarmonicIlluminationFieldConfig(
                    spherical_harmonic_order=2,
                ),
                eval_optimisation_params={
                    "num_steps": 5000,
                    "lr_start": 0.1,
                    "lr_end": 0.0001,
                },
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


SGField = MethodSpecification(
    config=TrainerConfig(
        method_name="sg-illumination-field",
        steps_per_eval_image=5000,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=RENIPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR_AUG"),
                    download_data=False,
                    train_subset_size=None,
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None, # in e^min = 0.0111, e^max = 8103.08
                ),
                train_num_rays_per_batch=8192,
                full_image_per_batch=False,
            ),
            model=RENIModelConfig(
                field=SphericalGaussianFieldConfig(
                    row_col_gaussian_dims=(2, 1),
                    channel_dim=3
                ),
                eval_optimisation_params={
                    "num_steps": 5000,
                    "lr_start": 0.1,
                    "lr_end": 0.0001,
                },
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
                    data=Path("data/RENI_HDR_AUG"),
                    download_data=False,
                    train_subset_size=None,
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None, # in e^min = 0.0111, e^max = 8103.08
                ),
                train_num_rays_per_batch=8192,
                full_image_per_batch=False,
            ),
            model=RENIModelConfig(
                field=EnvironmentMapFieldConfig(
                    path=Path('path/to/environment_map.exr'),
                    resolution=(512, 1024),
                    trainable=False,
                    apply_padding=True,
                ),
                eval_optimisation_params={
                    "num_steps": 5000,
                    "lr_start": 0.1,
                    "lr_end": 0.0001,
                },
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