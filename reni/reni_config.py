"""
RENI-NeuS configuration file.
"""
from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSRDataParserConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    MultiStepSchedulerConfig,
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig

from reni.data.reni_dataparser import RENIDataParserConfig
from reni.data.reni_datamanager import RENIDataManagerConfig
from reni.reni_model import RENIModelConfig
from reni.reni_pipeline import RENIPipelineConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.illumination_fields.sg_illumination_field import SphericalGaussianFieldConfig
from reni.illumination_fields.sh_illumination_field import SphericalHarmonicIlluminationFieldConfig


RENIField = MethodSpecification(
    config=TrainerConfig(
        method_name="reni",
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
                field=RENIFieldConfig(
                    conditioning='Concat',
                    equivariance="SO2",
                    axis_of_invariance="z", # Nerfstudio world space is z-up
                    positional_encoding="NeRF",
                    encoded_input="Directions", # "InvarDirection", "Directions", "Conditioning", "Both"
                    latent_dim=100,
                    hidden_features=256,
                    hidden_layers=9,
                    mapping_layers=5,
                    mapping_features=128,
                    output_activation="None",
                    last_layer_linear=True,
                ),
                # field=SphericalGaussianFieldConfig(
                #     row_col_gaussian_dims=(2, 1),
                #     channel_dim=3
                # ),
                # field=SphericalHarmonicIlluminationFieldConfig(
                #     spherical_harmonic_order=2,
                # ),
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
