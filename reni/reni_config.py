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


RENIField = MethodSpecification(
    config=TrainerConfig(
        method_name="reni",
        steps_per_eval_image=500,
        steps_per_eval_batch=100000,
        steps_per_save=1000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=RENIPipelineConfig(
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    download_data=True,
                    train_subset_size=1,
                    convert_to_ldr=False,
                ),
                train_num_rays_per_batch=8192,
            ),
            model=RENIModelConfig(
                field=RENIFieldConfig(
                    equivariance="SO2",
                    positional_encoding="None",
                    latent_dim=36,
                    hidden_features=256,
                    hidden_layers=9,
                    output_activation="exp",
                    split_head=True,
                ),
                loss_coefficients={
                    "rgb_hdr_loss": 0.1,
                    "rgb_ldr_loss": 10.0,
                    "kld_loss": 0.0001,
                }
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Directional Distance Field.",
)
