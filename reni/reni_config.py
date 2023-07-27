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
            ),
            model=RENIModelConfig(
                field=RENIFieldConfig(
                    conditioning='Concat',
                    equivariance="SO2",
                    axis_of_invariance="z", # Nerfstudio world space is z-up
                    positional_encoding="NeRF",
                    encoded_input="Directions",
                    latent_dim=36,
                    hidden_features=256,
                    hidden_layers=9,
                    mapping_layers=5,
                    mapping_features=128,
                    output_activation="None",
                    last_layer_linear=True,
                ),
                loss_coefficients={
                    "rgb_hdr_loss": 10.0,
                    "rgb_ldr_loss": 10.0,
                    "cosine_similarity_loss": 1.0,
                    "kld_loss": 0.00001,
                    "scale_inv_loss": 1.0,
                },
                scale_invariant_loss=True,
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
