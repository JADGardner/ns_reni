"""
RENI configuration file.
"""
from pathlib import Path

from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
from reni.models.reni_model import RENIModelConfig
from reni.pipelines.reni_pipeline import RENIPipelineConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.field_components.vn_encoder import VariationalVNEncoderConfig
from reni.discriminators.discriminators import CNNDiscriminatorConfig

from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
)

RENIField = MethodSpecification(
    config=TrainerConfig(
        method_name="reni",
        experiment_name="reni",
        machine=MachineConfig(num_gpus=1),
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
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None, # in e^min = 0.0111, e^max = 8103.08
                    use_validation_as_train=False,
                ),
                train_num_rays_per_batch=8192,
                full_image_per_batch=False,
                number_of_images_per_batch=2, # if using full images
            ),
            model=RENIModelConfig(
                field=RENIFieldConfig(
                    conditioning='Attention',
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z", # Nerfstudio world space is z-up
                    positional_encoding="NeRF",
                    encoded_input="Directions", # "InvarDirection", "Directions", "Conditioning", "Both"
                    latent_dim=100, # N for a latent code size of (N x 3)
                    hidden_features=128, # ALL
                    hidden_layers=9, # SIRENs
                    mapping_layers=5, # FiLM MAPPING NETWORK
                    mapping_features=128, # FiLM MAPPING NETWORK
                    num_attention_heads=8, # TRANSFORMER
                    num_attention_layers=6, # TRANSFORMER
                    output_activation="None", # ALL
                    last_layer_linear=True, # SIRENs
                ),
                discriminator=CNNDiscriminatorConfig(
                    num_layers=5,
                    initial_filters=64,
                ),
                encoder=VariationalVNEncoderConfig(
                    l2_dist_attn=True,
                    invariance="SO2",
                    fusion_strategy='late',
                ),
                eval_optimisation_params={
                    "num_steps": 2500,
                    "lr_start": 1e-1,
                    "lr_end": 1e-7, 
                },
                loss_coefficients={
                    "log_mse_loss": 10.0,
                    "hdr_mse_loss": 1.0,
                    "ldr_mse_loss": 1.0,
                    "cosine_similarity_loss": 1.0,
                    "kld_loss": 0.00001,
                    "scale_inv_loss": 1.0,
                    "scale_inv_grad_loss": 1.0,
                },
                loss_inclusions={
                    "log_mse_loss": False,
                    "hdr_mse_loss": False,
                    "ldr_mse_loss": False,
                    "cosine_similarity_loss": True,
                    "kld_loss": True,
                    "scale_inv_loss": True,
                    "scale_inv_grad_loss": False,
                },
                include_sine_weighting=False, # This is already done by the equirectangular pixel sampler
                training_regime="autodecoder",
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15), # 1e-3 for Attention, 1e-5 for Other
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
            "encoder": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Rotation-Equivariant Natural Illumination Field.",
)