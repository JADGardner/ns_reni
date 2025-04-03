"""
RENI configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, ExponentialDecaySchedulerConfig

from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
from reni.models.reni_model import RENIModelConfig
from reni.pipelines.reni_pipeline import RENIPipelineConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.data.reni_pixel_sampler import RENIEquirectangularPixelSamplerConfig

RENIField = MethodSpecification(
    config=TrainerConfig(
        method_name="reni",
        experiment_name="reni",
        machine=MachineConfig(),
        steps_per_eval_image=5000,
        steps_per_eval_batch=50002,
        steps_per_save=10000,
        save_only_latest_checkpoint=True,
        steps_per_eval_all_images=5000,
        max_num_iterations=50001,
        mixed_precision=False,
        pipeline=RENIPipelineConfig(
          test_mode='val',
            datamanager=RENIDataManagerConfig(
                dataparser=RENIDataParserConfig(
                    data=Path("data/RENI_HDR"),
                    train_subset_size=None,
                    val_subset_size=None,
                    convert_to_ldr=False,
                    convert_to_log_domain=True,
                    min_max_normalize=None, # Prior RENI implementation used (-18.0536, 11.4533) in log domain
                    use_validation_as_train=False,
                    augment_with_mirror=True,
                    fit_val_in_ldr=False,
                ),
                pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                    full_image_per_batch=False,
                    images_per_batch=1,  # if full_image_per_batch
                    is_equirectangular=True,
                ),
                images_on_gpu=True,
                masks_on_gpu=True,
                train_num_rays_per_batch=8192,  # if not full_image_per_batch
                eval_num_rays_per_batch=8192,  # if not full_image_per_batch
            ),
            model=RENIModelConfig(
                field=RENIFieldConfig(
                    conditioning="Attention",
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z",  # Nerfstudio world space is z-up # old reni implementation was y-up
                    positional_encoding="NeRF",
                    encoded_input="Directions",  # "InvarDirection", "Directions", "Conditioning", "Both", "None"
                    latent_dim=100,  # N for a latent code size of (N x 3) # 9, 36, 49, 100 (for paper sizes)
                    hidden_features=128,  # ALL
                    hidden_layers=9,  # SIRENs
                    mapping_layers=5,  # FiLM MAPPING NETWORK
                    mapping_features=128,  # FiLM MAPPING NETWORK
                    num_attention_heads=8,  # TRANSFORMER
                    num_attention_layers=6,  # TRANSFORMER
                    output_activation="None",  # ALL
                    last_layer_linear=True,  # SIRENs
                    fixed_decoder=False,  # ALL
                    trainable_scale=False, # Used in inverse setting
                    old_implementation=False, # Used to match prior RENI implementation, input conditioning order is different
                ),
                eval_latent_optimizer={
                    "eval_latents": {
                        "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
                        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=2500),
                    },
                },
                loss_coefficients={
                    "log_mse_loss": 1.0,
                    "hdr_mse_loss": 1.0,
                    "ldr_mse_loss": 1.0,
                    "cosine_similarity_loss": 1.0,
                    "kld_loss": 0.000001,
                },
                loss_inclusions={
                    "log_mse_loss": False,
                    "hdr_mse_loss": False,
                    "ldr_mse_loss": False,
                    "cosine_similarity_loss": True,
                    "kld_loss": "train",
                    "scale_inv_loss": True,
                },
            ),
        ),
        optimizers={
            "field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),  # 1e-3 for Attention, 1e-5 for Other
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=50001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for Rotation-Equivariant Natural Illumination Fields.",
)
