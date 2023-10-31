"""
A Very Simple Inverse Setting for RENI configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from reni.data.dataparsers.reni_inverse_dataparser import RENIInverseDataParserConfig
from reni.data.datamanagers.reni_inverse_datamanager import RENIInverseDataManagerConfig
from reni.models.reni_inverse_model import RENIInverseModelConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import IcosahedronSamplerConfig, EquirectangularSamplerConfig
from reni.pipelines.reni_inverse_pipeline import RENIInvesePipelineConfig

RENIInverse = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-inverse",
        steps_per_eval_batch=2500,
        steps_per_eval_image=500,
        steps_per_save=2000,
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=RENIInvesePipelineConfig(
            datamanager=RENIInverseDataManagerConfig(
                dataparser=RENIInverseDataParserConfig(
                  shininess=500.0,
                ),
                images_on_gpu=True,
                masks_on_gpu=True,
                normals_on_gpu=True,
                albedo_on_gpu=True,
                specular_on_gpu=True,
                shininess_on_gpu=True,
                train_num_images_to_sample_from=40,
                train_num_times_to_repeat_images=50,
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=512,
            ),
            # model=NerfactoModelConfig(),
            model=RENIInverseModelConfig(
                eval_num_rays_per_chunk=512,
                # illumination_sampler=IcosahedronSamplerConfig(
                #     num_directions=256,
                #     apply_random_rotation=False,
                #     remove_lower_hemisphere=False,
                # ),
                illumination_sampler=EquirectangularSamplerConfig(
                    width=128,
                ),
                illumination_field=RENIFieldConfig(
                    conditioning="Attention",
                    invariant_function="VN",
                    equivariance="SO2",
                    axis_of_invariance="z",  # Nerfstudio world space is z-up
                    positional_encoding="NeRF",
                    encoded_input="Directions",
                    latent_dim=100,
                    hidden_features=128,
                    hidden_layers=9,
                    mapping_layers=5,
                    mapping_features=128,
                    num_attention_heads=8,
                    num_attention_layers=6,
                    output_activation="None",
                    last_layer_linear=True,
                    fixed_decoder=True,
                    trainable_scale=True,
                ),
                illumination_field_ckpt_path=Path("outputs/reni/reni_plus_plus_models/latent_dim_100/"),
                illumination_field_ckpt_step=50000,
                loss_inclusions={
                  'rgb_l1_loss': True,
                  'rgb_l2_loss': False,
                  'cosine_similarity_loss': True,
                },
                loss_coefficients={
                  'rgb_l1_loss': 1.0,
                  'rgb_l2_loss': 0.01,
                  'cosine_similarity_loss': 1.0,
                },
            ),
        ),
        optimizers={
            "illumination_latents": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.000001, max_steps=20001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Nerfacto.",
)
