"""
A Very Simple Inverse Setting for RENI, configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from reni.data.dataparsers.reni_inverse_dataparser import RENIInverseDataParserConfig
from reni.data.datamanagers.reni_inverse_datamanager import RENIInverseDataManagerConfig
from reni.illumination_fields.sg_illumination_field import SphericalGaussianFieldConfig
from reni.illumination_fields.sh_illumination_field import SphericalHarmonicIlluminationFieldConfig
from reni.models.reni_inverse_model import RENIInverseModelConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni.pipelines.reni_inverse_pipeline import RENIInvesePipelineConfig

RENIInverse = MethodSpecification(
    config=TrainerConfig(
        method_name="reni-inverse",
        steps_per_eval_batch=50401,
        steps_per_eval_image=200,
        steps_per_save=200,
        steps_per_eval_all_images=50401,
        max_num_iterations=50400,
        mixed_precision=False,
        log_gradients=True,
        pipeline=RENIInvesePipelineConfig(
            datamanager=RENIInverseDataManagerConfig(
                dataparser=RENIInverseDataParserConfig(
                  shininess=500.0,
                  subset_index=None,
                  envmap_remove_indicies=None,
                ),
                images_on_gpu=True,
                masks_on_gpu=True,
                normals_on_gpu=True,
                albedo_on_gpu=True,
                specular_on_gpu=True,
                shininess_on_gpu=True,
                train_num_images_to_sample_from=1,
                train_num_times_to_repeat_images=200,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=32,
            ),
            model=RENIInverseModelConfig(
                eval_num_rays_per_chunk=512,
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
                    trainable_scale=False,
                ),
                # illumination_field=SphericalHarmonicIlluminationFieldConfig(
                #     spherical_harmonic_order=9,
                # ),
                # illumination_field=SphericalGaussianFieldConfig(
                #   row_col_gaussian_dims=(5, 10),
                #   channel_dim=3
                # ),
                illumination_field_ckpt_path=Path("outputs/reni/reni_plus_plus_models/latent_dim_100/"),
                illumination_field_ckpt_step=50000,
                loss_inclusions={
                  'rgb_l1_loss': False,
                  'rgb_l2_loss': True,
                  'cosine_similarity_loss': True,
                  'prior_loss': True,
                },
                loss_coefficients={
                  'rgb_l1_loss': 0.1,
                  'rgb_l2_loss': 1e2,
                  'cosine_similarity_loss': 1.0,
                  'prior_loss': 0.001
                },
                print_nan=True,
            ),
        ),
        optimizers={
            "illumination_latents": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=0, lr_final=1e-2, max_steps=50401),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Nerfacto.",
)
