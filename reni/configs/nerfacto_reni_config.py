"""
RENI + NeRF configuration file.
"""
from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.models.nerfacto import NerfactoModelConfig

from reni.data.dataparsers.nerd_dataparser import NeRDDataParserConfig
from reni.data.datamanagers.nerd_datamanager import NeRDDataManagerConfig
from reni.models.nerfacto_reni import NerfactoRENIModelConfig
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni.model_components.illumination_samplers import IcosahedronSamplerConfig

NeRFactoRENI = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-reni",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=NeRDDataManagerConfig(
                dataparser=NeRDDataParserConfig(
                    scene="Car",
                    background_color="white",
                    mask_out_background=True,
                ),
                masks_on_gpu=True,
                images_on_gpu=True,
                normals_on_gpu=True,
                depths_on_gpu=True,
                train_num_images_to_sample_from=-1,
                train_num_times_to_repeat_images=-1,
                train_num_rays_per_batch=256,
                eval_num_rays_per_batch=256,
            ),
            # model=NerfactoModelConfig(),
            model=NerfactoRENIModelConfig(
                eval_num_rays_per_chunk=256,
                background_color="white",
                disable_scene_contraction=True,
                predict_normals=True,
                predict_specular=False,
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
                illumination_sampler=IcosahedronSamplerConfig(
                    num_directions=300,
                    apply_random_rotation=False,
                    remove_lower_hemisphere=False,
                ),
                illumination_field_ckpt_path=Path("outputs/reni/reni/2023-08-23_075123/"),
                illumination_field_ckpt_step=50000,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "illumination_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Nerfacto.",
)
