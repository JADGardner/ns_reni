#!/usr/bin/env python3
"""
Publication Figure Generation Script

Generates all publication figures for the RENI++ paper.
Refactored from legacy notebook-based figure generation.

Usage:
    python generate_figures.py
    python generate_figures.py --figures comparison,interpolation
    python generate_figures.py --output-dir publication/figures_custom/
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Literal, Tuple
import re

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cm import get_cmap
import yaml
from PIL import Image
import io

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

from reni.configs.reni_config import RENIField
from reni.configs.sh_sg_envmap_configs import SHField, SGField
from reni.pipelines.reni_pipeline import RENIPipeline
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.utils import find_nerfstudio_project_root, rot_z, rot_y
from reni.utils.colourspace import linear_to_sRGB
from reni.data.datamanagers.reni_datamanager import RENIDataManagerConfig
from reni.data.dataparsers.reni_dataparser import RENIDataParserConfig
from reni.data.reni_pixel_sampler import RENIEquirectangularPixelSamplerConfig

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    
    # Base directories
    project_root: Path = field(default_factory=lambda: find_nerfstudio_project_root(Path(os.getcwd())))
    checkpoint_base: Path = field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("publication/figures_updated"))
    teaser_base_path: Path = field(default_factory=lambda: Path("publication/figures/teaser_base.png"))
    
    # Device settings
    device: str = "cuda:0"
    
    # Figure selection
    figures: List[str] = field(default_factory=lambda: ["all"])
    
    # Image settings
    default_image_height: int = 64
    high_res_image_height: int = 256
    
    def __post_init__(self):
        """Resolve paths relative to project root."""
        if not self.checkpoint_base.is_absolute():
            self.checkpoint_base = self.project_root / self.checkpoint_base
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        if not self.teaser_base_path.is_absolute():
            self.teaser_base_path = self.project_root / self.teaser_base_path


class ModelLoader:
    """Handles loading RENI, SH, and SG models from checkpoints."""
    
    def __init__(self, config: FigureConfig):
        self.config = config
        self.device = config.device
        
    def clean_and_load_yaml(self, yaml_content: str) -> Dict:
        """Remove Python tags and load YAML content."""
        cleaned_content = re.sub(r'!!python[^\s]*', '', yaml_content)
        return yaml.safe_load(cleaned_content)
    
    def _find_checkpoint_dir(self, load_dir: Path) -> Path:
        """
        Find the actual checkpoint directory, handling nested structures.
        
        Checkpoint dirs can be:
        - Direct: load_dir/nerfstudio_models/
        - Nested: load_dir/RENI_HDR/*/timestamp/nerfstudio_models/
        """
        # Direct structure
        if (load_dir / 'nerfstudio_models').exists():
            return load_dir
        
        # Nested structure (e.g., RENI_HDR/sh-illumination-field/2025-12-27_072308/)
        reni_hdr = load_dir / 'RENI_HDR'
        if reni_hdr.exists():
            # Find subdirectory with method name
            for method_dir in reni_hdr.iterdir():
                if method_dir.is_dir():
                    # Find most recent timestamp directory
                    timestamp_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()], reverse=True)
                    for ts_dir in timestamp_dirs:
                        if (ts_dir / 'nerfstudio_models').exists():
                            return ts_dir
        
        return load_dir  # Fall back to original
    
    def load_model(self, load_dir: Path, load_step: Optional[int] = None):
        """
        Load a model from a checkpoint directory.
        
        Handles both new nerfstudio format and old RENI format (.pt files).
        """
        # Check if this is an old-format checkpoint (has RENI.pt file)
        old_format_path = load_dir / "files" / "RENI.pt"
        if old_format_path.exists():
            return self._load_old_format(load_dir)
        
        # Find actual checkpoint directory (handles nested structures)
        actual_dir = self._find_checkpoint_dir(load_dir)
        
        # New nerfstudio format
        return self._load_nerfstudio_format(actual_dir, load_step)
    
    def _load_nerfstudio_format(self, load_dir: Path, load_step: Optional[int] = None):
        """Load model from nerfstudio checkpoint format."""
        ckpt_dir = load_dir / 'nerfstudio_models'
        
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        if load_step is None:
            checkpoint_files = list(ckpt_dir.glob("step-*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
            load_step = max(int(x.stem.split("-")[1]) for x in checkpoint_files)
        
        ckpt_path = ckpt_dir / f'step-{load_step:09d}.ckpt'
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        reni_model_dict = {}
        for key in ckpt['pipeline'].keys():
            if key.startswith('_model.'):
                reni_model_dict[key[7:]] = ckpt['pipeline'][key]
        
        config_path = load_dir / 'config.yml'
        with open(config_path, 'r') as f:
            config = self.clean_and_load_yaml(f.read())
        
        # Determine model type and setup config
        if 'latent_dim' in config['pipeline']['model']['field'].keys():
            model_config = self._setup_reni_config(config)
        elif 'spherical_harmonic_order' in config['pipeline']['model']['field'].keys():
            model_config = self._setup_sh_sg_config(config, 'sh')
        elif 'row_col_gaussian_dims' in config['pipeline']['model']['field'].keys():
            model_config = self._setup_sh_sg_config(config, 'sg')
        else:
            raise ValueError(f"Unknown model type in config: {config_path}")
        
        # Handle missing test_mode (use 'inference' as default for evaluation)
        test_mode = config['pipeline'].get('test_mode', 'inference')
        model_config.pipeline.test_mode = test_mode
        
        pipeline: RENIPipeline = model_config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=1,
            local_rank=0,
            grad_scaler=None,
        )
        
        # CRITICAL: Explicitly setup train and eval dataloaders for the datamanager
        # These create the eval_image_dataloader needed by next_eval_image()
        pipeline.datamanager.setup_train()
        pipeline.datamanager.setup_eval()
        
        model = pipeline.model
        model.to(self.device)
        model.load_state_dict(reni_model_dict)
        
        # For SH/SG models that were trained with use_test_as_train=True,
        # we need to copy train_params to eval_params since the optimized
        # latent codes are stored in train_params but eval mode uses eval_params
        if hasattr(model.field, 'copy_train_to_eval'):
            model.field.copy_train_to_eval()
        
        model.eval()
        
        return pipeline, pipeline.datamanager, model
    
    def _setup_reni_config(self, config: Dict) -> Any:
        """Setup RENI model config from loaded config dict."""
        model_config = RENIField.config
        field_config = config['pipeline']['model']['field']
        dataparser_config = config['pipeline']['datamanager']['dataparser']
        
        # CRITICAL: Override the default VanillaDataManagerConfig with RENIDataManagerConfig
        # which has the next_eval_image() method we need for figure generation
        model_config.pipeline.datamanager = RENIDataManagerConfig(
            dataparser=RENIDataParserConfig(
                data=Path("data/RENI_HDR"),  # Will be replaced by actual data path
                convert_to_ldr=dataparser_config.get('convert_to_ldr', False),
                convert_to_log_domain=dataparser_config.get('convert_to_log_domain', True),
                min_max_normalize=tuple(dataparser_config['min_max_normalize']) if isinstance(dataparser_config.get('min_max_normalize'), list) else dataparser_config.get('min_max_normalize'),
                augment_with_mirror=dataparser_config.get('augment_with_mirror', False),
                use_validation_as_train=dataparser_config.get('use_validation_as_train', False),
                val_in_ldr=dataparser_config.get('val_in_ldr', False),
            ),
            pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                full_image_per_batch=False,
                images_per_batch=1,
                is_equirectangular=True,
            ),
            images_on_gpu=True,
            masks_on_gpu=True,
            train_num_rays_per_batch=8192,
            eval_num_rays_per_batch=8192,
        )
        
        # Handle eval_mask_path separately
        if dataparser_config.get('eval_mask_path') is not None:
            eval_mask_path = Path(os.path.join(*dataparser_config['eval_mask_path']))
            model_config.pipeline.datamanager.dataparser.eval_mask_path = eval_mask_path
        
        model_config.pipeline.model.loss_inclusions = config['pipeline']['model']['loss_inclusions']
        
        # Copy field config
        for key in ['conditioning', 'invariant_function', 'equivariance', 'axis_of_invariance',
                    'positional_encoding', 'encoded_input', 'latent_dim', 'hidden_features',
                    'hidden_layers', 'mapping_layers', 'mapping_features', 'num_attention_heads',
                    'num_attention_layers', 'output_activation', 'last_layer_linear',
                    'trainable_scale', 'old_implementation']:
            if key in field_config:
                setattr(model_config.pipeline.model.field, key, field_config[key])
        
        return model_config
    
    def _setup_sh_sg_config(self, config: Dict, model_type: Literal['sh', 'sg']) -> Any:
        """Setup SH or SG model config from loaded config dict."""
        # Use appropriate base config
        if model_type == 'sh':
            model_config = SHField.config
        else:
            model_config = SGField.config
        
        field_config = config['pipeline']['model']['field']
        dataparser_config = config['pipeline']['datamanager']['dataparser']
        
        # Copy field-specific config
        if model_type == 'sh' and 'spherical_harmonic_order' in field_config:
            model_config.pipeline.model.field.spherical_harmonic_order = field_config['spherical_harmonic_order']
        elif model_type == 'sg' and 'row_col_gaussian_dims' in field_config:
            row_col = field_config['row_col_gaussian_dims']
            if isinstance(row_col, list):
                row_col = tuple(row_col)
            model_config.pipeline.model.field.row_col_gaussian_dims = row_col
        
        # Copy dataparser config
        model_config.pipeline.datamanager.dataparser.convert_to_ldr = dataparser_config.get('convert_to_ldr', False)
        model_config.pipeline.datamanager.dataparser.convert_to_log_domain = dataparser_config.get('convert_to_log_domain', True)
        
        if dataparser_config.get('eval_mask_path') is not None:
            eval_mask_path = Path(os.path.join(*dataparser_config['eval_mask_path']))
            model_config.pipeline.datamanager.dataparser.eval_mask_path = eval_mask_path
        else:
            model_config.pipeline.datamanager.dataparser.eval_mask_path = None
        
        if isinstance(dataparser_config.get('min_max_normalize'), list):
            model_config.pipeline.datamanager.dataparser.min_max_normalize = tuple(dataparser_config['min_max_normalize'])
        else:
            model_config.pipeline.datamanager.dataparser.min_max_normalize = dataparser_config.get('min_max_normalize')
        
        model_config.pipeline.datamanager.dataparser.augment_with_mirror = dataparser_config.get('augment_with_mirror', False)
        
        # Copy model loss inclusions
        if 'loss_inclusions' in config['pipeline']['model']:
            model_config.pipeline.model.loss_inclusions = config['pipeline']['model']['loss_inclusions']
        
        return model_config
    
    def _load_old_format(self, load_dir: Path):
        """Load model from old RENI format (.pt files).
        
        Old format has:
        - RENI.pt: Contains mu, log_var (for training images), and network weights (net.X.linear.weight/bias)
        - RENI_Latent.pt: Contains mu, log_var (for eval/test images) and network weights (same)
        - config.yaml: wandb config with model parameters
        """
        files_dir = load_dir / "files"
        
        reni_path = files_dir / "RENI.pt"
        reni_latent_path = files_dir / "RENI_Latent.pt"
        config_path = files_dir / "config.yaml"
        
        if not reni_path.exists():
            raise FileNotFoundError(f"RENI.pt not found in {files_dir}")
        
        # Load checkpoints
        train_ckpt = torch.load(reni_path, map_location=self.device)
        
        # Load eval checkpoint if exists, otherwise use train checkpoint for both
        if reni_latent_path.exists():
            eval_ckpt = torch.load(reni_latent_path, map_location=self.device)
        else:
            eval_ckpt = train_ckpt
        
        # Parse old config
        old_config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f.read())
                # Extract values from wandb format
                for k, v in raw_config.items():
                    if isinstance(v, dict) and 'value' in v:
                        old_config[k] = v['value']
                    elif not k.startswith('_'):
                        old_config[k] = v
        
        # Extract old model parameters with defaults
        latent_dim = old_config.get('ndims', 100)
        hidden_features = old_config.get('RENI_hidden_features', 128)
        hidden_layers = old_config.get('RENI_hidden_layers', 5)
        equivariance = old_config.get('equivariance', 'SO2')
        last_layer_linear = old_config.get('last_layer_linear', False)
        
        # Get number of training and eval images from latent codes
        num_train_data = train_ckpt['mu'].shape[0]
        num_eval_data = eval_ckpt['mu'].shape[0]
        
        logger.info(f"Loading old RENI checkpoint: latent_dim={latent_dim}, hidden={hidden_features}, "
                    f"layers={hidden_layers}, train_imgs={num_train_data}, eval_imgs={num_eval_data}")
        
        # Create new config matching old architecture
        # Old RENI used: Concat conditioning, GramMatrix invariance, NeRF encoding on directions
        model_config = RENIField.config
        model_config.pipeline.model.field.latent_dim = latent_dim
        model_config.pipeline.model.field.hidden_features = hidden_features
        model_config.pipeline.model.field.hidden_layers = hidden_layers
        model_config.pipeline.model.field.equivariance = equivariance
        model_config.pipeline.model.field.axis_of_invariance = 'y'
        model_config.pipeline.model.field.invariant_function = 'GramMatrix'
        model_config.pipeline.model.field.conditioning = 'Concat'
        model_config.pipeline.model.field.encoded_input = 'None'  # Old RENI doesn't use positional encoding
        model_config.pipeline.model.field.positional_encoding = 'NeRF'
        model_config.pipeline.model.field.mapping_layers = 5
        model_config.pipeline.model.field.last_layer_linear = last_layer_linear
        # Old RENI was trained on min-max normalized data, SIREN output is already in correct range
        model_config.pipeline.model.field.output_activation = 'None'
        model_config.pipeline.model.field.old_implementation = True
        
        # Setup datamanager config (use test set for eval)
        model_config.pipeline.datamanager = RENIDataManagerConfig(
            dataparser=RENIDataParserConfig(
                data=Path("data/RENI_HDR"),
                convert_to_ldr=False,
                convert_to_log_domain=True,
                # Old RENI used min-max normalization in log domain with these values
                min_max_normalize=(-18.0536, 11.4633),
            ),
            pixel_sampler=RENIEquirectangularPixelSamplerConfig(
                full_image_per_batch=True,
                images_per_batch=1,
                is_equirectangular=True,
            ),
        )
        
        model_config.pipeline.test_mode = 'inference'
        
        # Setup pipeline
        pipeline: RENIPipeline = model_config.pipeline.setup(
            device=self.device,
            test_mode='inference',
            world_size=1,
            local_rank=0,
            grad_scaler=None,
        )
        
        pipeline.datamanager.setup_train()
        pipeline.datamanager.setup_eval()
        
        model = pipeline.model
        model.to(self.device)
        
        # Map old network weights to new format
        # Old format: net.X.linear.weight/bias where X is layer index (0 to num_layers)
        # New format: field.network.net.X.linear.weight/bias (for Siren)
        new_state_dict = {}
        
        # Map network weights
        for key, value in train_ckpt.items():
            if key.startswith('net.'):
                # Map to new format: field.network.net.X.linear.weight/bias
                new_key = f'field.network.{key}'
                new_state_dict[new_key] = value
        
        # Map latent codes
        # Train latents
        new_state_dict['field.train_mu'] = train_ckpt['mu']
        new_state_dict['field.train_log_var'] = train_ckpt['log_var']
        
        # Eval latents (from RENI_Latent.pt if exists)
        new_state_dict['field.eval_mu'] = eval_ckpt['mu']
        new_state_dict['field.eval_log_var'] = eval_ckpt['log_var']
        
        # Load the mapped weights
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Loaded old RENI weights successfully")
        except Exception as e:
            logger.warning(f"Partial weight loading: {e}")
            # Try loading what we can
            model_state = model.state_dict()
            for key in new_state_dict:
                if key in model_state and new_state_dict[key].shape == model_state[key].shape:
                    model_state[key] = new_state_dict[key]
            model.load_state_dict(model_state)
        
        model.eval()
        
        return pipeline, pipeline.datamanager, model


class FigureGenerator:
    """Generates all publication figures."""
    
    def __init__(self, config: FigureConfig):
        self.config = config
        self.loader = ModelLoader(config)
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to project root for relative imports
        os.chdir(self.config.project_root)
    
    def _get_checkpoint_path(self, *parts: str) -> Path:
        """Get full checkpoint path from parts."""
        return self.config.checkpoint_base / Path(*parts)
    
    def _check_checkpoint_exists(self, path: Path, figure_name: str) -> bool:
        """Check if checkpoint exists, log warning if not."""
        if not path.exists():
            logger.warning(f"Skipping {figure_name}: checkpoint not found at {path}")
            return False
        return True
    
    def _save_figure(self, fig: plt.Figure, name: str, dpi: int = 150):
        """Save figure as both PNG and PDF."""
        png_path = self.config.output_dir / f"{name}.png"
        pdf_path = self.config.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        logger.info(f"Saved: {png_path.name} and {pdf_path.name}")
        plt.close(fig)
    
    def _generate_ray_bundle(self, model, idx: int, H: int, W: int):
        """Generate ray bundle for equirectangular camera."""
        cx = torch.tensor(W // 2, dtype=torch.float32).repeat(1)
        cy = torch.tensor(H // 2, dtype=torch.float32).repeat(1)
        fx = torch.tensor(H, dtype=torch.float32).repeat(1)
        fy = torch.tensor(H, dtype=torch.float32).repeat(1)
        
        c2w = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]], dtype=torch.float32)
        cameras = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.EQUIRECTANGULAR)
        
        ray_bundle = cameras.generate_rays(0).flatten().to(self.config.device)
        ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * idx
        
        return ray_bundle
    
    def _get_rotation_func(self, model):
        """Get appropriate rotation function for model."""
        if hasattr(model.field, 'old_implementation') and model.field.old_implementation:
            return rot_y
        return rot_z
    
    def generate_images_from_models(self, image_indices: List[int], model_paths: List[Path]) -> Dict:
        """Generate images from multiple models for comparison."""
        all_model_outputs = {}
        H = self.config.default_image_height
        W = H * 2
        
        for model_path in model_paths:
            if not self._check_checkpoint_exists(model_path, f"model at {model_path}"):
                continue
            
            model_name = model_path.name
            if 'masked' in str(model_path):
                model_name = model_name + '_masked'
            
            try:
                pipeline, datamanager, model = self.loader.load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model {model_path}: {e}")
                continue
            
            model_outputs = {}
            
            for idx in image_indices:
                model.eval()
                _, ray_bundle, batch = datamanager.next_eval_image(idx)
                
                ray_bundle = self._generate_ray_bundle(model, idx, H, W)
                batch['image'] = batch['image'].to(self.config.device)
                
                outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle, rotation=None)
                
                pred_img = model.field.unnormalise(outputs['rgb'])
                gt_image = model.field.unnormalise(batch['image'])
                
                gt_image = gt_image.reshape(H, W, 3)
                pred_img = pred_img.reshape(H, W, 3)
                
                # Create grayscale for heatmap
                gt_image_gray = torch.mean(gt_image, dim=-1, keepdim=True)
                pred_image_gray = torch.mean(pred_img, dim=-1, keepdim=True)
                
                gt_min, gt_max = gt_image_gray.min(), gt_image_gray.max()
                
                combined_log_heatmap = torch.cat([gt_image_gray, pred_image_gray], dim=1)
                combined_log_heatmap = colormaps.apply_depth_colormap(
                    combined_log_heatmap, near_plane=gt_min, far_plane=gt_max
                )
                
                gt_heatmap = combined_log_heatmap[:, :W, :]
                pred_heatmap = combined_log_heatmap[:, W:, :]
                
                pred_img = linear_to_sRGB(pred_img, use_quantile=True)
                gt_img = linear_to_sRGB(gt_image, use_quantile=True)
                
                if 'mask' in batch:
                    mask = batch["mask"].reshape(H, W, 1).expand_as(gt_img).to(self.config.device)
                    gt_img = gt_img * mask
                
                model_outputs[idx] = {
                    'pred_img': pred_img.cpu().detach(),
                    'gt_img': gt_img.cpu().detach(),
                    'pred_heatmap': pred_heatmap.cpu().detach(),
                    'gt_heatmap': gt_heatmap.cpu().detach(),
                    'min_max': (gt_min, gt_max)
                }
            
            all_model_outputs[model_name] = model_outputs
        
        return all_model_outputs
    
    def generate_comparison_figure(self):
        """Generate comparison figure: RENI vs SH vs SG."""
        logger.info("Generating comparison figure...")
        
        image_indices = [1, 2, 3, 4]
        model_paths = [
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_9'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_49'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100'),
            self._get_checkpoint_path('spherical_harmonics', '2nd_order'),
            self._get_checkpoint_path('spherical_harmonics', '6th_order'),
            self._get_checkpoint_path('spherical_harmonics', '9th_order'),
            self._get_checkpoint_path('spherical_gaussians', 'num_param_30'),
            self._get_checkpoint_path('spherical_gaussians', 'num_param_150'),
            self._get_checkpoint_path('spherical_gaussians', 'num_param_300'),
        ]
        
        # Filter to existing paths
        model_paths = [p for p in model_paths if p.exists()]
        if not model_paths:
            logger.warning("No models found for comparison figure")
            return
        
        output_images = self.generate_images_from_models(image_indices, model_paths)
        
        if not output_images:
            logger.warning("No model outputs generated for comparison figure")
            return
        
        # Plot figure
        fig = plt.figure(figsize=(26, 16))
        
        reni_tags = ['latent_dim_9', 'latent_dim_49', 'latent_dim_100']
        sh_tags = ['2nd_order', '6th_order', '9th_order']
        sg_tags = ['num_param_30', 'num_param_150', 'num_param_300']
        
        rows_per_idx = len(reni_tags) + 2
        cols_per_model = 4
        
        for i, idx in enumerate(image_indices):
            row_offset = (i // 2) * rows_per_idx
            col_offset = (i % 2) * cols_per_model
            
            # Find a valid model for GT
            gt_key = next((k for k in reni_tags if k in output_images), None)
            if gt_key is None:
                continue
            
            gt_img = output_images[gt_key][idx]['gt_img']
            gt_heatmap = output_images[gt_key][idx]['gt_heatmap']
            
            h, w, _ = gt_img.shape
            padding = int(h * 0.1)
            
            padded_gt_img = np.pad(gt_img, [(0, padding), (0, 0), (0, 0)], mode='constant', constant_values=1)
            padded_gt_heatmap = np.pad(gt_heatmap, [(0, padding), (0, 0), (0, 0)], mode='constant', constant_values=1)
            
            ax = plt.subplot2grid((2*rows_per_idx, 2*cols_per_model), (row_offset, col_offset), rowspan=2, colspan=2)
            ax.imshow(padded_gt_img)
            ax.axis('off')
            ax.set_aspect(1)
            
            ax = plt.subplot2grid((2*rows_per_idx, 2*cols_per_model), (row_offset, col_offset+2), rowspan=2, colspan=2)
            ax.imshow(padded_gt_heatmap)
            ax.axis('off')
            ax.set_aspect(1)
            
            for j in range(3):
                # RENI images
                if reni_tags[j] in output_images and idx in output_images[reni_tags[j]]:
                    ax = plt.subplot2grid((2*rows_per_idx, 2*cols_per_model), (row_offset + 2 + j, col_offset))
                    ax.imshow(output_images[reni_tags[j]][idx]['pred_img'])
                    ax.axis('off')
                    ax.set_aspect(1)
                    
                    ax = plt.subplot2grid((2*rows_per_idx, 2*cols_per_model), (row_offset + 2 + j, col_offset + 1))
                    ax.imshow(output_images[reni_tags[j]][idx]['pred_heatmap'])
                    ax.axis('off')
                    ax.set_aspect(1)
                
                # SH images
                if sh_tags[j] in output_images and idx in output_images[sh_tags[j]]:
                    ax = plt.subplot2grid((2*rows_per_idx, 2*cols_per_model), (row_offset + 2 + j, col_offset + 2))
                    ax.imshow(output_images[sh_tags[j]][idx]['pred_img'])
                    ax.axis('off')
                    ax.set_aspect(1)
                
                # SG images
                if sg_tags[j] in output_images and idx in output_images[sg_tags[j]]:
                    ax = plt.subplot2grid((2*rows_per_idx, 2*cols_per_model), (row_offset + 2 + j, col_offset + 3))
                    ax.imshow(output_images[sg_tags[j]][idx]['pred_img'])
                    ax.axis('off')
                    ax.set_aspect(1)
        
        plt.tight_layout()
        self._save_figure(fig, 'comparison')
    
    def generate_interpolation_figure(self):
        """Generate latent space interpolation figure."""
        logger.info("Generating interpolation figure...")
        
        model_path = self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100')
        if not self._check_checkpoint_exists(model_path, "interpolation figure"):
            return
        
        try:
            pipeline, datamanager, model = self.loader.load_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model for interpolation: {e}")
            return
        
        model.eval()
        H = self.config.high_res_image_height
        W = H * 2
        
        # Setup ray bundle
        idx = 0
        ray_bundle = self._generate_ray_bundle(model, idx, H, W)
        ray_samples = model.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)
        
        def lerp(start, end, t):
            return start * (1 - t) + end * t
        
        K = 6  # Number of interpolation steps
        
        fig, axs = plt.subplots(3, K + 2, figsize=(20, 4))
        
        torch.manual_seed(54)
        torch.cuda.manual_seed_all(54)
        
        # Random interpolation rows
        for row in range(2):
            axs[row, -1].axis('off')
            axs[row, 0].axis('off')
            
            latent_code_1 = torch.randn(1, model.field.latent_dim, 3).to(self.config.device)
            latent_code_2 = torch.randn(1, model.field.latent_dim, 3).to(self.config.device)
            
            for col in range(1, K + 1):
                t = (col - 1) / (K - 1)
                interpolated = lerp(latent_code_1, latent_code_2, t)
                interpolated = interpolated.repeat(ray_samples.shape[0], 1, 1)
                
                field_outputs = model.field.forward(ray_samples, rotation=None, latent_codes=interpolated)
                image = field_outputs[RENIFieldHeadNames.RGB].reshape(H, W, 3)
                image = model.field.unnormalise(image)
                image = linear_to_sRGB(image, use_quantile=True)
                
                axs[row, col].imshow(image.cpu().detach().numpy())
                axs[row, col].axis('off')
        
        # Trained latent interpolation row
        idx1, idx2 = 6, 12
        latent_code_1 = model.field.eval_mu[idx1].unsqueeze(0)
        latent_code_2 = model.field.eval_mu[idx2].unsqueeze(0)
        
        for col, latent_code in zip([0, -1], [latent_code_1, latent_code_2]):
            pure = latent_code.repeat(ray_samples.shape[0], 1, 1)
            field_outputs = model.field.forward(ray_samples, rotation=None, latent_codes=pure)
            image = field_outputs[RENIFieldHeadNames.RGB].reshape(H, W, 3)
            image = model.field.unnormalise(image)
            image = linear_to_sRGB(image, use_quantile=True)
            
            axs[2, col].imshow(image.cpu().detach().numpy())
            axs[2, col].axis('off')
        
        for col in range(1, K + 1):
            t = (col - 1) / (K - 1)
            interpolated = lerp(latent_code_1, latent_code_2, t)
            interpolated = interpolated.repeat(ray_samples.shape[0], 1, 1)
            
            field_outputs = model.field.forward(ray_samples, rotation=None, latent_codes=interpolated)
            image = field_outputs[RENIFieldHeadNames.RGB].reshape(H, W, 3)
            image = model.field.unnormalise(image)
            image = linear_to_sRGB(image, use_quantile=True)
            
            axs[2, col].imshow(image.cpu().detach().numpy())
            axs[2, col].axis('off')
        
        plt.tight_layout()
        self._save_figure(fig, 'interpolations_and_random_samples')
    
    def generate_outpainting_figure(self):
        """Generate outpainting figure (masked vs unmasked)."""
        logger.info("Generating outpainting figure...")
        
        model_paths = [
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100'),
            self._get_checkpoint_path('reni_plus_plus_models', 'masked_models', 'latent_dim_100'),
        ]
        
        existing_paths = [p for p in model_paths if p.exists()]
        if len(existing_paths) < 2:
            logger.warning("Not enough models for outpainting figure (need both masked and unmasked)")
            return
        
        image_indices = [1, 2, 3, 4, 5, 6]
        output_images = self.generate_images_from_models(image_indices, existing_paths)
        
        keys = ['latent_dim_100', 'latent_dim_100_masked']
        if not all(k in output_images for k in keys):
            logger.warning(f"Missing required models for outpainting: have {list(output_images.keys())}")
            return
        
        n_rows = len(image_indices)
        n_cols = 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
        
        for i, idx in enumerate(image_indices):
            # Unmasked model
            axes[i, 0].imshow(output_images[keys[0]][idx]['gt_img'])
            axes[i, 0].axis('off')
            axes[i, 0].set_aspect(1)
            
            axes[i, 1].imshow(output_images[keys[0]][idx]['pred_img'])
            axes[i, 1].axis('off')
            axes[i, 1].set_aspect(1)
            
            # Masked model
            axes[i, 2].imshow(output_images[keys[1]][idx]['gt_img'])
            axes[i, 2].axis('off')
            axes[i, 2].set_aspect(1)
            
            axes[i, 3].imshow(output_images[keys[1]][idx]['pred_img'])
            axes[i, 3].axis('off')
            axes[i, 3].set_aspect(1)
        
        plt.tight_layout()
        self._save_figure(fig, 'outpainting')
    
    def generate_mirror_figure(self):
        """Generate mirror/reflection symmetry figure."""
        logger.info("Generating mirror figure...")
        
        model_path = self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100')
        if not self._check_checkpoint_exists(model_path, "mirror figure"):
            return
        
        try:
            pipeline, datamanager, model = self.loader.load_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model for mirror figure: {e}")
            return
        
        model.field.config.view_train_latents = True
        model.eval()
        
        H = self.config.default_image_height
        W = H * 2
        image_idx = 96
        
        ray_bundle = self._generate_ray_bundle(model, image_idx, H, W)
        ray_samples = model.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)
        
        def get_negated_image(Z, negate_axis=None):
            if negate_axis is not None:
                negate = torch.eye(3, dtype=torch.float32).type_as(Z)
                negate[negate_axis, negate_axis] = -1
                Z = torch.matmul(Z, negate)
            
            Z = Z.unsqueeze(0).repeat(ray_samples.shape[0], 1, 1)
            field_outputs = model.field.forward(ray_samples=ray_samples, latent_codes=Z)
            pred_img = model.field.unnormalise(field_outputs[RENIFieldHeadNames.RGB])
            pred_img = pred_img.view(H, W, 3)
            return linear_to_sRGB(pred_img, use_quantile=True).cpu().detach().numpy()
        
        Z = model.field.train_mu[image_idx]
        normal_img = get_negated_image(Z.clone())
        x_negated_img = get_negated_image(Z.clone(), 0)
        y_negated_img = get_negated_image(Z.clone(), 1)
        z_negated_img = get_negated_image(Z.clone(), 2)
        
        model.field.config.view_train_latents = False
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 5))
        
        axs[0, 0].axis('off')
        axs[0, 2].axis('off')
        
        axs[0, 1].imshow(normal_img)
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(x_negated_img)
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(z_negated_img)
        axs[1, 1].axis('off')
        
        axs[1, 2].imshow(y_negated_img)
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        self._save_figure(fig, 'mirror')
    
    def generate_teaser_figure(self):
        """Generate teaser figure with overlay on base image."""
        logger.info("Generating teaser figure...")
        
        if not self.config.teaser_base_path.exists():
            logger.warning(f"Teaser base image not found: {self.config.teaser_base_path}")
            return
        
        model_path = self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100')
        if not self._check_checkpoint_exists(model_path, "teaser figure"):
            return
        
        try:
            pipeline, datamanager, model = self.loader.load_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model for teaser: {e}")
            return
        
        model.eval()
        H = self.config.default_image_height
        W = H * 2
        
        num_random_samples = 5
        image_idx = 0
        images = {}
        
        get_rotation = self._get_rotation_func(model)
        
        for i in range(num_random_samples + 3):
            _, ray_bundle, batch = datamanager.next_eval_image(image_idx)
            ray_bundle = self._generate_ray_bundle(model, image_idx, H, W)
            ray_samples = model.create_ray_samples(ray_bundle.origins, ray_bundle.directions, ray_bundle.camera_indices)
            
            rot_angle = 90.0 if i == 1 else 0.0
            rotation = get_rotation(torch.tensor(np.deg2rad(rot_angle)).float()).to(self.config.device)
            
            if i in [0, 1]:
                outputs = model.field.forward(ray_samples=ray_samples, rotation=rotation)
                latent_code = model.field.train_mu[200].cpu()
                latent_code = torch.matmul(rotation.cpu(), latent_code.unsqueeze(-1)).squeeze(-1)
                latent_code_image = self._plot_latent_vectors(latent_code, 30)
            elif i == 2:
                latent_code = torch.zeros_like(model.field.train_mu[image_idx]).unsqueeze(0).to(self.config.device)
                latent_code = latent_code.repeat(ray_bundle.shape[0], 1, 1)
                outputs = model.field.forward(ray_samples=ray_samples, latent_codes=latent_code)
            else:
                latent_code_one = model.field.train_mu[i].unsqueeze(0).to(self.config.device)
                latent_code_two = model.field.train_mu[i+1].unsqueeze(0).to(self.config.device)
                latent_code = torch.lerp(latent_code_one, latent_code_two, 0.5)
                latent_code = latent_code.repeat(ray_bundle.shape[0], 1, 1)
                outputs = model.field.forward(ray_samples=ray_samples, latent_codes=latent_code)
            
            pred_img = model.field.unnormalise(outputs[RENIFieldHeadNames.RGB])
            pred_img = pred_img.reshape(H, W, 3)
            pred_img = linear_to_sRGB(pred_img, use_quantile=True)
            
            if i in [0, 1]:
                images[i] = {'pred_img': pred_img.cpu().detach(), 'latent_code_image': latent_code_image}
            else:
                images[i] = {'pred_img': pred_img.cpu().detach()}
        
        # Create teaser figure with overlays
        base_image = plt.imread(str(self.config.teaser_base_path))
        base_height, base_width = base_image.shape[:2]
        
        dpi = 100
        fig, ax = plt.subplots(figsize=(base_width/dpi, base_height/dpi), dpi=dpi)
        ax.imshow(base_image)
        ax.set_axis_off()
        
        image_properties = {
            'image1': {'x': 2875, 'y': 260, 'zoom': 3.5},
            'image2': {'x': 2875, 'y': 1095, 'zoom': 3.5},
            'image3': {'x': 630, 'y': 660, 'zoom': 1.7},
            'image4': {'x': 850, 'y': 250, 'zoom': 1.7},
            'image5': {'x': 180, 'y': 380, 'zoom': 1.7},
            'image6': {'x': 1150, 'y': 700, 'zoom': 1.7},
            'image7': {'x': 230, 'y': 950, 'zoom': 1.7},
            'image8': {'x': 900, 'y': 1100, 'zoom': 1.7},
            'latent_code1': {'x': 1450, 'y': 260, 'zoom': 0.8},
            'latent_code2': {'x': 1450, 'y': 1100, 'zoom': 0.8},
        }
        
        for i, img_data in images.items():
            img = img_data['pred_img']
            props = image_properties.get(f'image{i+1}', {'x': 0, 'y': 0, 'zoom': 1.0})
            
            imagebox = OffsetImage(img, zoom=props['zoom'])
            ab = AnnotationBbox(imagebox, (props['x'], props['y']), frameon=False, pad=0)
            ax.add_artist(ab)
            
            if 'latent_code_image' in img_data:
                lc_img = img_data['latent_code_image']
                lc_props = image_properties.get(f'latent_code{i+1}', {'x': 0, 'y': 0, 'zoom': 1.0})
                
                lc_imagebox = OffsetImage(lc_img, zoom=lc_props['zoom'])
                lc_ab = AnnotationBbox(lc_imagebox, (lc_props['x'], lc_props['y']), frameon=False, pad=0)
                ax.add_artist(lc_ab)
        
        self._save_figure(fig, 'teaser', dpi=dpi)
    
    def _plot_latent_vectors(self, latent_code, num_vectors: int = 30) -> np.ndarray:
        """Plot 3D latent vectors and return as image array."""
        if latent_code.dim() > 2:
            latent_code_np = latent_code.cpu().detach().numpy().squeeze()
        else:
            latent_code_np = latent_code.cpu().detach().numpy()
        
        latent_code_np = latent_code_np[:num_vectors]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        origin = np.zeros((len(latent_code_np), 3))
        cmap = get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(latent_code_np)))
        
        ax.quiver(origin[:, 0], origin[:, 1], origin[:, 2],
                  latent_code_np[:, 0], latent_code_np[:, 1], latent_code_np[:, 2],
                  color=colors, arrow_length_ratio=0.1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])
        ax.set_zlim([-2.0, 2.0])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='raw')
        buf.seek(0)
        
        image_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # Crop the image
        cropped = image_array[80:410, 157:500, :]
        
        plt.close(fig)
        return cropped
    
    def generate_numerical_results(self) -> Dict:
        """Generate numerical comparison results from all models."""
        logger.info("Generating numerical results...")
        
        model_paths = [
            # Old RENI (if available)
            self._get_checkpoint_path('reni_original', 'ndims_9'),
            self._get_checkpoint_path('reni_original', 'ndims_36'),
            self._get_checkpoint_path('reni_original', 'ndims_49'),
            self._get_checkpoint_path('reni_original', 'ndims_100'),
            # RENI++
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_9'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_36'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_49'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100'),
            # SH - 2nd, 5th, 6th, 9th order (D = 27, 108, 147, 300)
            self._get_checkpoint_path('spherical_harmonics', '2nd_order'),
            self._get_checkpoint_path('spherical_harmonics', '5th_order'),
            self._get_checkpoint_path('spherical_harmonics', '6th_order'),
            self._get_checkpoint_path('spherical_harmonics', '9th_order'),
            # SG - 30, 108, 150, 300 params (ceil(D/6)*6 gaussians)
            self._get_checkpoint_path('spherical_gaussians', 'num_param_30'),
            self._get_checkpoint_path('spherical_gaussians', 'num_param_108'),
            self._get_checkpoint_path('spherical_gaussians', 'num_param_150'),
            self._get_checkpoint_path('spherical_gaussians', 'num_param_300'),
        ]
        
        all_metrics = {}
        
        for model_path in model_paths:
            if not model_path.exists():
                continue
            
            model_name = model_path.name
            if 'reni_original' in str(model_path):
                model_name = model_name.replace('ndims_', 'latent_dim_') + '_old'
            
            try:
                pipeline, _, model = self.loader.load_model(model_path)
                
                if hasattr(model.field, 'latent_dim'):
                    metrics = pipeline.get_average_eval_image_metrics(optimise_latents=False)
                else:
                    metrics = pipeline.get_average_eval_image_metrics()
                
                all_metrics[model_name] = metrics
                logger.info(f"  {model_name}: PSNR={metrics.get('psnr_ldr', 'N/A'):.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to get metrics from {model_path}: {e}")
        
        return all_metrics
    
    def generate_old_vs_new_figure(self):
        """Generate old RENI vs RENI++ comparison figure."""
        logger.info("Generating old vs new figure...")
        
        image_indices = [6, 7, 8, 9, 10]
        
        # Old RENI models
        old_model_paths = [
            self._get_checkpoint_path('reni_original', 'ndims_9'),
            self._get_checkpoint_path('reni_original', 'ndims_49'),
            self._get_checkpoint_path('reni_original', 'ndims_100'),
        ]
        
        # New RENI++ models
        new_model_paths = [
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_9'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_49'),
            self._get_checkpoint_path('reni_plus_plus_models', 'latent_dim_100'),
        ]
        
        # Filter to existing paths
        old_model_paths = [p for p in old_model_paths if p.exists()]
        new_model_paths = [p for p in new_model_paths if p.exists()]
        
        if not old_model_paths or not new_model_paths:
            logger.warning("Not enough models for old vs new figure")
            return
        
        old_output_images = self.generate_images_from_models(image_indices, old_model_paths)
        new_output_images = self.generate_images_from_models(image_indices, new_model_paths)
        
        if not old_output_images or not new_output_images:
            logger.warning("No model outputs for old vs new figure")
            return
        
        # Use keys that exist in both
        old_keys = list(old_output_images.keys())
        new_keys = list(new_output_images.keys())
        
        # Map old keys to new keys for comparison
        key_mapping = {
            'ndims_9': 'latent_dim_9',
            'ndims_49': 'latent_dim_49', 
            'ndims_100': 'latent_dim_100',
        }
        
        n_rows = len(image_indices)
        n_cols = 1 + len(old_keys) * 2  # GT + old models + new models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10))
        
        for i, idx in enumerate(image_indices):
            # Ground truth from first old model
            first_key = old_keys[0]
            if idx in old_output_images[first_key]:
                gt_img = old_output_images[first_key][idx]['gt_img']
                axes[i, 0].imshow(gt_img)
            axes[i, 0].axis('off')
            axes[i, 0].set_aspect(1)
            if i == 0:
                axes[i, 0].set_title('Ground Truth', fontsize=10)

            for j, old_key in enumerate(old_keys):
                # Old model predictions
                if idx in old_output_images.get(old_key, {}):
                    pred_img_old = old_output_images[old_key][idx]['pred_img']
                    axes[i, j + 1].imshow(pred_img_old)
                axes[i, j + 1].axis('off')
                axes[i, j + 1].set_aspect(1)
                if i == 0:
                    axes[i, j + 1].set_title(f'RENI {old_key}', fontsize=10)
                
                # New model predictions  
                new_key = key_mapping.get(old_key, old_key.replace('ndims_', 'latent_dim_'))
                if new_key in new_output_images and idx in new_output_images[new_key]:
                    pred_img_new = new_output_images[new_key][idx]['pred_img']
                    axes[i, j + 1 + len(old_keys)].imshow(pred_img_new)
                axes[i, j + 1 + len(old_keys)].axis('off')
                axes[i, j + 1 + len(old_keys)].set_aspect(1)
                if i == 0:
                    axes[i, j + 1 + len(old_keys)].set_title(f'RENI++ {new_key}', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, 'old_vs_new')
    
    def generate_latex_table(self, output_metrics: Optional[Dict] = None):
        """Generate LaTeX table from numerical results."""
        if output_metrics is None:
            output_metrics = self.generate_numerical_results()
        
        if not output_metrics:
            logger.warning("No metrics available for LaTeX table")
            return
        
        def format_metric(value, best, is_better_high):
            if (is_better_high and value == best) or (not is_better_high and value == best):
                return f"\\textbf{{{value:.2f}}}"
            return f"{value:.2f}"
        
        # Map dimensions to model keys
        # Dimensions are based on the number of parameters for fair comparison
        # RENI/RENI++: latent_dim * 3 (latent codes are Nx3)
        # SH: (order+1)^2 coefficients
        # SG: num_gaussians * 6 (center + color per gaussian=2*3)
        dim_to_keys = {
            27: ['latent_dim_9_old', 'latent_dim_9', '2nd_order', 'num_param_30'],  # 9*3=27, SH: 9 coeffs, SG: 5 gaussians
            108: ['latent_dim_36_old', 'latent_dim_36', '5th_order', 'num_param_108'],  # 36*3=108, SH: 36 coeffs, SG: 18 gaussians
            147: ['latent_dim_49_old', 'latent_dim_49', '6th_order', 'num_param_150'],  # 49*3=147, SH: 49 coeffs, SG: 25 gaussians  
            300: ['latent_dim_100_old', 'latent_dim_100', '9th_order', 'num_param_300'],  # 100*3=300, SH: 100 coeffs, SG: 50 gaussians
        }
        
        lines = [
            "\\begin{table*}",
            "\\begin{center}",
            "\\caption{The mean PSNR, SSIM, and LPIPS scores when fitting to the test set for increasing latent dimensions.}",
            "\\label{tab:comparison_PSNR_SSIM_LPIPS}",
            "\\begin{tabular}{| c | c | c | c | c |}",
            "\\hline",
            "$D$ & RENI & RENI++ & SH & SG \\\\",
            "& (PSNR/SSIM/LPIPS) & (PSNR/SSIM/LPIPS) & (PSNR/SSIM/LPIPS) & (PSNR/SSIM/LPIPS) \\\\",
            "\\hline"
        ]
        
        for dim, model_keys in dim_to_keys.items():
            best_metrics = {
                'psnr_ldr': {'value': float('-inf'), 'is_better_high': True},
                'ssim_ldr': {'value': float('-inf'), 'is_better_high': True},
                'lpips_ldr': {'value': float('inf'), 'is_better_high': False}
            }
            
            for model_key in model_keys:
                if model_key in output_metrics:
                    for metric, info in best_metrics.items():
                        val = output_metrics[model_key].get(metric, None)
                        if val is None:
                            continue
                        if info['is_better_high'] and val > info['value']:
                            best_metrics[metric]['value'] = val
                        elif not info['is_better_high'] and val < info['value']:
                            best_metrics[metric]['value'] = val
            
            row = f"{dim} & "
            for idx, model_key in enumerate(model_keys):
                if model_key in output_metrics:
                    m = output_metrics[model_key]
                    formatted = {
                        k: format_metric(m.get(k, 0), best_metrics[k]['value'], best_metrics[k]['is_better_high'])
                        for k in ['psnr_ldr', 'ssim_ldr', 'lpips_ldr']
                    }
                    cell = f"{formatted['psnr_ldr']}/{formatted['ssim_ldr']}/{formatted['lpips_ldr']}"
                else:
                    cell = "N/A"
                
                if idx == len(model_keys) - 1:
                    row += f"{cell} \\\\ \\hline"
                else:
                    row += f"{cell} & "
            
            lines.append(row)
        
        lines.extend([
            "\\end{tabular}",
            "\\end{center}",
            "\\end{table*}"
        ])
        
        table = "\n".join(lines)
        
        # Save to file
        table_path = self.config.output_dir / "comparison_table.tex"
        with open(table_path, 'w') as f:
            f.write(table)
        logger.info(f"Saved: {table_path.name}")
        
        return table
    
    def run(self, figures: Optional[List[str]] = None):
        """Run figure generation for specified or all figures."""
        if figures is None:
            figures = self.config.figures
        
        all_figures = figures == ["all"] or "all" in figures
        
        figure_methods = {
            'comparison': self.generate_comparison_figure,
            'interpolation': self.generate_interpolation_figure,
            'outpainting': self.generate_outpainting_figure,
            'mirror': self.generate_mirror_figure,
            'teaser': self.generate_teaser_figure,
            'old_vs_new': self.generate_old_vs_new_figure,
            'table': self.generate_latex_table,
        }
        
        for name, method in figure_methods.items():
            if all_figures or name in figures:
                try:
                    method()
                except Exception as e:
                    logger.error(f"Failed to generate {name}: {e}")
                    import traceback
                    traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication figures for RENI++")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='publication/figures_updated',
        help='Output directory for figures (default: publication/figures_updated)'
    )
    parser.add_argument(
        '--checkpoint-base',
        type=str,
        default='checkpoints',
        help='Base directory for model checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--figures',
        type=str,
        default='all',
        help='Comma-separated list of figures to generate: comparison,interpolation,outpainting,mirror,teaser,table (default: all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='CUDA device (default: cuda:0)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = FigureConfig(
        output_dir=Path(args.output_dir),
        checkpoint_base=Path(args.checkpoint_base),
        device=args.device,
        figures=args.figures.split(','),
    )
    
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Checkpoint base: {config.checkpoint_base}")
    logger.info(f"Figures to generate: {config.figures}")
    
    generator = FigureGenerator(config)
    generator.run()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
