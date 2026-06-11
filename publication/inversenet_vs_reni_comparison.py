#!/usr/bin/env python3
"""
InverseRenderNet vs RENI++ Comparison Script

Compares InverseRenderNet SH lighting estimation against RENI++ environment map 
outpainting on the task of predicting full environment maps from narrow FoV crops.

InverseRenderNet estimates 2nd order (9 coefficient) spherical harmonics from a 
single image. RENI++ uses a learned generative prior to outpaint partial environment maps.

Usage:
    python inversenet_vs_reni_comparison.py
    python inversenet_vs_reni_comparison.py --num_images 5 --output_dir figures_inversenet
"""

import argparse
import functools
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LPIPS metric
try:
    from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

# Additional imports for RENI++ pipeline context
from torch.utils.tensorboard import SummaryWriter
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
    
# InverseRenderNet
from reni.baselines.inversenet import InverseRenderNet, load_pytorch_weights
from reni.utils.checkpoint_locator import find_checkpoint

# SH rendering utilities
from reni.illumination_fields.sh_illumination_field import shReconstructSignal

# RENI++ and utilities  
from reni.utils.colourspace import linear_to_sRGB

# For RENI++ pipeline
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from reni.configs.reni_config import RENIField
from reni.utils.utils import find_nerfstudio_project_root

# EXR loading
import OpenEXR
import Imath

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_fov_mask(
    output_size: Tuple[int, int],
    h_fov_deg: float = 90.0,
    v_fov_deg: float = 45.0,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
) -> np.ndarray:
    """
    Generate an equirectangular mask for a given FoV crop.
    
    This creates a binary mask where 1 indicates pixels visible in the 
    perspective crop (the conditioning region for RENI++).
    
    Args:
        output_size: (height, width) of the equirectangular mask
        h_fov_deg: Horizontal field of view in degrees
        v_fov_deg: Vertical field of view in degrees
        azimuth_deg: Horizontal viewing direction (0 = front)
        elevation_deg: Vertical viewing direction (0 = horizon)
        
    Returns:
        mask: Binary mask [H, W] with 1s in the visible region
    """
    H, W = output_size
    
    # Convert to radians
    h_fov = np.radians(h_fov_deg)
    v_fov = np.radians(v_fov_deg)
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    # Create a grid in the perspective image space
    # Use higher resolution for accurate mask
    out_h, out_w = 512, 512
    y, x = np.meshgrid(
        np.linspace(-1, 1, out_h),
        np.linspace(-1, 1, out_w),
        indexing='ij'
    )
    
    # Convert to 3D ray directions (perspective projection)
    z = np.ones_like(x)
    x = x * np.tan(h_fov / 2)
    y = -y * np.tan(v_fov / 2)
    
    # Stack and normalize
    dirs = np.stack([x, y, z], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    
    # Rotate by elevation (around x-axis)
    cos_e, sin_e = np.cos(elevation), np.sin(elevation)
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_e, -sin_e],
        [0, sin_e, cos_e]
    ])
    
    # Rotate by azimuth (around y-axis)
    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    rot_y = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Apply rotations
    rot = rot_y @ rot_x
    dirs_flat = dirs.reshape(-1, 3)
    dirs_rot = (rot @ dirs_flat.T).T
    dirs_rot = dirs_rot.reshape(out_h, out_w, 3)
    
    # Convert to spherical coordinates (equirectangular mapping)
    theta = np.arctan2(dirs_rot[..., 0], dirs_rot[..., 2])  # [-pi, pi]
    phi = np.arcsin(np.clip(dirs_rot[..., 1], -1, 1))  # [-pi/2, pi/2]
    
    # Map to equirectangular pixel coordinates
    u = (theta / np.pi + 1) / 2 * (W - 1)  # [0, W-1]
    v = (0.5 - phi / np.pi) * (H - 1)  # [0, H-1]
    
    # Create mask by marking all pixels that fall within bounds
    mask = np.zeros((H, W), dtype=np.float32)
    u_int = np.round(u).astype(int) % W
    v_int = np.clip(np.round(v).astype(int), 0, H - 1)
    mask[v_int.flatten(), u_int.flatten()] = 1.0
    
    # Dilate the mask slightly to fill gaps from discrete sampling
    from scipy import ndimage
    mask = ndimage.binary_dilation(mask, iterations=2).astype(np.float32)
    
    return mask


def save_fov_mask(
    output_path: Path,
    mask_size: Tuple[int, int],
    h_fov_deg: float = 90.0,
    v_fov_deg: float = 90.0,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
) -> Path:
    """
    Generate and save an FoV mask as a PNG file.
    
    Args:
        output_path: Path to save the mask
        mask_size: (height, width) of the mask
        h_fov_deg: Horizontal field of view in degrees
        v_fov_deg: Vertical field of view in degrees
        azimuth_deg: Horizontal viewing direction
        elevation_deg: Vertical viewing direction
        
    Returns:
        Path to the saved mask file
    """
    import cv2
    
    mask = generate_fov_mask(
        output_size=mask_size,
        h_fov_deg=h_fov_deg,
        v_fov_deg=v_fov_deg,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )
    
    # Convert to uint8 for saving as PNG
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask_uint8)
    
    logger.info(f"Saved FoV mask to {output_path}")
    return output_path


def load_exr(path: Path) -> np.ndarray:
    """Load HDR EXR image as float32 numpy array [H, W, 3]."""
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read RGB channels
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_data = []
    for channel in ['R', 'G', 'B']:
        channel_str = exr_file.channel(channel, pt)
        channel_array = np.frombuffer(channel_str, dtype=np.float32)
        channel_array = channel_array.reshape(height, width)
        rgb_data.append(channel_array)
    
    return np.stack(rgb_data, axis=-1)


def extract_fov_crop(
    envmap: np.ndarray, 
    h_fov_deg: float = 90.0,
    v_fov_deg: float = 45.0,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    output_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a perspective crop from an equirectangular environment map.
    
    Args:
        envmap: Equirectangular HDR image [H, W, 3]
        h_fov_deg: Horizontal field of view in degrees
        v_fov_deg: Vertical field of view in degrees
        azimuth_deg: Horizontal viewing direction (0 = front, 90 = right)
        elevation_deg: Vertical viewing direction (0 = horizon, 90 = up)
        output_size: (height, width) of output crop
        
    Returns:
        crop: Perspective crop [H_out, W_out, 3]
        mask: Binary mask of valid pixels in equirectangular [H, W]
    """
    H, W = envmap.shape[:2]
    out_h, out_w = output_size
    
    # Convert to radians
    h_fov = np.radians(h_fov_deg)
    v_fov = np.radians(v_fov_deg)
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    # Create output pixel grid
    y, x = np.meshgrid(
        np.linspace(-1, 1, out_h),
        np.linspace(-1, 1, out_w),
        indexing='ij'
    )
    
    # Convert to 3D ray directions (perspective projection)
    # Using pinhole camera model
    # Note: negate y because image y increases downward, but camera y is up
    z = np.ones_like(x)
    x = x * np.tan(h_fov / 2)
    y = -y * np.tan(v_fov / 2)  # Negated to fix upside-down issue
    
    # Stack and normalize
    dirs = np.stack([x, y, z], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    
    # Rotate by elevation (around x-axis)
    cos_e, sin_e = np.cos(elevation), np.sin(elevation)
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_e, -sin_e],
        [0, sin_e, cos_e]
    ])
    
    # Rotate by azimuth (around y-axis)
    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    rot_y = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Apply rotations
    rot = rot_y @ rot_x
    dirs_flat = dirs.reshape(-1, 3)
    dirs_rot = (rot @ dirs_flat.T).T
    dirs_rot = dirs_rot.reshape(out_h, out_w, 3)
    
    # Convert to spherical coordinates (equirectangular mapping)
    # theta = azimuth angle, phi = elevation angle
    theta = np.arctan2(dirs_rot[..., 0], dirs_rot[..., 2])  # [-pi, pi]
    phi = np.arcsin(np.clip(dirs_rot[..., 1], -1, 1))  # [-pi/2, pi/2]
    
    # Map to equirectangular pixel coordinates
    u = (theta / np.pi + 1) / 2 * (W - 1)  # [0, W-1]
    v = (0.5 - phi / np.pi) * (H - 1)  # [0, H-1]
    
    # Bilinear interpolation
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = u0 + 1, v0 + 1
    
    # Wrap u (azimuth)
    u0 = u0 % W
    u1 = u1 % W
    
    # Clamp v (elevation)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)
    
    # Interpolation weights
    wu = u - np.floor(u)
    wv = v - np.floor(v)
    
    # Sample
    c00 = envmap[v0, u0]
    c01 = envmap[v0, u1]
    c10 = envmap[v1, u0]
    c11 = envmap[v1, u1]
    
    crop = (
        c00 * (1 - wu)[..., None] * (1 - wv)[..., None] +
        c01 * wu[..., None] * (1 - wv)[..., None] +
        c10 * (1 - wu)[..., None] * wv[..., None] +
        c11 * wu[..., None] * wv[..., None]
    )
    
    # Create mask for visible region in equirectangular
    mask = np.zeros((H, W), dtype=np.float32)
    u_int = np.round(u).astype(int) % W
    v_int = np.clip(np.round(v).astype(int), 0, H - 1)
    mask[v_int.flatten(), u_int.flatten()] = 1.0
    
    return crop.astype(np.float32), mask


def compute_metrics(
    gt: np.ndarray, 
    pred: np.ndarray, 
    lpips_model: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute comparison metrics between ground truth and prediction.
    
    Args:
        gt: Ground truth HDR image [H, W, 3]
        pred: Predicted HDR image [H, W, 3]
        lpips_model: Optional LPIPS model for perceptual metric
        device: Device for LPIPS computation
        
    Returns:
        Dictionary with metric values
    """
    assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"
    
    gt = np.clip(gt, 0, None)
    pred = np.clip(pred, 0, None)
    
    # HDR PSNR (clipped to reasonable range)
    MAX_VAL = 100.0
    gt_clip = np.clip(gt, 0, MAX_VAL)
    pred_clip = np.clip(pred, 0, MAX_VAL)
    
    data_range = gt_clip.max() - gt_clip.min()
    if data_range > 0:
        psnr_val = psnr(gt_clip, pred_clip, data_range=data_range)
    else:
        psnr_val = 0.0
    
    # LDR metrics (tonemapped with consistent normalization)
    # Use GT's max for normalization so intensity differences are preserved
    gt_tensor = torch.from_numpy(gt).float()
    pred_tensor = torch.from_numpy(pred).float()
    
    # Find GT's normalization value (use 99.9th percentile to be robust to outliers)
    gt_max = torch.quantile(gt_tensor, 0.999).item()
    if gt_max < 1e-6:
        gt_max = 1.0  # Avoid division by zero
    
    # Normalize both using GT's range, then apply sRGB gamma
    ldr_gt = torch.clamp(gt_tensor / gt_max, 0, 1) ** (1/2.2)
    ldr_pred = torch.clamp(pred_tensor / gt_max, 0, 1) ** (1/2.2)
    
    ldr_gt = ldr_gt.numpy()
    ldr_pred = ldr_pred.numpy()
    
    ldr_gt_uint8 = (np.clip(ldr_gt, 0, 1) * 255).astype(np.uint8)
    ldr_pred_uint8 = (np.clip(ldr_pred, 0, 1) * 255).astype(np.uint8)
    
    ldr_psnr = psnr(ldr_gt_uint8, ldr_pred_uint8, data_range=255)
    ssim_val = ssim(ldr_gt_uint8, ldr_pred_uint8, channel_axis=2, data_range=255)
    
    metrics = {
        'PSNR': psnr_val,
        'LDR_PSNR': ldr_psnr,
        'SSIM': ssim_val,
    }
    
    # LPIPS 
    if lpips_model is not None:
        # LPIPS expects [B, 3, H, W] in range [0, 1]
        ldr_gt_t = torch.from_numpy(ldr_gt).float().permute(2, 0, 1).unsqueeze(0).to(device)
        ldr_pred_t = torch.from_numpy(ldr_pred).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Resize to at least 64x64 for LPIPS
        if ldr_gt_t.shape[-1] < 64 or ldr_gt_t.shape[-2] < 64:
            ldr_gt_t = F.interpolate(ldr_gt_t, size=(64, 128), mode='bilinear', align_corners=False)
            ldr_pred_t = F.interpolate(ldr_pred_t, size=(64, 128), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            lpips_val = lpips_model(ldr_gt_t, ldr_pred_t).item()
        metrics['LPIPS'] = lpips_val
    
    return metrics


class InverseRenderNetVsRENI:
    """Compare InverseRenderNet SH estimation against RENI++ outpainting."""
    
    def __init__(
        self,
        data_dir: str = "data/RENI_HDR/test",
        inversenet_weights: str = "checkpoints/inverserendernet/inversenet_weights.pth",
        reni_checkpoint: str = "checkpoints/reni_plus_plus_models/latent_dim_100",
        output_dir: str = "publication/figures_inversenet",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        envmap_width: int = 128,
        crop_h_fov: float = 90.0,
        crop_v_fov: float = 45.0,
        crop_azimuth: float = 0.0,
        crop_elevation: float = 0.0,
        crop_size: Tuple[int, int] = (256, 256),
        reni_fit_steps: int = 1000,
        custom_val_folder: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.envmap_width = envmap_width
        self.envmap_height = envmap_width // 2
        self.crop_h_fov = crop_h_fov
        self.crop_v_fov = crop_v_fov
        self.crop_azimuth = crop_azimuth
        self.crop_elevation = crop_elevation
        self.crop_size = crop_size
        self.reni_fit_steps = reni_fit_steps
        self.custom_val_folder = custom_val_folder
        
        # Load InverseRenderNet
        logger.info("Loading InverseRenderNet...")
        self.inversenet = InverseRenderNet().to(device)
        self.inversenet.eval()
        weight_path = find_checkpoint(inversenet_weights)
        if weight_path.exists():
            load_pytorch_weights(self.inversenet, str(weight_path))
        else:
            logger.warning(f"InverseRenderNet weights not found at {weight_path}")
        
        # Load RENI++
        logger.info("Loading RENI++...")
        self.reni_config = RENIField
        self.reni_checkpoint = find_checkpoint(reni_checkpoint)
        self.reni_pipeline = None
        self.reni_model = None
        self._load_reni_model()
        
        # LPIPS
        if HAS_LPIPS:
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        else:
            self.lpips = None
            logger.warning("LPIPS not available")
        
        # Find test images
        self.image_paths = sorted(self.data_dir.glob("*.exr"))
        logger.info(f"Found {len(self.image_paths)} test images")
    
    def _load_reni_model(self):
        """Load RENI++ model with FoV mask matching the crop parameters and actual image size."""
        try:
            # First, we need to determine the actual image size
            # Either from the custom_val_folder or from the data dir
            if self.custom_val_folder is not None:
                # Get image size from the first EXR in custom folder
                custom_path = Path(self.data_dir).parent / self.custom_val_folder
                exr_files = sorted(custom_path.glob("*.exr"))
                if exr_files:
                    first_exr = exr_files[0]
                    exr_file = OpenEXR.InputFile(str(first_exr))
                    header = exr_file.header()
                    dw = header['dataWindow']
                    mask_width = dw.max.x - dw.min.x + 1
                    mask_height = dw.max.y - dw.min.y + 1
                    logger.info(f"Detected image size from custom folder: {mask_width}x{mask_height}")
                else:
                    # Fall back to default
                    mask_height = self.envmap_height
                    mask_width = self.envmap_width
            else:
                # Get image size from the test folder
                exr_files = sorted(self.data_dir.glob("*.exr"))
                if exr_files:
                    first_exr = exr_files[0]
                    exr_file = OpenEXR.InputFile(str(first_exr))
                    header = exr_file.header()
                    dw = header['dataWindow']
                    mask_width = dw.max.x - dw.min.x + 1
                    mask_height = dw.max.y - dw.min.y + 1
                    logger.info(f"Detected image size from data dir: {mask_width}x{mask_height}")
                else:
                    # Fall back to default
                    mask_height = self.envmap_height
                    mask_width = self.envmap_width
            
            # Create a temporary mask file at the actual image resolution
            self.temp_mask_dir = tempfile.mkdtemp(prefix="reni_fov_mask_")
            mask_path = Path(self.temp_mask_dir) / "fov_mask.png"
            
            save_fov_mask(
                output_path=mask_path,
                mask_size=(mask_height, mask_width),
                h_fov_deg=self.crop_h_fov,
                v_fov_deg=self.crop_v_fov,
                azimuth_deg=self.crop_azimuth,
                elevation_deg=self.crop_elevation,
            )
            
            logger.info(f"Generated FoV mask at {mask_width}x{mask_height}: h_fov={self.crop_h_fov}°, v_fov={self.crop_v_fov}°, azimuth={self.crop_azimuth}°, elevation={self.crop_elevation}°")
            
            # Setup config
            self.reni_config.config.load_dir = self.reni_checkpoint / "nerfstudio_models"
            self.reni_config.config.load_step = 50000
            self.reni_config.config.pipeline.test_mode = "test"
            self.reni_config.config.pipeline.model_load_strict = False
            self.reni_config.config.vis = "tensorboard"
            
            # Set the FoV mask for evaluation
            self.reni_config.config.pipeline.datamanager.dataparser.eval_mask_path = mask_path
            # Use test data as training data (pipeline requires training data to initialize)
            self.reni_config.config.pipeline.datamanager.dataparser.use_test_as_train = True
            # Set resize_image_width to the detected width to prevent resizing
            self.reni_config.config.pipeline.datamanager.dataparser.resize_image_width = mask_width
            logger.info(f"RENI++ will use FoV mask from: {mask_path}")
            logger.info(f"Set resize_image_width to: {mask_width}")
            
            # Set custom validation folder if provided
            if self.custom_val_folder is not None:
                self.reni_config.config.pipeline.datamanager.dataparser.custom_val_folder = self.custom_val_folder
                logger.info(f"Using custom eval folder: {self.custom_val_folder}")
            
            trainer = self.reni_config.config.setup(local_rank=0, world_size=1)
            trainer.setup(test_mode="test")
            self.reni_pipeline = trainer.pipeline
            self.reni_model = self.reni_pipeline.model
            self.reni_model.eval()
            self.reni_model.fitting_eval_latents = True
            self.datamanager = self.reni_pipeline.datamanager
            logger.info("RENI++ loaded successfully with FoV mask conditioning")
        except Exception as e:
            logger.error(f"Could not load RENI++: {e}")
            import traceback
            traceback.print_exc()
            logger.info("RENI++ comparison will be skipped")
            self.reni_model = None
    
    @torch.no_grad()
    def run_inversenet(self, crop: np.ndarray, target_width: Optional[int] = None) -> np.ndarray:
        """
        Run InverseRenderNet on a crop to get SH coefficients, then render as envmap.
        
        Args:
            crop: Perspective crop [H, W, 3], HDR linear space
            target_width: Target width for environment map reconstruction. 
                         If None, uses self.envmap_width.
            
        Returns:
            envmap: Full equirectangular environment map [H, W, 3]
        """
        if target_width is None:
            target_width = self.envmap_width
            
        # Normalize to [-1, 1] for network
        crop_normalized = crop / (crop.max() + 1e-8) * 2 - 1
        crop_tensor = torch.from_numpy(crop_normalized).float().permute(2, 0, 1).unsqueeze(0)
        crop_tensor = crop_tensor.to(self.device)
        
        # Create a simple mask (all valid)
        mask = torch.ones(1, 1, crop.shape[0], crop.shape[1], device=self.device)
        
        # Forward pass
        albedo, normal, shadow = self.inversenet(crop_tensor, mask)
        
        # Estimate SH coefficients
        sh_coeffs = self.inversenet.estimate_lighting(
            crop_tensor, albedo, normal, shadow, mask
        )  # [1, 9, 3]
        
        # Render as full environment map at target resolution
        sh_coeffs = sh_coeffs[0]  # [9, 3]

        # Remap SH coeffs from InverseRenderNet camera space to RENI equirectangular space
        # IRN normal map: x=right, y=up, z=toward_viewer
        # RENI equirect:  x_reni=right, y_reni=forward, z_reni=up
        # Mapping: x_irn=x_reni, y_irn=z_reni, z_irn=-y_reni
        # IRN basis:  B1∝y_irn, B2∝z_irn, B3∝x_irn
        # RENI basis: B1∝-x_reni, B2∝z_reni, B3∝-y_reni
        remapped = torch.zeros_like(sh_coeffs)
        remapped[0] = sh_coeffs[0]
        # l=1
        remapped[1] = -sh_coeffs[3]   # x_irn=x_reni; B1_reni∝-x → negate
        remapped[2] = sh_coeffs[1]    # y_irn=z_reni; B2_reni∝z → direct
        remapped[3] = sh_coeffs[2]    # z_irn=-y_reni; B3_reni∝-y → direct
        # l=2
        remapped[4] = -sh_coeffs[7]                             # xz_irn → B4∝xy_reni
        remapped[5] = -sh_coeffs[4]                             # xy_irn → B5∝-xz_reni
        remapped[6] = -0.5 * (sh_coeffs[6] + sh_coeffs[8])     # diagonal mixing
        remapped[7] = sh_coeffs[5]                              # yz_irn → B7∝-yz_reni
        remapped[8] = 0.5 * (3 * sh_coeffs[6] - sh_coeffs[8])  # diagonal mixing
        sh_coeffs = remapped

        envmap = shReconstructSignal(sh_coeffs, width=target_width, device=self.device)
        envmap = envmap.cpu().numpy()
        
        # SH can produce negative values; clip
        envmap = np.clip(envmap, 0, None)
        
        return envmap
    
    def run_reni_outpainting(
        self, 
        image_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run RENI++ outpainting for a given evaluation image.
        
        This follows the pattern from outpainting_example.ipynb:
        1. Get ray bundle and batch from datamanager
        2. Create latent codes as parameters
        3. Optimize using model.get_outputs and model.get_loss_dict
        4. Unnormalize using model.field.unnormalise
        
        Args:
            image_idx: Index of eval image to fit
            
        Returns:
            Tuple of (gt_envmap, reni_envmap, mask) all [H, W, 3] or [H, W]
        """
        # Debug: log datamanager type
        logger.debug(f"Datamanager type: {type(self.datamanager).__name__}")
        if self.reni_model is None:
            return None, None, None
        
        device = self.device
        H = self.datamanager.eval_dataset.metadata['image_height']
        W = self.datamanager.eval_dataset.metadata['image_width']
        
        # Create latent codes to fit (following notebook pattern)
        latent_codes = torch.nn.Parameter(
            torch.zeros((len(self.datamanager.eval_dataset), self.reni_model.field.latent_dim, 3), 
                       requires_grad=True, device=device)
        )
        scale = torch.nn.Parameter(
            torch.ones((len(self.datamanager.eval_dataset)), requires_grad=True, device=device)
        )
        
        # Setup optimizer (following notebook pattern)
        optimizer_config = {
            "latents": {
                "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-7, max_steps=self.reni_fit_steps
                ),
            },
        }
        param_group = {"latents": [latent_codes, scale]}
        optimizer = Optimizers(optimizer_config, param_group)
        
        # Optimization loop (following notebook pattern)
        # For large images, we sample a subset of rays per step from the masked region
        ray_batch_size = 8192  # Number of rays to sample per optimization step
        
        for step in range(self.reni_fit_steps):
            # Get the image data (we'll sample rays ourselves)
            result = self.datamanager.next_eval_image(step)
            
            # Handle the return type (can be 2-tuple or 3-tuple depending on datamanager)
            if len(result) == 3:
                _, camera, batch = result
            else:
                camera, batch = result
            
            # Get image dimensions and actual image index
            actual_idx = int(batch['image_idx'])
            image = batch['image']  # [H*W, 3] or [H, W, 3]
            
            # Get mask if available
            if 'mask' in batch:
                mask = batch['mask'].flatten().bool()
            else:
                mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            
            # Get masked indices
            masked_indices = torch.where(mask)[0]
            if len(masked_indices) == 0:
                continue
            
            # Sample a random subset of masked rays
            num_samples = min(ray_batch_size, len(masked_indices))
            perm = torch.randperm(len(masked_indices), device=masked_indices.device)[:num_samples]
            sampled_indices = masked_indices[perm]
            
            # Generate rays for the sampled pixels
            if hasattr(camera, 'generate_rays'):
                H = camera.height[0].int().item()
                W = camera.width[0].int().item()
                full_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                full_ray_bundle = full_ray_bundle.flatten()
                # Select only sampled rays
                ray_bundle = full_ray_bundle[sampled_indices]
            else:
                ray_bundle = camera[sampled_indices]
            
            # Get sampled image pixels for loss
            sampled_indices_cpu = sampled_indices.cpu()
            sampled_pixels = image.view(-1, 3)[sampled_indices_cpu]
            sampled_batch = {'image': sampled_pixels}
            if 'mask' in batch:
                sampled_batch['mask'] = batch['mask'].view(-1)[sampled_indices_cpu]
            
            # Sample the corresponding latent codes and scale (single image)
            num_rays = ray_bundle.shape[0]
            latent_code_sample = latent_codes[actual_idx].unsqueeze(0).expand(num_rays, -1, -1)
            scale_sample = scale[actual_idx].unsqueeze(0).expand(num_rays)
            
            # Get model output
            model_outputs = self.reni_model.get_outputs(
                ray_bundle, rotation=None, 
                latent_codes=latent_code_sample, scale=scale_sample
            )
            
            # Apply LDR fitting if configured
            if self.reni_model.metadata.get("fit_val_in_ldr", False):
                model_outputs["rgb"] = linear_to_sRGB(
                    self.reni_model.field.unnormalise(model_outputs["rgb"])
                )
            
            # Get loss using model's loss function
            loss_dict = self.reni_model.get_loss_dict(model_outputs, sampled_batch, ray_bundle)
            loss = functools.reduce(torch.add, loss_dict.values())
            
            optimizer.zero_grad_all()
            loss.backward()
            optimizer.optimizer_step("latents")
            optimizer.scheduler_step("latents")
        
        # Get the specific image for evaluation
        eval_result = self.datamanager.next_eval_image(image_idx)
        if len(eval_result) == 3:
            returned_idx, ray_bundle, batch = eval_result
            logger.info(f"Requested image_idx={image_idx}, datamanager returned idx={returned_idx}")
        else:
            # Some datamanagers return only 2 values
            ray_bundle, batch = eval_result
            returned_idx = image_idx
            logger.info(f"Requested image_idx={image_idx}, got 2-tuple return")
            
        # Handle Cameras object if returned
        if isinstance(ray_bundle, Cameras):
            camera = ray_bundle
            ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
            # Reshape to flat if needed or keep shape? 
            # get_outputs usually expects flattened or shaped rays?
            # get_outputs calls create_ray_samples which handles it.
            # But the optimization loop used flattened rays?
            # Actually next_eval_image usually returns shaped rays (H, W) or (1, H, W)
            # if generate_rays(keep_shape=True), we get (H, W).
            pass

        # Get ground truth
        gt_img = batch['image'].reshape(H, W, 3)
        gt_img_hdr = self.reni_model.field.unnormalise(gt_img)
        mask = batch['mask'].reshape(H, W).cpu().numpy() if 'mask' in batch else np.ones((H, W))
        
        # Get the actual image index from the batch (this is the real index, not the one we requested)
        actual_image_idx = int(batch['image_idx'])
        logger.debug(f"Actual image index from batch: {actual_image_idx}")
        
        # Get RENI++ output with fitted latents (batched to avoid OOM on large images)
        with torch.no_grad():
            # Flatten ray_bundle first (needed for consistent shapes)
            ray_bundle_flat = ray_bundle.flatten()
            num_rays = ray_bundle_flat.shape[0]
            
            # Process in batches to avoid OOM
            eval_batch_size = 8192
            rgb_chunks = []
            
            for start_idx in range(0, num_rays, eval_batch_size):
                end_idx = min(start_idx + eval_batch_size, num_rays)
                chunk_size = end_idx - start_idx
                
                # Get ray chunk
                ray_chunk = ray_bundle_flat[start_idx:end_idx]
                
                # Use the actual image index from batch to get the correct fitted latent codes
                latent_code_chunk = latent_codes[actual_image_idx].unsqueeze(0).expand(chunk_size, -1, -1)
                scale_chunk = scale[actual_image_idx].unsqueeze(0).expand(chunk_size)
                
                chunk_outputs = self.reni_model.get_outputs(
                    ray_chunk, rotation=None,
                    latent_codes=latent_code_chunk, scale=scale_chunk
                )
                rgb_chunks.append(chunk_outputs["rgb"])
            
            # Concatenate all chunks
            rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
            rgb_hdr = self.reni_model.field.unnormalise(rgb)
        
        return gt_img_hdr.cpu().numpy(), rgb_hdr.cpu().numpy(), mask
    
    def run_evaluation(self, num_images: Optional[int] = None, image_indices: Optional[List[int]] = None) -> Tuple[Dict, Dict]:
        """
        Run full evaluation.

        Both methods are evaluated on the same images from RENI++ eval dataset:
        - InverseRenderNet: Extract FoV crop from GT envmap, predict full envmap from SH
        - RENI++: Use datamanager with masks, fit latent codes, predict full envmap
        """
        n_eval = len(self.datamanager.eval_dataset) if self.reni_model else 0

        if image_indices is not None:
            eval_indices = [i for i in image_indices if i < n_eval]
        elif num_images:
            eval_indices = list(range(min(num_images, n_eval)))
        else:
            eval_indices = list(range(n_eval))

        if len(eval_indices) == 0:
            logger.warning("No evaluation images available")
            return {}, {}

        metrics = {'InverseRenderNet': [], 'RENI++': []}
        images_data = {
            'crop': [],
            'gt': [],
            'conditioning': [],  # Equirectangular with only visible region
            'InverseRenderNet': [],
            'RENI++': [],
            'mask': [],
        }

        logger.info(f"Running evaluation on {len(eval_indices)} images (indices: {eval_indices})...")

        for i in tqdm(eval_indices, desc="Evaluating"):
            # Run RENI++ outpainting first to get the GT image
            gt_hdr, reni_envmap, mask = self.run_reni_outpainting(i)
            
            if gt_hdr is None:
                continue
            
            # Extract FoV crop from the SAME GT image for InverseRenderNet
            crop, crop_mask = extract_fov_crop(
                gt_hdr, 
                h_fov_deg=self.crop_h_fov,
                v_fov_deg=self.crop_v_fov,
                azimuth_deg=self.crop_azimuth,
                elevation_deg=self.crop_elevation,
                output_size=self.crop_size,
            )
            
            # Run InverseRenderNet on the crop - render SH at GT resolution
            gt_h, gt_w = gt_hdr.shape[:2]
            inversenet_envmap = self.run_inversenet(crop, target_width=gt_w)
            
            # Create conditioning image: GT with only visible crop, black elsewhere
            # Use the mask from RENI++ batch (our generated FoV mask) for consistency
            conditioning = gt_hdr * mask[..., np.newaxis]
            
            # Store results
            images_data['crop'].append(crop)
            images_data['gt'].append(gt_hdr)
            images_data['conditioning'].append(conditioning)
            images_data['InverseRenderNet'].append(inversenet_envmap)
            images_data['RENI++'].append(reni_envmap)
            images_data['mask'].append(mask)
            
            # Compute metrics for both methods against the SAME GT
            # Both outputs should now be at GT resolution
            metrics['InverseRenderNet'].append(
                compute_metrics(gt_hdr, inversenet_envmap, self.lpips, self.device)
            )
            metrics['RENI++'].append(
                compute_metrics(gt_hdr, reni_envmap, self.lpips, self.device)
            )        
        # Average metrics
        avg_metrics = {}
        for method, method_metrics in metrics.items():
            if len(method_metrics) > 0:
                avg_metrics[method] = {}
                for key in method_metrics[0].keys():
                    avg_metrics[method][key] = np.mean([m[key] for m in method_metrics])
        
        return avg_metrics, images_data
    
    def generate_comparison_figure(
        self,
        images_data: Dict,
        indices: Optional[List[int]] = None,
        save_name: str = "comparison.png",
    ):
        """Generate visual comparison figure."""
        if indices is None:
            indices = list(range(min(5, len(images_data['gt']))))

        n_images = len(indices)
        # 5 columns: Ground Truth, Input Crop, RENI++ Conditioning, InverseRenderNet (SH), RENI++ (Outpainting)
        fig, axes = plt.subplots(n_images, 5, figsize=(17, n_images * 2.2),
                                 gridspec_kw={'width_ratios': [2, 1, 2, 2, 2]})

        if n_images == 1:
            axes = axes.reshape(1, -1)

        titles = ['Ground Truth', 'Input Crop', 'RENI++ Conditioning', 'InverseRenderNet (SH)', 'RENI++ (Outpainting)']

        for row, idx in enumerate(indices):
            # Get the equirectangular height for padding reference
            gt_img = images_data['gt'][idx]

            # Resize crop to match equirectangular image height (no padding)
            crop_img = images_data['crop'][idx]
            eq_h, eq_w = gt_img.shape[:2]
            crop_h, crop_w = crop_img.shape[:2]
            scale = eq_h / crop_h
            new_h = eq_h
            new_w = int(crop_w * scale)
            p999 = np.percentile(crop_img, 99.9) + 1e-8
            crop_ldr = (np.clip(crop_img / p999, 0, 1) * 255).astype(np.uint8)
            crop_resized = np.array(Image.fromarray(crop_ldr).resize((new_w, new_h), Image.LANCZOS)).astype(np.float32) / 255.0
            crop_padded = crop_resized * p999

            images = [
                gt_img,
                crop_padded,
                images_data['conditioning'][idx],
                images_data['InverseRenderNet'][idx],
                images_data['RENI++'][idx],
            ]

            for col, (img, title) in enumerate(zip(images, titles)):
                # Convert to sRGB for display
                img_tensor = torch.from_numpy(img).float()
                display = linear_to_sRGB(img_tensor, use_quantile=True)
                display = np.clip(display.numpy(), 0, 1)

                axes[row, col].imshow(display)
                if row == 0:
                    axes[row, col].set_title(title, fontsize=10, fontweight='bold')
                axes[row, col].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {save_path}")
    
    def generate_metrics_table(self, metrics: Dict) -> str:
        """Generate LaTeX metrics table."""
        lines = [
            r"\begin{tabular}{l|cccc}",
            r"\hline",
            r"Method & PSNR$\uparrow$ & LDR PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\",
            r"\hline",
        ]
        
        for method, m in metrics.items():
            lpips_str = f"{m.get('LPIPS', float('nan')):.4f}"
            line = f"{method} & {m['PSNR']:.2f} & {m['LDR_PSNR']:.2f} & {m['SSIM']:.4f} & {lpips_str} \\\\"
            lines.append(line)
        
        lines.extend([r"\hline", r"\end{tabular}"])
        
        table = "\n".join(lines)
        
        table_path = self.output_dir / "metrics.tex"
        with open(table_path, 'w') as f:
            f.write(table)
        logger.info(f"Saved {table_path}")
        
        return table
    
    def run(self, num_images: Optional[int] = None, image_indices: Optional[List[int]] = None):
        """Run full comparison pipeline."""
        logger.info("=" * 60)
        logger.info("InverseRenderNet vs RENI++ Comparison")
        logger.info("=" * 60)

        metrics, images_data = self.run_evaluation(num_images, image_indices=image_indices)
        
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for method, m in metrics.items():
            print(f"\n{method}:")
            for key, value in m.items():
                print(f"  {key}: {value:.4f}")
        
        # Generate outputs
        logger.info("\nGenerating comparison figure...")
        self.generate_comparison_figure(images_data)
        
        logger.info("Generating metrics table...")
        table = self.generate_metrics_table(metrics)
        print("\nLaTeX Table:")
        print(table)
        
        logger.info("\n" + "=" * 60)
        logger.info("Comparison complete!")
        logger.info(f"Outputs saved to: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="InverseRenderNet vs RENI++ Comparison")
    parser.add_argument("--data_dir", type=str, default="data/RENI_HDR/test",
                       help="Directory with test HDR images")
    parser.add_argument("--inversenet_weights", type=str, 
                       default="checkpoints/inverserendernet/model_ckpt.pth",
                       help="Path to InverseRenderNet weights")
    parser.add_argument("--reni_checkpoint", type=str,
                       default="checkpoints/reni_plus_plus_models/latent_dim_100",
                       help="Path to RENI++ checkpoint")
    parser.add_argument("--output_dir", type=str, default="publication/figures_inversenet",
                       help="Output directory")
    parser.add_argument("--num_images", type=int, default=None,
                       help="Limit to first N images")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on")
    parser.add_argument("--crop_fov", type=float, default=120.0,
                       help="Horizontal FoV of crop in degrees")
    parser.add_argument("--crop_v_fov", type=float, default=120.0,
                       help="Vertical FoV of crop in degrees")
    parser.add_argument("--azimuth", type=float, default=80.0,
                       help="Azimuth angle for mask center in degrees (-180 to 180)")
    parser.add_argument("--elevation", type=float, default=-10.0,
                       help="Elevation angle for mask center in degrees (-90 to 90)")
    parser.add_argument("--reni_fit_steps", type=int, default=2500,
                       help="Number of latent fitting steps for RENI++")
    parser.add_argument("--custom_val_folder", type=str, default=None,
                       help="Custom folder for evaluation images (relative to data root)")
    parser.add_argument("--image_indices", type=int, nargs='+', default=None,
                       help="Specific image indices to evaluate (e.g. --image_indices 0 9 10 11 20)")

    args = parser.parse_args()
    
    comparison = InverseRenderNetVsRENI(
        data_dir=args.data_dir,
        inversenet_weights=args.inversenet_weights,
        reni_checkpoint=args.reni_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        crop_h_fov=args.crop_fov,
        crop_v_fov=args.crop_v_fov,
        crop_azimuth=args.azimuth,
        crop_elevation=args.elevation,
        reni_fit_steps=args.reni_fit_steps,
        custom_val_folder=args.custom_val_folder,
    )
    
    comparison.run(num_images=args.num_images, image_indices=args.image_indices)


if __name__ == "__main__":
    main()

# python publication/inversenet_vs_reni_comparison.py     --custom_val_folder "irn_test"     --reni_fit_steps 500     --crop_fov 120     --crop_v_fov 120     --azimuth 80     --elevation -10