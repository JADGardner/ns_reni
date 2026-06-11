#!/usr/bin/env python3
"""
Baseline Comparison Evaluation Script

Compares RENI++ against baseline methods (SOLD-Net, Hosek-Wilkie) on 
environment map reconstruction quality.

This script follows the same data loading pattern as generate_figures.py
to ensure consistent GT/prediction handling.

Usage:
    python baseline_comparison.py
    python baseline_comparison.py --output_dir figures_baseline
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import baseline models
from reni.baselines.soldnet import SOLDNetGlobalModel
from reni.baselines.hosek_wilkie import HosekWilkieSkyModel
from reni.utils.checkpoint_locator import find_checkpoint

# Import RENI++ loader and utilities (same as generate_figures.py)
from generate_figures import ModelLoader, FigureConfig
from reni.utils.colourspace import linear_to_sRGB

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comparison metrics between ground truth and prediction.
    
    All inputs should be in HDR linear space (not log, not normalized).
    
    Args:
        gt: Ground truth HDR image [H, W, 3]
        pred: Predicted HDR image [H, W, 3]
        mask: Optional binary mask for valid regions [H, W]
        
    Returns:
        Dictionary with metric values
    """
    # Ensure same shape
    assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"
    
    # Clip negative values (can occur from neural network outputs)
    gt = np.clip(gt, 0, None)
    pred = np.clip(pred, 0, None)
    
    # HDR metrics (clipped to reasonable range for PSNR computation)
    MAX_VAL = 100.0
    gt_clip = np.clip(gt, 0, MAX_VAL)
    pred_clip = np.clip(pred, 0, MAX_VAL)
    
    mse = np.mean((gt_clip - pred_clip) ** 2)
    mae = np.mean(np.abs(gt_clip - pred_clip))
    rmse = np.sqrt(mse)
    
    # PSNR (on clipped values)
    data_range = gt_clip.max() - gt_clip.min()
    if data_range > 0:
        psnr_val = psnr(gt_clip, pred_clip, data_range=data_range)
    else:
        psnr_val = 0.0
    
    # Log-domain MSE (more perceptually relevant for HDR)
    log_gt = np.log1p(gt)
    log_pred = np.log1p(pred)
    log_mse = np.mean((log_gt - log_pred) ** 2)
    
    # LDR metrics (tonemapped via sRGB conversion for display)
    # Use quantile-based exposure matching
    gt_tensor = torch.from_numpy(gt).float()
    pred_tensor = torch.from_numpy(pred).float()
    
    ldr_gt = linear_to_sRGB(gt_tensor, use_quantile=True).numpy()
    ldr_pred = linear_to_sRGB(pred_tensor, use_quantile=True).numpy()
    
    ldr_gt_uint8 = (np.clip(ldr_gt, 0, 1) * 255).astype(np.uint8)
    ldr_pred_uint8 = (np.clip(ldr_pred, 0, 1) * 255).astype(np.uint8)
    
    ldr_psnr = psnr(ldr_gt_uint8, ldr_pred_uint8, data_range=255)
    
    # SSIM on LDR
    ssim_val = ssim(ldr_gt_uint8, ldr_pred_uint8, channel_axis=2, data_range=255)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'PSNR': psnr_val,
        'LogMSE': log_mse,
        'LDR_PSNR': ldr_psnr,
        'SSIM': ssim_val,
    }


class BaselineComparison:
    """Compare RENI++ against baseline methods.
    
    Uses the same data loading pattern as generate_figures.py to ensure
    consistent GT/prediction handling through the pipeline datamanager.
    """
    
    def __init__(
        self,
        reni_checkpoint: str = "checkpoints/reni_plus_plus_models/latent_dim_100",
        soldnet_checkpoint: str = "checkpoints/SOLD_Net/pretrained_model",
        output_dir: str = "publication/figures_baseline",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load RENI++ model FIRST - we use its datamanager for GT
        logger.info("Loading RENI++...")
        config = FigureConfig()
        config.device = device
        self.reni_loader = ModelLoader(config)
        self.reni_checkpoint = find_checkpoint(reni_checkpoint)
        self.reni_pipeline = None
        self.reni_datamanager = None
        self.reni_model = None
        self._load_reni_model()
        
        # Load baseline models
        logger.info("Loading SOLD-Net...")
        self.soldnet = SOLDNetGlobalModel(device=device)
        self.soldnet.load_pretrained(str(find_checkpoint(soldnet_checkpoint)))
        
        logger.info("Loading Hosek-Wilkie...")
        self.hosek = HosekWilkieSkyModel()
        
        # Get number of eval images from datamanager
        if self.reni_pipeline is not None:
            self.n_eval_images = len(self.reni_pipeline.datamanager.eval_dataset)
            logger.info(f"Found {self.n_eval_images} evaluation images")
        else:
            self.n_eval_images = 0
    
    def _load_reni_model(self):
        """Load RENI++ model from checkpoint."""
        try:
            self.reni_pipeline, self.reni_datamanager, self.reni_model = \
                self.reni_loader.load_model(self.reni_checkpoint)
            self.reni_model.eval()
            logger.info(f"Loaded RENI++ from {self.reni_checkpoint}")
        except Exception as e:
            logger.error(f"Could not load RENI++ model: {e}")
            raise
    
    @torch.no_grad()
    def evaluate_soldnet_single(self, sky_hdr: np.ndarray) -> np.ndarray:
        """
        Run SOLD-Net reconstruction on a single sky hemisphere.
        
        Args:
            sky_hdr: [32, 128, 3] HDR sky hemisphere (linear space)
            
        Returns:
            [32, 128, 3] reconstructed HDR sky
        """
        # SOLD-Net expects [B, H, W, 3] in HDR linear space
        input_tensor = torch.from_numpy(sky_hdr).float().unsqueeze(0).to(self.device)
        # Use sky_only=True to avoid sun decoder blending artifacts
        recon, _ = self.soldnet.encode_decode(input_tensor, sky_only=True)
        return recon[0].cpu().numpy()
    
    def evaluate_hosek_fitted_single(self, sky_hdr: np.ndarray) -> np.ndarray:
        """
        Fit Hosek-Wilkie to sky image using gradient descent optimization.
        
        Uses a coarse grid search to find initial parameters, then refines
        with scipy L-BFGS-B optimization.
        
        Args:
            sky_hdr: [32, 128, 3] GT HDR sky hemisphere (linear space)
            
        Returns:
            [32, 128, 3] best-fit Hosek-Wilkie sky
        """
        from scipy.optimize import minimize
        
        H, W = sky_hdr.shape[:2]
        
        # Clip target to match metric computation (MAX_VAL=100)
        target_clip = np.clip(sky_hdr, 0, 100.0)
        
        def objective(params):
            """Objective function: linear MSE on clipped values (matches PSNR metric)."""
            sun_elev, sun_azim, turbidity, albedo, intensity = params
            
            envmap = self.hosek.generate_for_sun_elevation(
                sun_elevation_deg=sun_elev,
                sun_azimuth_deg=sun_azim,
                turbidity=turbidity,
                albedo=albedo,
                intensity=intensity,
                resolution=(H, W),
                return_torch=False,
                sky_only=True,
            )
            
            pred_clip = np.clip(envmap, 0, 100.0)
            return np.mean((target_clip - pred_clip) ** 2)
        
        # Stage 1: Coarse grid search for initial estimate
        best_loss = float('inf')
        best_init = None
        
        for sun_elev in [15, 35, 55, 75]:
            for sun_azim in [0, 90, 180, 270]:
                for turbidity in [2, 4, 7]:
                    # Quick intensity estimate
                    envmap = self.hosek.generate_for_sun_elevation(
                        sun_elevation_deg=sun_elev,
                        sun_azimuth_deg=sun_azim,
                        turbidity=turbidity,
                        intensity=1.0,
                        resolution=(H, W),
                        return_torch=False,
                        sky_only=True,
                    )
                    intensity = np.sum(sky_hdr * envmap) / (np.sum(envmap * envmap) + 1e-8)
                    intensity = np.clip(intensity, 0.01, 100.0)
                    
                    params = [sun_elev, sun_azim, turbidity, 0.1, intensity]
                    loss = objective(params)
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_init = params
        
        # Stage 2: Gradient descent optimization with L-BFGS-B
        # Bounds: [sun_elev, sun_azim, turbidity, albedo, intensity]
        bounds = [
            (5.0, 85.0),     # sun elevation (avoid exact horizon/zenith)
            (0.0, 360.0),    # sun azimuth (full circle)
            (1.0, 10.0),     # turbidity
            (0.0, 1.0),      # albedo
            (0.01, 1000.0),   # intensity
        ]
        
        result = minimize(
            objective,
            x0=best_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-6, 'eps': 1e-2},
        )
        
        # Generate final result with optimized parameters
        sun_elev, sun_azim, turbidity, albedo, intensity = result.x
        
        return self.hosek.generate_for_sun_elevation(
            sun_elevation_deg=sun_elev,
            sun_azimuth_deg=sun_azim,
            turbidity=turbidity,
            albedo=albedo,
            intensity=intensity,
            resolution=(H, W),
            return_torch=False,
            sky_only=True,
        )
    
    @torch.no_grad()
    def run_evaluation(self, num_images: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Run full evaluation on test set.
        
        Uses the pipeline datamanager for consistent GT/prediction handling.
        
        Args:
            num_images: Limit to first N images (for quick testing)
            
        Returns:
            (avg_metrics, images_data) tuple
        """
        if self.reni_model is None:
            raise RuntimeError("RENI++ model not loaded")
        
        n_images = min(num_images, self.n_eval_images) if num_images else self.n_eval_images
        
        # Storage for results
        metrics = {'RENI++': [], 'SOLD-Net': [], 'Hosek-Wilkie': []}
        images_data = {
            'gt': [],
            'RENI++': [],
            'SOLD-Net': [],
            'Hosek-Wilkie': [],
        }
        
        H_full = 64  # Full envmap height
        W = 128
        H_sky = 32   # Sky hemisphere height
        
        for i in tqdm(range(n_images), desc="Evaluating"):
            # Get GT and ray bundle from datamanager (like generate_figures.py)
            idx, ray_bundle, batch = self.reni_datamanager.next_eval_image(i)
            batch['image'] = batch['image'].to(self.device)
            ray_bundle = ray_bundle.to(self.device)
            
            # Get GT in HDR linear space
            gt_hdr = self.reni_model.field.unnormalise(batch['image'])
            gt_hdr = gt_hdr.reshape(H_full, W, 3).cpu().numpy()
            
            # RENI++ prediction
            outputs = self.reni_model.get_outputs_for_camera_ray_bundle(ray_bundle, rotation=None)
            reni_hdr = self.reni_model.field.unnormalise(outputs['rgb'])
            reni_hdr = reni_hdr.reshape(H_full, W, 3).cpu().numpy()
            
            # Extract sky hemisphere (top half) for baseline comparison
            gt_sky = gt_hdr[:H_sky, :, :]
            reni_sky = reni_hdr[:H_sky, :, :]
            
            # SOLD-Net reconstruction (only works on sky hemisphere)
            sold_sky = self.evaluate_soldnet_single(gt_sky)
            
            # Hosek-Wilkie fitted reconstruction
            hosek_sky = self.evaluate_hosek_fitted_single(gt_sky)
            
            # Store images (sky hemisphere for all methods)
            images_data['gt'].append(gt_sky)
            images_data['RENI++'].append(reni_sky)
            images_data['SOLD-Net'].append(sold_sky)
            images_data['Hosek-Wilkie'].append(hosek_sky)
            
            # Compute metrics (all in HDR linear space)
            metrics['RENI++'].append(compute_metrics(gt_sky, reni_sky))
            metrics['SOLD-Net'].append(compute_metrics(gt_sky, sold_sky))
            metrics['Hosek-Wilkie'].append(compute_metrics(gt_sky, hosek_sky))
        
        # Average metrics
        avg_metrics = {}
        for method, method_metrics in metrics.items():
            avg_metrics[method] = {}
            for key in method_metrics[0].keys():
                avg_metrics[method][key] = np.mean([m[key] for m in method_metrics])
        
        return avg_metrics, images_data
    
    def generate_comparison_figure(
        self,
        images_data: Dict,
        indices: List[int] = [0, 5, 10],
        save_name: str = "baseline_comparison.png",
        layout: str = "two_column",
    ):
        """Generate visual comparison figure.

        Args:
            layout: "two_column" for 2 major column groups (8 examples),
                    "single_column" for 1 column (5 examples).
        No text titles (added via tikz in LaTeX).
        """
        import matplotlib.gridspec as gridspec

        methods = ['GT', 'RENI++', 'SOLD-Net', 'Hosek-Wilkie']
        n_methods = len(methods)

        if layout == "two_column":
            half = (len(indices) + 1) // 2
            left_indices = indices[:half]
            right_indices = indices[half:]
            n_rows = max(len(left_indices), len(right_indices))

            fig = plt.figure(figsize=(n_methods * 2 * 1.8, n_rows * 0.48))
            outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.02)

            for group_idx, group_indices in enumerate([left_indices, right_indices]):
                inner = gridspec.GridSpecFromSubplotSpec(
                    len(group_indices), n_methods,
                    subplot_spec=outer[group_idx],
                    wspace=0.02, hspace=0.0,
                )
                for row, idx in enumerate(group_indices):
                    images = [
                        images_data['gt'][idx],
                        images_data['RENI++'][idx],
                        images_data['SOLD-Net'][idx],
                        images_data['Hosek-Wilkie'][idx],
                    ]
                    for col, img in enumerate(images):
                        ax = fig.add_subplot(inner[row, col])
                        img_tensor = torch.from_numpy(img).float()
                        display = linear_to_sRGB(img_tensor, use_quantile=True)
                        display = np.clip(display.numpy(), 0, 1)
                        ax.imshow(display)
                        ax.axis('off')
        else:  # single_column
            n_rows = len(indices)
            fig, axes = plt.subplots(
                n_rows, n_methods,
                figsize=(n_methods * 2.5, n_rows * 0.65),
                gridspec_kw={'wspace': 0.02, 'hspace': 0.0},
            )
            if n_rows == 1:
                axes = axes[np.newaxis, :]
            for row, idx in enumerate(indices):
                images = [
                    images_data['gt'][idx],
                    images_data['RENI++'][idx],
                    images_data['SOLD-Net'][idx],
                    images_data['Hosek-Wilkie'][idx],
                ]
                for col, img in enumerate(images):
                    ax = axes[row, col]
                    img_tensor = torch.from_numpy(img).float()
                    display = linear_to_sRGB(img_tensor, use_quantile=True)
                    display = np.clip(display.numpy(), 0, 1)
                    ax.imshow(display)
                    ax.axis('off')

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        logger.info(f"Saved {save_path}")
        
    def generate_metrics_table(self, metrics: Dict) -> str:
        """Generate LaTeX table of metrics."""
        lines = [
            r"\begin{tabular}{l|ccccc}",
            r"\hline",
            r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & MSE$\downarrow$ & LogMSE$\downarrow$ & LDR PSNR$\uparrow$ \\",
            r"\hline",
        ]
        
        for method, m in metrics.items():
            line = f"{method} & {m['PSNR']:.2f} & {m['SSIM']:.4f} & {m['MSE']:.4f} & {m['LogMSE']:.4f} & {m['LDR_PSNR']:.2f} \\\\"
            lines.append(line)
        
        lines.extend([
            r"\hline",
            r"\end{tabular}"
        ])
        
        table = "\n".join(lines)
        
        # Save to file
        table_path = self.output_dir / "baseline_metrics.tex"
        with open(table_path, 'w') as f:
            f.write(table)
        logger.info(f"Saved metrics table to {table_path}")
        
        return table
    
    def run(self, num_images: Optional[int] = None, image_indices: Optional[List[int]] = None):
        """Run full comparison pipeline."""
        logger.info("=" * 60)
        logger.info("Baseline Comparison Evaluation")
        logger.info("=" * 60)

        # Run evaluation
        metrics, images_data = self.run_evaluation(num_images)

        # Print metrics
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for method, m in metrics.items():
            print(f"\n{method}:")
            for key, value in m.items():
                print(f"  {key}: {value:.4f}")

        # Generate outputs
        logger.info("\nGenerating comparison figures...")
        n_available = len(images_data['gt'])
        if image_indices is not None:
            indices = [i for i in image_indices if i < n_available]
        else:
            indices = list(range(n_available))

        # Two-column version (8 examples)
        idx_8 = indices[:8] if len(indices) >= 8 else indices
        self.generate_comparison_figure(images_data, idx_8, save_name="baseline_comparison_2col.png", layout="two_column")

        # Single-column version (5 examples)
        idx_5 = indices[:5] if len(indices) >= 5 else indices
        self.generate_comparison_figure(images_data, idx_5, save_name="baseline_comparison_1col.png", layout="single_column")
        
        logger.info("Generating metrics table...")
        table = self.generate_metrics_table(metrics)
        print("\nLaTeX Table:")
        print(table)
        
        logger.info("\n" + "=" * 60)
        logger.info("Comparison complete!")
        logger.info(f"Outputs saved to: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison Evaluation")
    parser.add_argument("--reni_checkpoint", type=str, 
                       default="checkpoints/reni_plus_plus_models/latent_dim_100",
                       help="Path to RENI++ checkpoint")
    parser.add_argument("--soldnet_checkpoint", type=str,
                       default="checkpoints/SOLD_Net/pretrained_model",
                       help="Path to SOLD-Net pretrained models")
    parser.add_argument("--output_dir", type=str, default="publication/figures_baseline",
                       help="Output directory for figures")
    parser.add_argument("--num_images", type=int, default=None,
                       help="Limit to first N images (for quick testing)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on")
    parser.add_argument("--image_indices", type=int, nargs='+', default=None,
                       help="Specific image indices for the figure (e.g. --image_indices 0 2 5 7 9 11 15 18)")

    args = parser.parse_args()

    comparison = BaselineComparison(
        reni_checkpoint=args.reni_checkpoint,
        soldnet_checkpoint=args.soldnet_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )

    comparison.run(num_images=args.num_images, image_indices=args.image_indices)


if __name__ == "__main__":
    main()
