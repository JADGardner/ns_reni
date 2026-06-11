#!/usr/bin/env python3
"""
Toy reconstruction script for Spherical Voronoi.

Fits an SV representation to a single environment map to validate
the basic premise that SV can represent high-frequency features.

Supports two modes:
- LDR mode: Matches the original Spherical Voronoi paper methodology
  (tonemapping, gamma correction, blur, MSE loss). Achieves ~37-38 dB PSNR.
- HDR mode: Fits directly to HDR values using log-space loss.

Usage:
    # LDR mode (matches paper, high PSNR)
    python scripts/sv_toy_reconstruction.py --target-envmap data/test_envmap.exr --mode ldr --num-sites 1024

    # HDR mode (preserves dynamic range)
    python scripts/sv_toy_reconstruction.py --target-envmap data/test_envmap.exr --mode hdr --num-sites 128
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reni.field_components.sv_primitives import (
    SphericalVoronoiRepresentation,
    fit_sv_to_envmap,
    hdr_to_ldr,
)


def load_exr(path: str) -> torch.Tensor:
    """Load an EXR file as a torch tensor."""
    try:
        import pyexr
        img = pyexr.open(path).get()
    except ImportError:
        try:
            import imageio
            img = imageio.v3.imread(path)
        except Exception as e:
            raise RuntimeError(f"Cannot load EXR file. Install pyexr or imageio: {e}")
    
    # Ensure RGB only
    if img.shape[-1] > 3:
        img = img[..., :3]
    
    return torch.from_numpy(img.astype(np.float32))


def save_exr(path: str, img: torch.Tensor):
    """Save a torch tensor as an EXR file."""
    img_np = img.detach().cpu().numpy().astype(np.float32)
    
    try:
        import pyexr
        pyexr.write(path, img_np)
    except ImportError:
        try:
            import imageio
            imageio.v3.imwrite(path, img_np)
        except Exception as e:
            # Fallback: save as PNG with tonemapping
            import matplotlib.pyplot as plt
            tonemapped = np.clip(img_np / (1 + img_np), 0, 1)
            plt.imsave(path.replace('.exr', '.png'), tonemapped)
            print(f"  (Saved as PNG due to missing EXR support: {e})")


def main():
    parser = argparse.ArgumentParser(description='Fit SV representation to environment map')
    parser.add_argument('--target-envmap', type=str, required=True,
                        help='Path to target HDR environment map (EXR)')
    parser.add_argument('--num-sites', type=int, default=1024,
                        help='Number of Voronoi sites (default: 1024)')
    parser.add_argument('--num-iterations', type=int, default=5000,
                        help='Optimization iterations (default: 5000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Base learning rate for HDR mode (default: 0.01)')
    parser.add_argument('--output-dir', type=str, default='outputs/sv_toy',
                        help='Output directory (default: outputs/sv_toy)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['hdr', 'ldr'], default='ldr',
                        help='Fitting mode: "ldr" matches paper methodology (default), "hdr" preserves dynamic range')

    # LDR-specific options
    parser.add_argument('--exposure', type=float, default=1.0,
                        help='Exposure for HDR->LDR tonemapping (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Gamma for HDR->LDR conversion (default: 2.2)')
    parser.add_argument('--blur-sigma', type=float, default=2.5,
                        help='Gaussian blur sigma for LDR target (default: 2.5, 0 to disable)')

    # Training options
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size for stochastic training (default: 4096, 0 for full-image)')
    parser.add_argument('--init-temperature', type=float, default=None,
                        help='Initial temperature/beta value (default: 256 for LDR, 128 for HDR)')
    parser.add_argument('--shared-temperature', action='store_true', default=None,
                        help='Use single shared beta (default: True for both modes)')
    parser.add_argument('--per-site-temperature', dest='shared_temperature', action='store_false',
                        help='Use per-site temperatures instead of shared beta')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading target: {args.target_envmap}")
    target = load_exr(args.target_envmap)
    print(f"  Shape: {target.shape}, Range: [{target.min():.3f}, {target.max():.3f}]")

    # Move to device
    device = torch.device(args.device)
    target = target.to(device)

    print(f"\nFitting SV with {args.num_sites} sites for {args.num_iterations} iterations...")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Device: {device}")
    if args.mode == 'ldr':
        print(f"  Exposure: {args.exposure}, Gamma: {args.gamma}, Blur: {args.blur_sigma}")

    batch_size = args.batch_size if args.batch_size > 0 else None

    sv, metrics = fit_sv_to_envmap(
        target,
        num_sites=args.num_sites,
        num_iterations=args.num_iterations,
        lr=args.lr,
        mode=args.mode,
        exposure=args.exposure,
        gamma=args.gamma,
        blur_sigma=args.blur_sigma,
        batch_size=batch_size,
        init_temperature=args.init_temperature,
        shared_temperature=args.shared_temperature,
        verbose=True,
    )
    
    print(f"\n=== Results ===")
    print(f"Final PSNR: {metrics['final_psnr']:.2f} dB")
    print(f"Final Loss: {metrics['final_loss']:.6f}")

    # Render and save result
    print(f"\nSaving outputs to {args.output_dir}/")

    height, width = target.shape[:2]
    rendered = sv.render_equirectangular(height, width)

    # Get the processed target that was actually used for fitting
    target_processed = metrics['target_processed']
    color_activation = metrics.get('color_activation', 'none')

    # For LDR mode (relu), clamp rendered output to [0, 1]
    if color_activation == 'relu':
        rendered = torch.clamp(rendered, 0, 1)

    # Save rendered result
    save_exr(os.path.join(args.output_dir, 'sv_reconstruction.exr'), rendered)

    # Save error map (against processed target)
    error = torch.abs(rendered - target_processed)
    save_exr(os.path.join(args.output_dir, 'sv_error.exr'), error)

    # Save comparison visualization
    try:
        import matplotlib.pyplot as plt

        if color_activation == 'relu':
            # LDR mode: images are already in [0, 1]
            target_vis = target_processed.detach().cpu().numpy()
            rendered_vis = rendered.detach().cpu().numpy()
            error_vis = error.detach().cpu().numpy()
        else:
            # HDR mode (softplus or none): tonemap for visualization
            def tonemap(x):
                x_np = x.detach().cpu().numpy()
                return np.clip(x_np / (1 + x_np), 0, 1)
            target_vis = tonemap(target_processed)
            rendered_vis = tonemap(rendered)
            error_vis = tonemap(error * 5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.clip(target_vis, 0, 1))
        axes[0].set_title(f'Target ({args.mode.upper()})')
        axes[0].axis('off')

        axes[1].imshow(np.clip(rendered_vis, 0, 1))
        axes[1].set_title(f'SV Reconstruction ({args.num_sites} sites)')
        axes[1].axis('off')

        # Error visualization
        if color_activation == 'relu':
            # Amplify error for visibility in LDR mode
            error_vis = np.clip(error_vis * 10, 0, 1)
        axes[2].imshow(error_vis)
        error_title = 'Error (10x)' if color_activation == 'relu' else 'Error (5x)'
        axes[2].set_title(f'{error_title} | PSNR: {metrics["final_psnr"]:.2f} dB')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'sv_comparison.png'), dpi=150)
        plt.close()

        print(f"  Saved comparison visualization")

        # Plot loss curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(metrics['losses'])
        ax.set_xlabel('Iteration')
        loss_label = 'Loss (MSE)' if args.mode == 'ldr' else 'Loss (log-space MSE)'
        ax.set_ylabel(loss_label)
        ax.set_title(f'SV Fitting Loss Curve ({args.mode.upper()} mode)')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'sv_loss_curve.png'), dpi=150)
        plt.close()

        print(f"  Saved loss curve")

    except ImportError:
        print("  (Matplotlib not available, skipping visualizations)")

    # Save model parameters
    model_state = {
        'sites': sv.sites.data.cpu(),
        'colors': sv.colors.data.cpu(),
        'num_sites': args.num_sites,
        'mode': args.mode,
        'color_activation': color_activation,
        'metrics': {k: v for k, v in metrics.items() if k != 'target_processed'},
    }
    # Save temperature params based on mode
    if hasattr(sv, 'log_temperatures'):
        model_state['log_temperatures'] = sv.log_temperatures.data.cpu()
    if hasattr(sv, '_beta_raw'):
        model_state['beta_raw'] = sv._beta_raw.data.cpu()
        model_state['beta'] = sv.get_beta().item()

    torch.save(model_state, os.path.join(args.output_dir, 'sv_model.pt'))
    print(f"  Saved model parameters")

    print("\nDone!")

    # Return success/failure based on PSNR threshold
    # LDR mode should achieve higher PSNR
    psnr_threshold = 30 if args.mode == 'ldr' else 20
    if metrics['final_psnr'] > psnr_threshold:
        print(f"✓ Reconstruction quality acceptable (PSNR > {psnr_threshold} dB)")
        return 0
    else:
        print(f"✗ Reconstruction quality below threshold (PSNR < {psnr_threshold} dB)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
