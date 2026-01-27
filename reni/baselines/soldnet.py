"""
SOLD-Net Wrapper for baseline comparison.

Wraps the SOLD-Net global lighting encoder-decoder for direct comparison
with RENI++ on environment map reconstruction quality.

SOLD-Net paper: "Estimating Spatially-Varying Lighting in Urban Scenes 
with Disentangled Representation" (Tang et al., ECCV 2022)

For fair comparison with RENI++, we use the global encoder-decoder only
(not the full spatially-varying estimator), which:
- Encodes HDR sky envmaps to disentangled sky_code + sun_code
- Decodes back to 128x32 equirectangular HDR envmap
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Add SOLD-Net to path
SOLDNET_ROOT = Path(__file__).parent.parent.parent / "SOLD-Net" / "encoding"
sys.path.insert(0, str(SOLDNET_ROOT))

from model.Autoencoder import GlobalEncoder, SkyDecoder, SunDecoder


class SOLDNetGlobalModel:
    """
    SOLD-Net Global Lighting Encoder-Decoder for environment map comparison.
    
    This wraps the global encoder-decoder from SOLD-Net which:
    - Encodes 128x32 HDR envmaps to disentangled latent (sky_dim=16, sun_dim=45)
    - Decodes back to 128x32 HDR envmap
    
    Usage:
        model = SOLDNetGlobalModel()
        model.load_pretrained('/path/to/pretrained_model')
        
        # Reconstruct an environment map
        recon, latent = model.encode_decode(envmap)
    """
    
    def __init__(
        self,
        sky_dim: int = 16,
        sun_dim: int = 45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize SOLD-Net global model.
        
        Args:
            sky_dim: Dimension of sky latent code
            sun_dim: Dimension of sun latent code (includes position)
            device: Device to run on
        """
        self.sky_dim = sky_dim
        self.sun_dim = sun_dim
        self.device = device
        
        # Create encoder-decoder models
        self.enc_sky = GlobalEncoder(cin=3, cout=sky_dim, activ='relu').to(device)
        self.enc_sun = GlobalEncoder(cin=3, cout=sun_dim, activ='relu').to(device)
        self.dec_sky = SkyDecoder(cin=sky_dim, cout=3, activ='relu').to(device)
        self.dec_sun = SunDecoder(cin=sun_dim, cout=3, activ='relu').to(device)
        
        self.loaded = False
        
    def load_pretrained(self, checkpoint_dir: str) -> None:
        """
        Load pretrained weights from checkpoint directory.
        
        Args:
            checkpoint_dir: Path to directory containing SOLD-Net checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        self.enc_sky.load_state_dict(
            torch.load(checkpoint_dir / "enc_sky_log_ft_distort_info.pth", map_location=self.device)
        )
        self.enc_sun.load_state_dict(
            torch.load(checkpoint_dir / "enc_sun_log_ft_distort_info.pth", map_location=self.device)
        )
        self.dec_sky.load_state_dict(
            torch.load(checkpoint_dir / "dec_sky_log_ft_distort_info.pth", map_location=self.device)
        )
        self.dec_sun.load_state_dict(
            torch.load(checkpoint_dir / "dec_sun_log_ft_distort_info.pth", map_location=self.device)
        )
        
        self.enc_sky.eval()
        self.enc_sun.eval()
        self.dec_sky.eval()
        self.dec_sun.eval()
        
        self.loaded = True
        print(f"Loaded SOLD-Net pretrained weights from {checkpoint_dir}")
        
    def _to_log_domain(self, img: torch.Tensor, mu: float = 16.0) -> torch.Tensor:
        """Convert HDR to log domain (SOLD-Net preprocessing)."""
        return torch.log(1 + mu * img) / np.log(1 + mu)
    
    def _from_log_domain(self, img: torch.Tensor, mu: float = 16.0) -> torch.Tensor:
        """Convert from log domain back to HDR."""
        return (torch.exp(img * np.log(1 + mu)) - 1) / mu
    
    def _create_sun_mask(self, envmap: torch.Tensor) -> torch.Tensor:
        """
        Create sun position mask based on brightest region in envmap.
        
        Args:
            envmap: [B, 3, H, W] or [B, 3, 32, 128] HDR environment map
            
        Returns:
            [B, 1, H, W] binary mask with 1s in 8x8 region around sun
        """
        B, C, H, W = envmap.shape
        
        # Find brightest pixel (sun location)
        brightness = envmap.mean(dim=1)  # [B, H, W]
        
        masks = []
        for b in range(B):
            brightness_b = brightness[b]
            # Find location of max
            flat_idx = brightness_b.argmax()
            y = (flat_idx // W).item()
            x = (flat_idx % W).item()
            
            # Create 8x8 mask around sun
            mask = torch.zeros(1, H, W, device=envmap.device)
            y_start = max(0, y - 3)
            y_end = min(H, y_start + 8)
            y_start = y_end - 8
            x_start = max(0, x - 3)
            x_end = min(W, x_start + 8)
            x_start = x_end - 8
            mask[0, y_start:y_end, x_start:x_end] = 1.0
            masks.append(mask)
            
        return torch.stack(masks, dim=0)
        
    @torch.no_grad()
    def encode(self, envmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode environment map to latent codes.
        
        Args:
            envmap: [B, H, W, 3] or [B, 3, H, W] HDR environment map (32x128)
            
        Returns:
            sky_code: [B, sky_dim] sky latent
            sun_code: [B, sun_dim] sun latent  
            sun_mask: [B, 1, H, W] sun position mask
        """
        assert self.loaded, "Must call load_pretrained() first"
        
        # Ensure [B, C, H, W] format
        if envmap.dim() == 3:
            envmap = envmap.unsqueeze(0)
        if envmap.shape[-1] == 3:  # [B, H, W, 3] -> [B, 3, H, W]
            envmap = envmap.permute(0, 3, 1, 2)
            
        envmap = envmap.to(self.device)
        
        # Convert to log domain
        envmap_log = self._to_log_domain(envmap.clamp(min=0))
        
        # Separate sky (masked sun) for sky encoder
        sun_mask = self._create_sun_mask(envmap)
        sky_input = envmap_log * (1 - sun_mask)
        
        # Encode
        sky_code = self.enc_sky(sky_input)
        sun_code = self.enc_sun(envmap_log)
        
        return sky_code, sun_code, sun_mask
    
    @torch.no_grad()
    def decode(
        self, 
        sky_code: torch.Tensor, 
        sun_code: torch.Tensor,
        sun_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent codes to environment map.
        
        Args:
            sky_code: [B, sky_dim]
            sun_code: [B, sun_dim]
            sun_mask: [B, 1, H, W]
            
        Returns:
            envmap: [B, 3, H, W] HDR environment map
        """
        assert self.loaded, "Must call load_pretrained() first"
        
        # Decode
        sky_recon_log = self.dec_sky(sky_code)
        sun_recon_log = self.dec_sun(sun_code, sun_mask)
        
        # Combine sky and sun (sun region from sun decoder, rest from sky)
        combined_log = sky_recon_log * (1 - sun_mask) + sun_recon_log * sun_mask
        
        # Convert back to HDR
        envmap = self._from_log_domain(combined_log.clamp(0, 4.5))
        return envmap.clamp(min=0)
    
    @torch.no_grad()
    def decode_sky_only(self, sky_code: torch.Tensor) -> torch.Tensor:
        """
        Decode using only the sky decoder (no sun blending artifacts).
        
        Args:
            sky_code: [B, sky_dim]
            
        Returns:
            envmap: [B, 3, H, W] HDR environment map
        """
        assert self.loaded, "Must call load_pretrained() first"
        
        sky_recon_log = self.dec_sky(sky_code)
        envmap = self._from_log_domain(sky_recon_log.clamp(0, 4.5))
        return envmap.clamp(min=0)
    
    @torch.no_grad()
    def encode_decode(self, envmap: torch.Tensor, sky_only: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Encode then decode environment map (reconstruction).
        
        Args:
            envmap: [B, H, W, 3] or [B, 3, H, W] HDR environment map
            sky_only: If True, use only sky decoder (avoids sun blending artifacts)
            
        Returns:
            recon: [B, H, W, 3] reconstructed HDR environment map
            latent: dict with 'sky_code', 'sun_code', 'sun_mask'
        """
        sky_code, sun_code, sun_mask = self.encode(envmap)
        
        if sky_only:
            recon = self.decode_sky_only(sky_code)
        else:
            recon = self.decode(sky_code, sun_code, sun_mask)
        
        # Convert back to [B, H, W, 3]
        recon = recon.permute(0, 2, 3, 1)
        
        return recon, {
            'sky_code': sky_code,
            'sun_code': sun_code,
            'sun_mask': sun_mask,
        }
    
    def get_latent_dim(self) -> int:
        """Total latent dimension (sky + sun)."""
        return self.sky_dim + self.sun_dim
        

def demo():
    """Demo SOLD-Net global encoder-decoder."""
    import matplotlib.pyplot as plt
    
    # Create model
    model = SOLDNetGlobalModel()
    model.load_pretrained("/home/james/github/ns_reni/checkpoints/SOLD_Net/pretrained_model")
    
    # Create random test input (32x128, matching SOLD-Net expectation)
    test_input = torch.rand(1, 32, 128, 3) * 5.0  # HDR range
    
    # Encode-decode
    recon, latent = model.encode_decode(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Sky code shape: {latent['sky_code'].shape}")
    print(f"Sun code shape: {latent['sun_code'].shape}")
    print(f"Total latent dim: {model.get_latent_dim()}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].imshow(test_input[0].cpu().numpy() ** 0.4)  # Gamma for display
    axes[0].set_title("Input")
    axes[0].axis('off')
    axes[1].imshow(recon[0].cpu().numpy() ** 0.4)
    axes[1].set_title("SOLD-Net Reconstruction")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig("soldnet_demo.png", dpi=150)
    print("Saved soldnet_demo.png")


if __name__ == "__main__":
    demo()
