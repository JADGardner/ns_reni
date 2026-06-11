"""
InverseRenderNet_v2 PyTorch Implementation.

Pure PyTorch port of InverseRenderNet_v2 for environment map/lighting estimation.
Original paper: InverseRenderNet: Learning single image inverse rendering

This implementation focuses on the lighting estimation component, which outputs
2nd order (9 coefficient) spherical harmonics per color channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups, matching InverseRenderNet."""
    def __init__(self, num_channels: int):
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-5, affine=True)


class ConvGNReLU(nn.Module):
    """Conv2d + GroupNorm + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.gn = GroupNorm32(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class ConvBlock(nn.Module):
    """Conv2d without activation (for final layers)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InverseRenderNet(nn.Module):
    """
    PyTorch port of InverseRenderNet_v2 SfMNet.
    
    Encoder-decoder architecture with skip connections for predicting:
    - Albedo (H, W, 3)
    - Normal (H, W, 3)  
    - Shadow (H, W, 1)
    
    From these, Spherical Harmonic lighting coefficients are estimated.
    """
    
    def __init__(self, n_layers: int = 30, n_pools: int = 4, depth_base: int = 32):
        super().__init__()
        self.n_layers = n_layers
        self.n_pools = n_pools
        self.depth_base = depth_base
        
        conv_layers = n_layers // 2 - 1  # 14 for n_layers=30
        deconv_layers = n_layers // 2    # 15 for n_layers=30
        nlayers_bef_pool = int(np.ceil((conv_layers - 1) / n_pools) - 1)  # 3 for these params
        max_depth = 512
        
        # Calculate max_num_pool
        if depth_base * (2 ** n_pools) < max_depth:
            max_num_pool = n_pools
            tail = conv_layers - nlayers_bef_pool * n_pools
        else:
            max_num_pool = int(np.log2(max_depth / depth_base))  # 4 for depth_base=32
            tail = int(conv_layers - nlayers_bef_pool * max_num_pool)  # 2
            
        self.nlayers_bef_pool = nlayers_bef_pool
        self.max_num_pool = max_num_pool
        self.conv_layers = conv_layers
        self.deconv_layers = deconv_layers
        self.tail = tail
        
        # ========== ENCODER CHANNELS ==========
        # f_out_conv: output channels for each encoder layer
        # TF formula from SfMNet.py lines 48-58
        f_out_conv = (
            [64]  # First layer output is always 64
            + [
                int(depth_base * 2 ** (np.floor(i / nlayers_bef_pool)))
                for i in range(1, conv_layers - tail + 1)
            ]
            + [
                int(depth_base * 2 ** max_num_pool)
                for i in range(conv_layers - tail + 1, conv_layers + 1)
            ]
        )
        
        # f_in_conv: input channels for each encoder layer
        # First layer takes RGB (3), subsequent layers take output of previous
        f_in_conv = [3] + f_out_conv[:-1]
        
        # TF's formula-based f_in_conv (used only for decoder f_out computation)
        # This matches the original TF code which uses it for weight init stddev
        f_in_conv_tf = (
            [3]
            + [
                int(depth_base * 2 ** (np.ceil(i / nlayers_bef_pool) - 1))
                for i in range(1, conv_layers - tail + 1)
            ]
            + [
                int(depth_base * 2 ** max_num_pool)
                for i in range(conv_layers - tail + 1, conv_layers + 1)
            ]
        )
        
        self.f_in_conv = f_in_conv
        self.f_out_conv = f_out_conv
        
        # ========== DECODER CHANNELS ==========
        # f_out for decoders: TF uses f_in_conv_tf (formula-based) reversed
        # See SfMNet.py lines 61-63
        f_out_am_deconv = f_in_conv_tf[:0:-1] + [3]   # albedo ends with 3
        f_out_nm_deconv = f_in_conv_tf[:0:-1] + [2]   # normal ends with 2 (xy, z computed)
        f_out_mask_deconv = f_in_conv_tf[:0:-1] + [1] # shadow ends with 1
        
        # f_in for decoders: chain from encoder output, with skip concat doubling
        # First decoder layer gets encoder output (last element of f_out_conv)
        f_in_deconv = [f_out_conv[-1]]  # 512 for standard config
        
        for i in range(1, deconv_layers):
            prev_out = f_out_am_deconv[i-1]
            tf_prev_i = i  # 1-indexed previous layer number
            # Check if previous layer did a concat (layers 3, 6, 9, 12 for standard config)
            did_concat = (tf_prev_i % nlayers_bef_pool == 0 and 
                         tf_prev_i <= n_pools * nlayers_bef_pool)
            if did_concat:
                f_in_deconv.append(prev_out * 2)
            else:
                f_in_deconv.append(prev_out)
        
        self.f_in_deconv = f_in_deconv
        self.f_out_am_deconv = f_out_am_deconv
        self.f_out_nm_deconv = f_out_nm_deconv
        self.f_out_mask_deconv = f_out_mask_deconv
        
        # ========== BUILD ENCODER ==========
        self.encoder_layers = nn.ModuleList()
        for i in range(conv_layers + 1):
            in_c = f_in_conv[i]
            out_c = f_out_conv[i]
            self.encoder_layers.append(ConvGNReLU(in_c, out_c))
                
        # ========== BUILD DECODERS ==========
        self.albedo_decoder = self._build_decoder(f_in_deconv, f_out_am_deconv, 
                                                   deconv_layers, final_bias=True)
        self.normal_decoder = self._build_decoder(f_in_deconv, f_out_nm_deconv, 
                                                  deconv_layers, final_bias=False)
        self.shadow_decoder = self._build_decoder(f_in_deconv, f_out_mask_deconv, 
                                                  deconv_layers, final_bias=True)
        
    def _build_decoder(self, f_in: list, f_out: list, num_layers: int, 
                       final_bias: bool = True) -> nn.ModuleList:
        """Build a decoder branch."""
        layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_c = f_in[i]
            out_c = f_out[i]
            is_final = (i == num_layers - 1)
            
            if is_final:
                # Final layer: no normalization or activation
                layers.append(ConvBlock(in_c, out_c, bias=final_bias))
            else:
                layers.append(ConvGNReLU(in_c, out_c))
                
        return layers

    def _forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward through encoder, returning features and skip connections."""
        skip_features = []
        
        for i, layer in enumerate(self.encoder_layers):
            # TF uses 1-indexed loop, i goes from 1 to conv_layers+1
            tf_i = i + 1
            # Pool condition: (i-1) % nlayers_bef_pool == 0 and i <= n_pools * nlayers_bef_pool + 1 and i != 1
            do_pool = ((tf_i - 1) % self.nlayers_bef_pool == 0 and 
                       tf_i <= self.n_pools * self.nlayers_bef_pool + 1 and 
                       tf_i != 1)
            
            if do_pool:
                skip_features.append(x)
                x = layer(x)
                x = F.max_pool2d(x, 2, 2)
            else:
                x = layer(x)
                
        return x, skip_features
    
    def _forward_decoder(self, x: torch.Tensor, skip_features: list, 
                         decoder: nn.ModuleList) -> torch.Tensor:
        """Forward through a decoder branch."""
        # Skip features are stored in order: first skip corresponds to first pool
        # We use them in reverse order during decoding
        skip_idx = len(skip_features) - 1
        
        for i, layer in enumerate(decoder):
            tf_i = i + 1  # 1-indexed
            
            # Upsample + concat condition: i % nlayers_bef_pool == 0 and i <= n_pools * nlayers_bef_pool
            do_upsample = (tf_i % self.nlayers_bef_pool == 0 and 
                          tf_i <= self.n_pools * self.nlayers_bef_pool)
            
            if do_upsample and skip_idx >= 0:
                skip = skip_features[skip_idx]
                skip_idx -= 1
                # Upsample to match skip size
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = layer(x)
                x = torch.cat([x, skip], dim=1)
            else:
                x = layer(x)
                
        return x
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image (B, 3, H, W), normalized to [-1, 1]
            mask: Binary mask (B, 1, H, W)
            
        Returns:
            albedo: (B, 3, H, W), in [-1, 1]
            normal: (B, 3, H, W), unit vectors
            shadow: (B, 1, H, W), in [-1, 1]
        """
        # Encoder
        features, skip_features = self._forward_encoder(x)
        
        # Albedo decoder
        albedo_raw = self._forward_decoder(features, skip_features, self.albedo_decoder)
        albedo = torch.clamp(torch.tanh(albedo_raw) * mask, -0.9999, 0.9999)
        
        # Normal decoder (predicts x, y, computes z)
        nm_raw = self._forward_decoder(features, skip_features, self.normal_decoder)
        nm_norm = torch.sqrt(nm_raw.pow(2).sum(dim=1, keepdim=True) + 1.0)
        nm_xy = nm_raw / nm_norm
        nm_z = 1.0 / nm_norm
        normal = torch.cat([nm_xy, nm_z], dim=1) * mask
        
        # Shadow decoder
        shadow_raw = self._forward_decoder(features, skip_features, self.shadow_decoder)
        shadow = torch.clamp(torch.tanh(shadow_raw) * mask, -0.9999, 0.9999)
        
        return albedo, normal, shadow

    @staticmethod
    def estimate_lighting(image: torch.Tensor, albedo: torch.Tensor, 
                          normal: torch.Tensor, shadow: torch.Tensor, 
                          mask: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
        """
        Estimate 2nd order SH coefficients via least squares.
        
        Args:
            image: Input image (B, 3, H, W), in [-1, 1]
            albedo: Predicted albedo (B, 3, H, W), in [-1, 1]
            normal: Predicted normal (B, 3, H, W)
            shadow: Predicted shadow (B, 1, H, W), in [-1, 1]
            mask: Binary mask (B, 1, H, W)
            gamma: Gamma correction value
            
        Returns:
            sh_coeffs: (B, 9, 3) SH lighting coefficients
        """
        device = image.device
        B = image.shape[0]
        
        # Rescale from [-1,1] to [0,1]
        image_01 = image / 2.0 + 0.5
        albedo_01 = albedo / 2.0 + 0.5
        shadow_01 = shadow / 2.0 + 0.5
        
        # Apply gamma
        image_linear = torch.pow(image_01 * mask, gamma)
        
        # Compute D = albedo * shadow * mask
        D = albedo_01 * shadow_01 * mask
        
        # SH constants
        c1, c2, c3, c4, c5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708
        
        sh_coeffs_list = []
        
        for b in range(B):
            nm = normal[b].permute(1, 2, 0).reshape(-1, 3)  # (N, 3)
            img_pixels = image_linear[b].permute(1, 2, 0).reshape(-1, 3)  # (N, 3)
            d = D[b].permute(1, 2, 0).reshape(-1, 3)  # (N, 3)
            
            # Build A matrix (N, 9) for SH basis
            ones = torch.ones(nm.shape[0], device=device)
            A = torch.stack([
                c4 * ones,
                2 * c2 * nm[:, 1],
                2 * c2 * nm[:, 2],
                2 * c2 * nm[:, 0],
                2 * c1 * nm[:, 0] * nm[:, 1],
                2 * c1 * nm[:, 1] * nm[:, 2],
                c3 * nm[:, 2].pow(2) - c5,
                2 * c1 * nm[:, 2] * nm[:, 0],
                c1 * (nm[:, 0].pow(2) - nm[:, 1].pow(2)),
            ], dim=-1)  # (N, 9)
            
            # Solve for each channel via pseudo-inverse
            # Note: Use CPU for lstsq to avoid CUDA driver issues
            coeffs_rgb = []
            A_cpu = A.cpu()
            for c in range(3):
                # Weighted A matrix
                Ad = A_cpu * d[:, c:c+1].cpu()  # (N, 9)
                img_c = img_pixels[:, c:c+1].cpu()
                # Solve: L = pinv(A*D) @ I using numpy for stability
                try:
                    L = torch.linalg.lstsq(Ad, img_c).solution  # (9, 1)
                except Exception:
                    # Fallback to numpy
                    L_np = np.linalg.lstsq(Ad.numpy(), img_c.numpy(), rcond=None)[0]
                    L = torch.from_numpy(L_np).to(device)
                coeffs_rgb.append(L.to(device))
                
            sh_coeffs_list.append(torch.cat(coeffs_rgb, dim=-1))  # (9, 3)
            
        return torch.stack(sh_coeffs_list, dim=0)  # (B, 9, 3)


def load_pytorch_weights(model: InverseRenderNet, weight_path: str) -> None:
    """
    Load PyTorch weights from a .pth file.
    
    Args:
        model: PyTorch InverseRenderNet model
        weight_path: Path to the .pth weights file
    """
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded PyTorch weights from {weight_path}")
