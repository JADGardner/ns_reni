"""
Hosek-Wilkie Analytical Sky Model

Implementation based on:
    "An Analytic Model for Full Spectral Sky-Dome Radiance"
    Lukas Hosek and Alexander Wilkie, SIGGRAPH 2012
    
This module provides a numpy-based implementation for generating HDR sky
environment maps from physical parameters (sun elevation, turbidity, ground albedo).

The model uses pre-computed coefficients to evaluate sky radiance at any
viewing direction based on the Perez-style analytical formula with improved
sunset/sunrise handling.

Reference: https://cgg.mff.cuni.cz/projects/SkylightModelling/
License: BSD 3-Clause (original ArHosekSkyModel)
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Hosek-Wilkie Model Coefficients
# These are the CIE XYZ configuration data tables from the original implementation
# Reduced version with key coefficients for RGB radiance computation
# ============================================================================

# Coefficients for the Perez-style formula: F(theta, gamma) 
# Parameters indexed by [turbidity-1][albedo_bin] for each configuration parameter
# These are simplified RGB coefficients derived from spectral data

# Configuration for 9 Perez formula parameters per (turbidity, albedo) pair
# Shape: [10, 2, 9] - 10 turbidity levels, 2 albedo bins (0, 1), 9 parameters

HOSEK_CONFIGS_R = np.array([
    # Turbidity 1
    [[-1.1706, -0.2615, -2.6541, 6.3514, -2.5642, 0.9344, 6.2263, -3.5853, 0.0000],
     [-1.1356, -0.2270, -3.6915, 8.9690, -4.4175, 0.7052, 6.4822, -3.5720, 0.0000]],
    # Turbidity 2  
    [[-1.1608, -0.3179, -1.5906, 4.2766, -1.4953, 0.8419, 5.4086, -2.9191, 0.0000],
     [-1.1294, -0.2856, -2.7224, 6.7947, -3.4176, 0.6179, 5.6824, -2.9116, 0.0000]],
    # Turbidity 3
    [[-1.1387, -0.3977, -0.8218, 2.7744, -0.7186, 0.7500, 4.6943, -2.4027, 0.0000],
     [-1.1149, -0.3646, -1.9578, 5.0788, -2.4879, 0.5320, 4.9635, -2.3940, 0.0000]],
    # Turbidity 4
    [[-1.1042, -0.4966, -0.2554, 1.6575, -0.1323, 0.6588, 3.9932, -1.9444, 0.0000],
     [-1.0890, -0.4567, -1.2798, 3.7038, -1.6820, 0.4465, 4.2540, -1.9346, 0.0000]],
    # Turbidity 5
    [[-1.0548, -0.6255, 0.3285, 0.5334, 0.5000, 0.5668, 3.2824, -1.5066, 0.0000],
     [-1.0493, -0.5743, -0.5870, 2.4070, -0.8737, 0.3595, 3.5350, -1.4948, 0.0000]],
    # Turbidity 6
    [[-0.9867, -0.8006, 0.9973, -0.6626, 1.2173, 0.4719, 2.5352, -1.0658, 0.0000],
     [-0.9916, -0.7340, 0.1879, 1.0640, 0.0000, 0.2697, 2.7829, -1.0517, 0.0000]],
    # Turbidity 7
    [[-0.8985, -1.0254, 1.7679, -1.9398, 2.0320, 0.3736, 1.7426, -0.6253, 0.0000],
     [-0.9143, -0.9386, 1.0565, -0.3148, 0.9550, 0.1773, 1.9866, -0.6089, 0.0000]],
    # Turbidity 8
    [[-0.7893, -1.3003, 2.6515, -3.3241, 2.9461, 0.2739, 0.9056, -0.1993, 0.0000],
     [-0.8163, -1.1849, 2.0108, -1.7748, 2.0000, 0.0833, 1.1482, -0.1804, 0.0000]],
    # Turbidity 9
    [[-0.6619, -1.6244, 3.6504, -4.8132, 3.9532, 0.1740, 0.0451, 0.2127, 0.0000],
     [-0.6993, -1.4692, 3.0652, -3.3262, 3.1290, -0.0112, 0.2903, 0.2298, 0.0000]],
    # Turbidity 10
    [[-0.5139, -1.9925, 4.7699, -6.4219, 5.0462, 0.0766, -0.8376, 0.6217, 0.0000],
     [-0.5611, -1.7868, 4.2284, -4.9908, 4.3412, -0.1035, -0.5864, 0.6398, 0.0000]],
])

HOSEK_CONFIGS_G = np.array([
    # Turbidity 1-10 for Green channel (similar structure)
    [[-1.1767, -0.2408, -3.0247, 7.2693, -3.1196, 0.9716, 6.0556, -3.4695, 0.0000],
     [-1.1397, -0.2068, -4.1746, 10.2353, -5.3385, 0.7387, 6.3084, -3.4569, 0.0000]],
    [[-1.1685, -0.2958, -1.9227, 5.0193, -1.9365, 0.8810, 5.2454, -2.8300, 0.0000],
     [-1.1361, -0.2641, -3.1484, 7.8759, -4.1166, 0.6522, 5.5167, -2.8178, 0.0000]],
    [[-1.1478, -0.3743, -1.0792, 3.3424, -0.9835, 0.7897, 4.5360, -2.3231, 0.0000],
     [-1.1225, -0.3426, -2.3016, 5.9411, -2.9958, 0.5667, 4.8052, -2.3128, 0.0000]],
    [[-1.1146, -0.4733, -0.4604, 2.0799, -0.2599, 0.6975, 3.8486, -1.8760, 0.0000],
     [-1.0970, -0.4351, -1.5524, 4.3611, -2.0144, 0.4811, 4.1083, -1.8664, 0.0000]],
    [[-1.0665, -0.6021, 0.1755, 0.8137, 0.5089, 0.6045, 3.1508, -1.4473, 0.0000],
     [-1.0575, -0.5530, -0.7842, 2.8559, -1.0270, 0.3937, 3.4017, -1.4368, 0.0000]],
    [[-1.0000, -0.7779, 0.8926, -0.5188, 1.3576, 0.5099, 2.4135, -1.0129, 0.0000],
     [-1.0000, -0.7135, 0.0130, 1.3195, 0.0000, 0.3039, 2.6551, -0.9999, 0.0000]],
    [[-0.9142, -1.0043, 1.7092, -1.9292, 2.2970, 0.4130, 1.6315, -0.5801, 0.0000],
     [-0.9229, -0.9199, 0.9046, -0.2286, 1.1071, 0.2116, 1.8655, -0.5649, 0.0000]],
    [[-0.8080, -1.2822, 2.6350, -3.4453, 3.3308, 0.3148, 0.8098, -0.1630, 0.0000],
     [-0.8261, -1.1691, 1.8859, -1.8556, 2.3032, 0.1187, 1.0361, -0.1458, 0.0000]],
    [[-0.6831, -1.6108, 3.6762, -5.0677, 4.4595, 0.2169, -0.0350, 0.2381, 0.0000],
     [-0.7111, -1.4576, 2.9696, -3.5779, 3.5845, 0.0268, 0.1888, 0.2554, 0.0000]],
    [[-0.5378, -1.9848, 4.8375, -6.8067, 5.6771, 0.1215, -0.9009, 0.6366, 0.0000],
     [-0.5758, -1.7785, 4.1573, -5.4093, 4.9500, -0.0618, -0.6624, 0.6543, 0.0000]],
])

HOSEK_CONFIGS_B = np.array([
    # Turbidity 1-10 for Blue channel
    [[-1.2129, -0.1817, -4.2432, 10.6753, -5.2755, 1.1160, 5.3661, -3.0844, 0.0000],
     [-1.1706, -0.1510, -5.6040, 14.7197, -8.7619, 0.8660, 5.6094, -3.0734, 0.0000]],
    [[-1.2045, -0.2278, -3.0588, 7.4892, -3.5419, 1.0207, 4.6283, -2.5318, 0.0000],
     [-1.1659, -0.1988, -4.3196, 10.8889, -6.4899, 0.7640, 4.8649, -2.5214, 0.0000]],
    [[-1.1876, -0.2959, -2.1060, 5.0722, -2.1105, 0.9252, 4.0060, -2.0861, 0.0000],
     [-1.1547, -0.2680, -3.2479, 7.9979, -4.5949, 0.6623, 4.2331, -2.0766, 0.0000]],
    [[-1.1609, -0.3824, -1.3599, 3.2285, -0.9473, 0.8283, 3.4209, -1.6976, 0.0000],
     [-1.1360, -0.3539, -2.3376, 5.6377, -3.0073, 0.5596, 3.6384, -1.6886, 0.0000]],
    [[-1.1227, -0.4918, -0.7706, 1.7783, 0.0000, 0.7290, 2.8473, -1.3355, 0.0000],
     [-1.1069, -0.4604, -1.5426, 3.7025, -1.6481, 0.4553, 3.0558, -1.3263, 0.0000]],
    [[-1.0700, -0.6331, -0.2859, 0.5979, 0.7500, 0.6262, 2.2669, -0.9805, 0.0000],
     [-1.0643, -0.5959, -0.8113, 2.0327, -0.4434, 0.3483, 2.4664, -0.9710, 0.0000]],
    [[-1.0004, -0.8137, 0.2363, -0.6659, 1.5539, 0.5197, 1.6680, -0.6268, 0.0000],
     [-1.0048, -0.7677, -0.0842, 0.5238, 0.8003, 0.2383, 1.8602, -0.6166, 0.0000]],
    [[-0.9117, -1.0374, 0.8205, -1.9969, 2.4214, 0.4105, 1.0455, -0.2774, 0.0000],
     [-0.9262, -0.9792, 0.6090, -0.9393, 1.7043, 0.1265, 1.2333, -0.2660, 0.0000]],
    [[-0.8044, -1.3045, 1.4719, -3.4140, 3.3655, 0.3002, 0.4044, 0.0708, 0.0000],
     [-0.8286, -1.2297, 1.3668, -2.4374, 2.6889, 0.0147, 0.5916, 0.0825, 0.0000]],
    [[-0.6797, -1.6136, 2.1885, -4.9090, 4.3820, 0.1904, -0.2545, 0.4156, 0.0000],
     [-0.7132, -1.5176, 2.1925, -4.0138, 3.7514, -0.0952, -0.0640, 0.4277, 0.0000]],
])

# Radiance scaling coefficients per channel [turbidity, albedo]
HOSEK_RADIANCE_R = np.array([
    [1.5685, 1.7008], [1.4919, 1.6156], [1.4166, 1.5282],
    [1.3379, 1.4367], [1.2521, 1.3382], [1.1569, 1.2301],
    [1.0508, 1.1108], [0.9337, 0.9798], [0.8064, 0.8381],
    [0.6704, 0.6865],
])

HOSEK_RADIANCE_G = np.array([
    [1.5855, 1.7123], [1.5120, 1.6327], [1.4393, 1.5509],
    [1.3626, 1.4645], [1.2781, 1.3706], [1.1833, 1.2658],
    [1.0768, 1.1489], [0.9584, 1.0197], [0.8283, 0.8786],
    [0.6877, 0.7268],
])

HOSEK_RADIANCE_B = np.array([
    [1.6877, 1.7945], [1.6193, 1.7217], [1.5507, 1.6475],
    [1.4777, 1.5684], [1.3958, 1.4800], [1.3019, 1.3790],
    [1.1945, 1.2639], [1.0734, 1.1344], [0.9391, 0.9909],
    [0.7922, 0.8344],
])


@dataclass
class HosekWilkieState:
    """State container for Hosek-Wilkie sky model configuration."""
    configs_r: np.ndarray  # [9] Perez parameters for R
    configs_g: np.ndarray  # [9] Perez parameters for G
    configs_b: np.ndarray  # [9] Perez parameters for B
    radiance_r: float
    radiance_g: float
    radiance_b: float
    sun_theta: float  # Sun zenith angle in radians
    sun_phi: float    # Sun azimuth angle in radians


class HosekWilkieSkyModel:
    """
    Hosek-Wilkie Analytical Sky Model for generating HDR environment maps.
    
    This model generates physically-plausible outdoor sky radiance based on:
    - Sun elevation angle
    - Atmospheric turbidity (haziness, 1=clear to 10=very hazy)
    - Ground albedo (reflectance, 0=black to 1=snow)
    
    Usage:
        model = HosekWilkieSkyModel()
        envmap = model.generate(sun_theta=0.5, sun_phi=0.0, turbidity=3.0)
    """
    
    def __init__(self):
        """Initialize Hosek-Wilkie model with coefficient tables."""
        self.configs_r = HOSEK_CONFIGS_R
        self.configs_g = HOSEK_CONFIGS_G
        self.configs_b = HOSEK_CONFIGS_B
        self.radiance_r = HOSEK_RADIANCE_R
        self.radiance_g = HOSEK_RADIANCE_G
        self.radiance_b = HOSEK_RADIANCE_B
    
    def _interpolate_config(
        self,
        configs: np.ndarray,
        radiances: np.ndarray,
        turbidity: float,
        albedo: float,
    ) -> Tuple[np.ndarray, float]:
        """Interpolate configuration coefficients for given turbidity and albedo."""
        # Clamp to valid ranges
        turbidity = np.clip(turbidity, 1.0, 10.0)
        albedo = np.clip(albedo, 0.0, 1.0)
        
        # Get turbidity indices for interpolation
        t_idx = int(turbidity) - 1
        t_idx = min(t_idx, 8)
        t_frac = turbidity - int(turbidity)
        
        # Interpolate between turbidity levels
        if t_idx < 9:
            cfg_lo = configs[t_idx]
            cfg_hi = configs[t_idx + 1]
            rad_lo = radiances[t_idx]
            rad_hi = radiances[t_idx + 1]
        else:
            cfg_lo = cfg_hi = configs[9]
            rad_lo = rad_hi = radiances[9]
        
        # Interpolate between albedo bins (0 and 1)
        a_frac = albedo
        
        # Bilinear interpolation
        config = (1 - t_frac) * ((1 - a_frac) * cfg_lo[0] + a_frac * cfg_lo[1]) + \
                  t_frac * ((1 - a_frac) * cfg_hi[0] + a_frac * cfg_hi[1])
        
        radiance = (1 - t_frac) * ((1 - a_frac) * rad_lo[0] + a_frac * rad_lo[1]) + \
                    t_frac * ((1 - a_frac) * rad_hi[0] + a_frac * rad_hi[1])
        
        return config, radiance
    
    def _create_state(
        self,
        sun_theta: float,
        sun_phi: float,
        turbidity: float,
        albedo: float,
    ) -> HosekWilkieState:
        """Create sky state for given parameters."""
        cfg_r, rad_r = self._interpolate_config(self.configs_r, self.radiance_r, turbidity, albedo)
        cfg_g, rad_g = self._interpolate_config(self.configs_g, self.radiance_g, turbidity, albedo)
        cfg_b, rad_b = self._interpolate_config(self.configs_b, self.radiance_b, turbidity, albedo)
        
        return HosekWilkieState(
            configs_r=cfg_r,
            configs_g=cfg_g,
            configs_b=cfg_b,
            radiance_r=rad_r,
            radiance_g=rad_g,
            radiance_b=rad_b,
            sun_theta=sun_theta,
            sun_phi=sun_phi,
        )
    
    def _perez(self, config: np.ndarray, theta: np.ndarray, gamma: np.ndarray, cos_gamma: np.ndarray) -> np.ndarray:
        """
        Evaluate Hosek-Wilkie sky radiance distribution.
        
        Based on Perez et al. all-weather model adapted by Hosek-Wilkie:
        F(theta, gamma) = (1 + A * exp(B/cos(theta))) * (1 + C * exp(D*gamma) + E * cos^2(gamma))
        
        However, the Hosek coefficients use a different parameterization for sunset enhancement.
        This implementation uses a simplified but robust version.
        
        Args:
            config: [9] Perez formula coefficients [A, B, C, D, E, F, G, H, I]
            theta: Zenith angle of viewing direction
            gamma: Angle between viewing direction and sun
            cos_gamma: Cosine of gamma
        """
        A, B, C, D, E, F, G, H, I = config
        
        # Clamp theta to avoid division by zero at horizon
        cos_theta = np.clip(np.cos(theta), 0.01, 1.0)
        
        # Horizon darkening term - use absolute value to ensure positive
        # A is typically negative (darkening) and B is negative (decay rate)
        horizon_term = 1 + np.abs(A) * np.exp(B / cos_theta)
        horizon_term = np.maximum(horizon_term, 0.0)  # Ensure non-negative
        
        # Circumsolar and horizon gradient term
        # C controls circumsolar intensity, D controls falloff, E controls backscatter
        # For Hosek-Wilkie, D should create a sharp peak around sun
        # Use negative D interpretation: exp(-|D|*gamma) for falloff from sun
        circumsolar = np.abs(C) * np.exp(-np.abs(D) * gamma * 0.5)
        backscatter = np.abs(E) * cos_gamma * cos_gamma
        angular_term = 1 + circumsolar + backscatter
        
        # Combine terms
        radiance = horizon_term * angular_term
        
        # Additional zenith luminance correction using G, H parameters
        zenith_lum = 1.0 + G * np.exp(H * (theta - np.pi/2))
        zenith_lum = np.clip(zenith_lum, 0.1, 10.0)
        
        radiance = radiance * zenith_lum
        
        return np.maximum(radiance, 0.0)
    
    def _compute_radiance(
        self,
        state: HosekWilkieState,
        theta: np.ndarray,
        phi: np.ndarray,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Compute sky radiance for given viewing directions.
        
        Args:
            state: Pre-computed sky state
            theta: [N] Zenith angles of viewing directions (0 = up, pi/2 = horizon)
            phi: [N] Azimuth angles of viewing directions
            intensity: Overall intensity multiplier
            
        Returns:
            [N, 3] RGB radiance values
        """
        # Compute angle between view direction and sun
        cos_gamma = (
            np.sin(theta) * np.sin(state.sun_theta) * np.cos(phi - state.sun_phi) +
            np.cos(theta) * np.cos(state.sun_theta)
        )
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.arccos(cos_gamma)
        
        # Evaluate Perez function for each channel
        r = self._perez(state.configs_r, theta, gamma, cos_gamma) * state.radiance_r
        g = self._perez(state.configs_g, theta, gamma, cos_gamma) * state.radiance_g
        b = self._perez(state.configs_b, theta, gamma, cos_gamma) * state.radiance_b
        
        # Stack into RGB
        rgb = np.stack([r, g, b], axis=-1)
        
        # Clamp negative values
        rgb = np.maximum(rgb, 0.0)
        
        # Physical sky radiance scaling - typical sky is 1-50 cd/m², sun is ~1e9 cd/m²
        # We scale to match typical HDR environment map ranges (0-100+)
        SKY_RADIANCE_SCALE = 15.0  # Base radiance scale for clear sky zenith
        
        # Adjust scale based on sun elevation (lower sun = dimmer sky)
        sun_elevation = np.pi/2 - state.sun_theta
        elevation_factor = 0.3 + 0.7 * np.clip(np.sin(sun_elevation), 0, 1)
        
        rgb = rgb * SKY_RADIANCE_SCALE * elevation_factor * intensity
        
        # Add sun disk contribution for near-sun directions
        SUN_ANGULAR_RADIUS = 0.0093  # ~0.53 degrees (sun's angular radius in radians)
        SUN_INTENSITY = 300.0  # Relative sun intensity
        
        # Smooth sun disk falloff
        sun_mask = gamma < (SUN_ANGULAR_RADIUS * 3)  # Extended corona
        if sun_mask.any():
            # Sun disk (hard center)
            disk_mask = gamma < SUN_ANGULAR_RADIUS
            # Corona falloff
            corona_factor = np.exp(-((gamma / SUN_ANGULAR_RADIUS) ** 2) * 0.5)
            
            # Sun color varies with elevation (redder at sunset)
            sun_color = np.array([1.0, 0.95, 0.9])  # Slightly warm white
            if sun_elevation < 0.1:  # Sunset reddening
                red_factor = 1.0 - sun_elevation / 0.1
                sun_color = np.array([1.0, 0.6 + 0.35 * (1 - red_factor), 0.3 + 0.6 * (1 - red_factor)])
            
            sun_contribution = SUN_INTENSITY * corona_factor[:, np.newaxis] * sun_color * elevation_factor
            rgb[sun_mask] = rgb[sun_mask] + sun_contribution[sun_mask]
        
        return rgb
    
    def generate(
        self,
        sun_theta: float,
        sun_phi: float = 0.0,
        turbidity: float = 3.0,
        albedo: float = 0.1,
        resolution: Tuple[int, int] = (64, 128),
        intensity: float = 1.0,
        return_torch: bool = True,
        sky_only: bool = False,
    ) -> torch.Tensor:
        """
        Generate HDR equirectangular environment map.
        
        Args:
            sun_theta: Sun zenith angle in radians (0 = overhead, pi/2 = horizon)
            sun_phi: Sun azimuth angle in radians (0 = front)
            turbidity: Atmospheric turbidity (1 = clear, 10 = very hazy)
            albedo: Ground albedo (0 = dark, 1 = snow)
            resolution: (height, width) of output envmap
            intensity: Overall intensity multiplier (use ~0.1 to match normalized datasets)
            return_torch: If True, return torch.Tensor, else numpy.ndarray
            sky_only: If True, generate only sky hemisphere (theta 0 to π/2), not full sphere
            
        Returns:
            HDR environment map of shape [H, W, 3]
        """
        H, W = resolution
        
        # Create coordinate grids for equirectangular projection
        # u: [0, 1] horizontal, v: [0, 1] vertical
        u = np.linspace(0.5/W, 1 - 0.5/W, W)
        v = np.linspace(0.5/H, 1 - 0.5/H, H)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Convert to spherical angles
        # phi: azimuth [-pi, pi], theta: zenith [0, pi] or [0, pi/2] for sky_only
        phi = (u_grid - 0.5) * 2 * np.pi  # [-pi, pi]
        if sky_only:
            theta = v_grid * np.pi / 2  # [0, pi/2] for sky hemisphere only
        else:
            theta = v_grid * np.pi  # [0, pi] for full sphere
        
        # Flatten for computation
        theta_flat = theta.flatten()
        phi_flat = phi.flatten()
        
        # Create sky state
        state = self._create_state(sun_theta, sun_phi, turbidity, albedo)
        
        # Compute radiance for sky dome (theta < pi/2)
        # For below horizon, use ground color
        sky_mask = theta_flat < np.pi / 2
        
        radiance = np.zeros((H * W, 3), dtype=np.float32)
        
        if sky_mask.any():
            radiance[sky_mask] = self._compute_radiance(
                state, theta_flat[sky_mask], phi_flat[sky_mask], intensity
            )
        
        # Below horizon: simple ground color based on albedo
        if (~sky_mask).any():
            ground_color = np.array([0.1, 0.08, 0.05]) * albedo  # earthy brown
            radiance[~sky_mask] = ground_color
        
        # Reshape to image
        envmap = radiance.reshape(H, W, 3)
        
        if return_torch:
            return torch.from_numpy(envmap).float()
        return envmap
    
    def generate_for_sun_elevation(
        self,
        sun_elevation_deg: float,
        sun_azimuth_deg: float = 180.0,
        turbidity: float = 3.0,
        albedo: float = 0.1,
        resolution: Tuple[int, int] = (64, 128),
        intensity: float = 1.0,
        return_torch: bool = True,
        sky_only: bool = False,
    ) -> torch.Tensor:
        """
        Convenience method using degrees for sun position.
        
        Args:
            sun_elevation_deg: Sun elevation from horizon in degrees (0 = horizon, 90 = overhead)
            sun_azimuth_deg: Sun azimuth in degrees (0 = North, 90 = East, 180 = South)
            turbidity: Atmospheric turbidity (1-10)
            albedo: Ground albedo (0-1)
            resolution: (height, width)
            intensity: Overall intensity multiplier
            return_torch: Return as torch.Tensor
            sky_only: If True, generate only sky hemisphere (theta 0 to π/2)
        """
        sun_theta = np.deg2rad(90 - sun_elevation_deg)  # Convert to zenith angle
        sun_phi = np.deg2rad(sun_azimuth_deg - 180)  # Convert to our azimuth convention
        
        return self.generate(
            sun_theta=sun_theta,
            sun_phi=sun_phi,
            turbidity=turbidity,
            albedo=albedo,
            resolution=resolution,
            intensity=intensity,
            return_torch=return_torch,
            sky_only=sky_only,
        )


def demo():
    """Demo function to visualize Hosek-Wilkie sky model."""
    import matplotlib.pyplot as plt
    
    model = HosekWilkieSkyModel()
    
    # Generate skies at different sun elevations
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    sun_elevations = [5, 15, 30, 45, 60, 90]
    
    for ax, elev in zip(axes.flatten(), sun_elevations):
        envmap = model.generate_for_sun_elevation(
            sun_elevation_deg=elev,
            turbidity=3.0,
            resolution=(64, 128),
            return_torch=False,
        )
        
        # Simple tonemapping for display
        envmap_display = np.clip(envmap ** (1/2.2), 0, 1)
        
        ax.imshow(envmap_display)
        ax.set_title(f'Sun elevation: {elev}°')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('hosek_wilkie_demo.png', dpi=150)
    print("Saved hosek_wilkie_demo.png")


if __name__ == "__main__":
    demo()
