import torch
from typing import Optional

def linear_to_sRGB(color, use_quantile=False, q: Optional[torch.Tensor] = None, clamp=True):
    """Convert linear RGB to sRGB.
    
    Args:
        color: [..., 3]
        use_quantile: Whether to use the 98th quantile to normalise the color values.
        
        Returns:
            color: [..., 3]
    """
    if use_quantile or q is not None:
        if q is None:
            q = torch.quantile(color.flatten(), 0.98)
        color = color / q.expand_as(color)

    color = torch.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * torch.pow(torch.abs(color), 1 / 2.4) - 0.055,
    )
    if clamp:
        color = torch.clamp(color, 0.0, 1.0)
    return color
