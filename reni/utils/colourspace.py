import torch

def linear_to_sRGB(color):
    q = torch.quantile(color.flatten(), 0.98)
    color = color / q.expand_as(color)
    color = torch.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * torch.pow(torch.abs(color), 1 / 2.4) - 0.055,
    )
    color = torch.clamp(color, 0.0, 1.0)
    return color