# Copyright 2023 The University of York. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Peak-aware HDR metrics for equirectangular environment maps (THESIS_PLAN G7h).

All metrics operate on LINEAR HDR images of shape [H, W, 3] (equirectangular,
W = 2H) and are solid-angle (sin-theta) weighted so poles do not dominate.

To make models trained in different gauges comparable, both images are first
mapped to a common fixed gauge (GT 99th-percentile luminance = 1). For
scale-invariant models the prediction is additionally exposure-aligned to the
GT with a robust median luminance ratio before the shared gauge is applied.

Metrics:
* ``log_rmse_hdr``: RMSE of log(E + eps), sin-weighted over all pixels.
* ``lum_weighted_rmse_hdr``: RMSE of linear E weighted by GT luminance * sin.
* ``peak_intensity_rel_error``: relative error of the mean luminance over the
  top-0.1% brightest GT directions (same directional set for GT and pred).
* ``peak_angle_error_deg``: angle between the luminance-weighted centroid
  directions of the top-0.1% brightest pixels of GT and of pred.
* ``peak_argmax_angle_error_deg``: angle between the single brightest pixel
  directions of GT and pred.
"""

from __future__ import annotations

import math
from typing import Dict, Literal

import torch
from torch import Tensor

from reni.utils.tonemap import luminance


def _sin_theta_weights(height: int, width: int, device, dtype) -> Tensor:
    """Per-pixel solid-angle weights sin(theta) for an equirect grid [H, W]."""
    theta = (torch.arange(height, device=device, dtype=dtype) + 0.5) * math.pi / height
    return torch.sin(theta)[:, None].expand(height, width)


def _pixel_directions(height: int, width: int, device, dtype) -> Tensor:
    """Unit direction per equirect pixel [H, W, 3] (convention-consistent)."""
    theta = (torch.arange(height, device=device, dtype=dtype) + 0.5) * math.pi / height
    phi = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2.0 * math.pi / width
    sin_t = torch.sin(theta)[:, None]
    directions = torch.stack(
        [
            sin_t * torch.cos(phi)[None, :],
            sin_t * torch.sin(phi)[None, :],
            torch.cos(theta)[:, None].expand(height, width),
        ],
        dim=-1,
    )
    return directions


def _angle_deg(a: Tensor, b: Tensor) -> Tensor:
    a = a / a.norm().clamp_min(1e-12)
    b = b / b.norm().clamp_min(1e-12)
    # atan2 formulation is numerically stable for near-parallel vectors.
    return torch.rad2deg(torch.atan2(torch.linalg.cross(a, b).norm(), torch.dot(a, b)))


def _weighted_centroid_direction(directions: Tensor, weights: Tensor) -> Tensor:
    centroid = (directions * weights[:, None]).sum(dim=0)
    return centroid / centroid.norm().clamp_min(1e-12)


@torch.no_grad()
def compute_hdr_peak_metrics(
    pred_linear: Tensor,
    gt_linear: Tensor,
    alignment: Literal["none", "median_ratio"] = "none",
    gauge_percentile: float = 0.99,
    gauge_target: float = 1.0,
    peak_fraction: float = 0.001,
    eps: float = 1e-8,
) -> Dict[str, Tensor]:
    """Compute peak-aware HDR metrics on linear equirect images [H, W, 3].

    Args:
        pred_linear: predicted linear HDR image [H, W, 3].
        gt_linear: ground-truth linear HDR image [H, W, 3].
        alignment: "median_ratio" exposure-aligns pred to GT first (use for
            scale-invariant models); "none" trusts the model's absolute scale.
        gauge_percentile / gauge_target: shared fixed gauge applied to both
            images based on the GT (pct(lum_gt) -> gauge_target).
        peak_fraction: fraction of brightest pixels treated as the peak set.
    """
    assert pred_linear.shape == gt_linear.shape and pred_linear.shape[-1] == 3
    height, width, _ = gt_linear.shape
    device, dtype = gt_linear.device, gt_linear.dtype

    pred = torch.clamp(pred_linear, min=0.0)
    gt = torch.clamp(gt_linear, min=0.0)

    lum_pred = luminance(pred)
    lum_gt = luminance(gt)

    if alignment == "median_ratio":
        ratio = lum_gt.flatten() / torch.clamp(lum_pred.flatten(), min=eps)
        scale = torch.median(ratio)
        pred = pred * scale
        lum_pred = lum_pred * scale

    # Common fixed gauge from the GT.
    gauge = gauge_target / torch.clamp(torch.quantile(lum_gt.flatten(), gauge_percentile), min=eps)
    pred = pred * gauge
    gt = gt * gauge
    lum_pred = lum_pred * gauge
    lum_gt = lum_gt * gauge

    sin_w = _sin_theta_weights(height, width, device, dtype)
    sin_w_sum = sin_w.sum()

    # Log-HDR RMSE (sin-weighted over pixels, mean over channels).
    log_sq = ((torch.log(pred + eps) - torch.log(gt + eps)) ** 2).mean(dim=-1)
    log_rmse = torch.sqrt((log_sq * sin_w).sum() / sin_w_sum)

    # Luminance-weighted HDR RMSE.
    lum_w = lum_gt * sin_w
    lin_sq = ((pred - gt) ** 2).mean(dim=-1)
    lum_weighted_rmse = torch.sqrt((lin_sq * lum_w).sum() / lum_w.sum().clamp_min(eps))

    # Peak set: top-`peak_fraction` pixels by GT luminance.
    num_pixels = height * width
    k = max(1, int(round(peak_fraction * num_pixels)))
    flat_lum_gt = lum_gt.flatten()
    flat_lum_pred = lum_pred.flatten()
    flat_sin = sin_w.flatten()
    top_gt = torch.topk(flat_lum_gt, k).indices

    peak_mean_gt = (flat_lum_gt[top_gt] * flat_sin[top_gt]).sum() / flat_sin[top_gt].sum()
    peak_mean_pred = (flat_lum_pred[top_gt] * flat_sin[top_gt]).sum() / flat_sin[top_gt].sum()
    peak_rel_error = torch.abs(peak_mean_pred - peak_mean_gt) / peak_mean_gt.clamp_min(eps)

    # Peak direction errors.
    directions = _pixel_directions(height, width, device, dtype).reshape(-1, 3)
    top_pred = torch.topk(flat_lum_pred, k).indices
    centroid_gt = _weighted_centroid_direction(directions[top_gt], flat_lum_gt[top_gt])
    centroid_pred = _weighted_centroid_direction(directions[top_pred], flat_lum_pred[top_pred])
    peak_angle = _angle_deg(centroid_pred, centroid_gt)

    argmax_gt_dir = directions[torch.argmax(flat_lum_gt)]
    argmax_pred_dir = directions[torch.argmax(flat_lum_pred)]
    argmax_angle = _angle_deg(argmax_pred_dir, argmax_gt_dir)

    return {
        "log_rmse_hdr": log_rmse,
        "lum_weighted_rmse_hdr": lum_weighted_rmse,
        "peak_intensity_rel_error": peak_rel_error,
        "peak_angle_error_deg": peak_angle,
        "peak_argmax_angle_error_deg": argmax_angle,
    }
