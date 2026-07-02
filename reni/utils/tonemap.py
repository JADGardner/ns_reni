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

"""Two-bracket (complementary tonemapping) utilities for RENI++.

The two-bracket parameterisation represents a linear HDR radiance value E with
two bounded targets in [0, 1]:

* an extended-Reinhard "LDR bracket" that owns the low/mid range,
      T_ldr(E) = E * (1 + E / M_ldr^2) / (1 + E),        T_ldr(M_ldr) = 1
* a log "highlight bracket" that owns the bright range,
      T_log(E) = log(1 + E) / log(1 + M_log),            T_log(M_log) = 1

Both brackets are inverted in closed form and blended with a differentiable
sigmoid saturation weight computed from the LINEARISED LDR bracket:

      E_ldr = T_ldr^{-1}(ldr_bracket)
      w     = sigmoid((max_c E_ldr - tau) / delta)
      E_hat = (1 - w) * E_ldr + w * E_log

so the reconstruction uses the Reinhard bracket for the low/mid range (below
tau, in fixed-gauge units where the 99th-percentile luminance is 1) and hands
over to the log bracket in the highlights. Since both inverses are exact below
their saturation points, the blend is an exact reconstruction for E in
[0, M_log] regardless of the transition sharpness. All functions are pure
torch and differentiable.

Defaults follow THESIS_PLAN G7h: M_ldr=16, M_log=10000, tau=0.95, delta=0.02.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

DEFAULT_M_LDR = 16.0
DEFAULT_M_LOG = 10000.0
DEFAULT_BLEND_TAU = 0.95
DEFAULT_BLEND_DELTA = 0.02

# BT.709 luminance coefficients.
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def luminance(rgb: Tensor) -> Tensor:
    """BT.709 luminance of a linear RGB tensor [..., 3] -> [...]."""
    r, g, b = LUMINANCE_WEIGHTS
    return r * rgb[..., 0] + g * rgb[..., 1] + b * rgb[..., 2]


def reinhard_extended(linear: Tensor, m_ldr: float = DEFAULT_M_LDR) -> Tensor:
    """Extended Reinhard tonemap; maps [0, m_ldr] to [0, 1] with T(m_ldr) = 1."""
    return linear * (1.0 + linear / (m_ldr**2)) / (1.0 + linear)


def inverse_reinhard_extended(tonemapped: Tensor, m_ldr: float = DEFAULT_M_LDR) -> Tensor:
    """Closed-form inverse of :func:`reinhard_extended` (positive quadratic root).

    Solves E^2 / M^2 + (1 - t) E - t = 0 for E >= 0, using the
    cancellation-free conjugate form of the root (stable for small t):
        E = 2 t / ((1 - t) + sqrt((1 - t)^2 + 4 t / M^2))
    """
    t = tonemapped
    disc = (1.0 - t) ** 2 + 4.0 * t / (m_ldr**2)
    return 2.0 * t / ((1.0 - t) + torch.sqrt(torch.clamp(disc, min=0.0)))


def log_tonemap(linear: Tensor, m_log: float = DEFAULT_M_LOG) -> Tensor:
    """Log tonemap log(1 + E) / log(1 + m_log); maps [0, m_log] to [0, 1]."""
    return torch.log1p(torch.clamp(linear, min=0.0)) / math.log1p(m_log)


def inverse_log_tonemap(tonemapped: Tensor, m_log: float = DEFAULT_M_LOG) -> Tensor:
    """Closed-form inverse of :func:`log_tonemap`: E = (1 + m_log)^t - 1."""
    return torch.expm1(tonemapped * math.log1p(m_log))


def encode_two_bracket(
    linear: Tensor,
    m_ldr: float = DEFAULT_M_LDR,
    m_log: float = DEFAULT_M_LOG,
) -> Tensor:
    """Encode linear HDR RGB [..., 3] into two-bracket targets [..., 6].

    Channels 0:3 hold the extended-Reinhard LDR bracket, channels 3:6 the log
    bracket; both clamped to [0, 1] (values above m_ldr saturate the LDR
    bracket, values above m_log saturate the log bracket).
    """
    ldr = torch.clamp(reinhard_extended(linear, m_ldr), 0.0, 1.0)
    logb = torch.clamp(log_tonemap(linear, m_log), 0.0, 1.0)
    return torch.cat([ldr, logb], dim=-1)


def blend_weight(
    ldr_linearised: Tensor,
    tau: float = DEFAULT_BLEND_TAU,
    delta: float = DEFAULT_BLEND_DELTA,
) -> Tensor:
    """Sigmoid saturation blend weight from the linearised LDR bracket.

    Args:
        ldr_linearised: [..., 3] inverse-Reinhard (linear) values E_ldr.

    Returns:
        [..., 1] weight w = sigmoid((max_c E_ldr - tau) / delta); w -> 1 where
        the LDR bracket is bright/saturated and the reconstruction should
        trust the log bracket instead.
    """
    max_c = ldr_linearised.max(dim=-1, keepdim=True).values
    return torch.sigmoid((max_c - tau) / delta)


def blend_reconstruct(
    ldr_bracket: Tensor,
    log_bracket: Tensor,
    m_ldr: float = DEFAULT_M_LDR,
    m_log: float = DEFAULT_M_LOG,
    tau: float = DEFAULT_BLEND_TAU,
    delta: float = DEFAULT_BLEND_DELTA,
) -> Tensor:
    """Differentiable linear-HDR reconstruction from the two brackets.

    Args:
        ldr_bracket: [..., 3] extended-Reinhard bracket in [0, 1].
        log_bracket: [..., 3] log bracket in [0, 1].

    Returns:
        [..., 3] linear HDR estimate E_hat = (1 - w) E_ldr + w E_log with
        w = sigmoid((max_c E_ldr - tau) / delta).
    """
    e_ldr = inverse_reinhard_extended(ldr_bracket, m_ldr)
    e_log = inverse_log_tonemap(log_bracket, m_log)
    w = blend_weight(e_ldr, tau, delta)
    return (1.0 - w) * e_ldr + w * e_log


def split_brackets(two_bracket: Tensor) -> Tuple[Tensor, Tensor]:
    """Split a [..., 6] two-bracket tensor into (ldr_bracket, log_bracket)."""
    return two_bracket[..., :3], two_bracket[..., 3:6]


def two_bracket_to_linear(
    two_bracket: Tensor,
    m_ldr: float = DEFAULT_M_LDR,
    m_log: float = DEFAULT_M_LOG,
    tau: float = DEFAULT_BLEND_TAU,
    delta: float = DEFAULT_BLEND_DELTA,
) -> Tensor:
    """Reconstruct linear HDR RGB [..., 3] from a [..., 6] two-bracket tensor."""
    ldr, logb = split_brackets(two_bracket)
    return blend_reconstruct(ldr, logb, m_ldr, m_log, tau, delta)


def fixed_gauge_scale(
    linear: Tensor,
    percentile: float = 0.99,
    target: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Scalar gauge factor s such that pct_{percentile}(luminance(s * E)) = target."""
    lum = luminance(linear.reshape(-1, 3))
    pct = torch.quantile(lum, percentile)
    return target / torch.clamp(pct, min=eps)


def apply_fixed_gauge(
    linear: Tensor,
    percentile: float = 0.99,
    target: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Normalise a linear HDR image [..., 3] to the fixed luminance gauge."""
    return linear * fixed_gauge_scale(linear, percentile, target, eps)
