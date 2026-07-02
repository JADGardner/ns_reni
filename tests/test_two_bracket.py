"""Unit tests for the G7h two-bracket tonemapping utilities and weighted losses.

CPU-only; run from the repo root:

    PYTHONPATH=. python -m pytest tests/test_two_bracket.py -q
"""

import math

import pytest
import torch

from reni.model_components.losses import ScaleInvariantLogLoss, WeightedMSELoss
from reni.utils.tonemap import (
    apply_fixed_gauge,
    blend_reconstruct,
    blend_weight,
    encode_two_bracket,
    inverse_log_tonemap,
    inverse_reinhard_extended,
    log_tonemap,
    luminance,
    reinhard_extended,
    two_bracket_to_linear,
)

torch.manual_seed(0)


def _random_hdr(shape=(4096, 3), lo=1e-4, hi=9e3):
    """Log-uniform positive HDR samples spanning shadows to bright suns."""
    u = torch.rand(*shape, dtype=torch.float64)
    return torch.exp(u * (math.log(hi) - math.log(lo)) + math.log(lo))


def test_reinhard_bounds_and_inverse():
    e = _random_hdr()
    t = reinhard_extended(e, m_ldr=16.0)
    assert torch.allclose(
        reinhard_extended(torch.tensor(16.0, dtype=torch.float64)), torch.tensor(1.0, dtype=torch.float64)
    )
    e_sub = e[e <= 16.0]
    t_sub = reinhard_extended(e_sub, m_ldr=16.0)
    assert (t_sub >= 0).all() and (t_sub <= 1.0 + 1e-9).all()
    recon = inverse_reinhard_extended(t_sub, m_ldr=16.0)
    assert torch.allclose(recon, e_sub, rtol=1e-6, atol=1e-9)


def test_log_tonemap_bounds_and_inverse():
    e = _random_hdr()
    t = log_tonemap(e, m_log=10000.0)
    assert (t >= 0).all() and (t <= 1.0 + 1e-9).all()
    recon = inverse_log_tonemap(t, m_log=10000.0)
    assert torch.allclose(recon, e, rtol=1e-6, atol=1e-9)


def test_blend_weight_monotone_saturation():
    lo = blend_weight(torch.tensor([[0.1, 0.1, 0.1]]))
    mid = blend_weight(torch.tensor([[0.95, 0.2, 0.2]]))
    hi = blend_weight(torch.tensor([[16.0, 0.2, 0.2]]))
    assert lo.item() < 1e-3
    assert abs(mid.item() - 0.5) < 1e-6
    assert hi.item() > 1.0 - 1e-6


def test_blend_round_trip_float64():
    """blend(reconstruct(E)) ~= E across the full [0, M_log] range."""
    e = _random_hdr()
    encoded = encode_two_bracket(e, m_ldr=16.0, m_log=10000.0)
    assert encoded.shape[-1] == 6
    recon = two_bracket_to_linear(encoded, m_ldr=16.0, m_log=10000.0)
    assert torch.allclose(recon, e, rtol=1e-5, atol=1e-8), (
        f"max rel err {(recon - e).abs().div(e).max().item()}"
    )


def test_blend_round_trip_float32():
    e = _random_hdr().float()
    recon = two_bracket_to_linear(encode_two_bracket(e), )
    rel = (recon - e).abs() / e.clamp_min(1e-6)
    assert rel.max().item() < 5e-3, f"max rel err {rel.max().item()}"


def test_blend_reconstruct_differentiable():
    ldr = torch.rand(128, 3, requires_grad=True)
    logb = torch.rand(128, 3, requires_grad=True)
    out = blend_reconstruct(ldr, logb)
    out.sum().backward()
    assert ldr.grad is not None and torch.isfinite(ldr.grad).all()
    assert logb.grad is not None and torch.isfinite(logb.grad).all()


def test_values_above_m_log_clamp_gracefully():
    e = torch.tensor([[2e4, 1.0, 1.0]], dtype=torch.float64)
    recon = two_bracket_to_linear(encode_two_bracket(e))
    assert torch.isfinite(recon).all()
    assert abs(recon[0, 0].item() - 10000.0) / 10000.0 < 1e-6  # clamped at M_log


def test_fixed_gauge_property():
    e = _random_hdr((64, 128, 3)).float()
    gauged = apply_fixed_gauge(e, percentile=0.99, target=1.0)
    lum = luminance(gauged.reshape(-1, 3))
    p99 = torch.quantile(lum, 0.99)
    assert abs(p99.item() - 1.0) < 1e-4


def test_weighted_scale_inv_reduces_to_unweighted():
    loss_fn = ScaleInvariantLogLoss()
    pred = torch.randn(2048, 3)
    gt = torch.randn(2048, 3)
    base = loss_fn(pred, gt)
    uniform = loss_fn(pred, gt, weights=torch.ones_like(pred))
    assert torch.allclose(base, uniform, rtol=1e-5, atol=1e-7)


def test_weighted_scale_inv_still_shift_invariant():
    loss_fn = ScaleInvariantLogLoss()
    pred = torch.randn(2048, 3)
    gt = torch.randn(2048, 3)
    weights = torch.rand(2048, 1) + 0.1
    a = loss_fn(pred, gt, weights=weights)
    b = loss_fn(pred + 3.7, gt, weights=weights)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_weighted_mse_matches_manual():
    loss_fn = WeightedMSELoss()
    pred = torch.randn(512, 3)
    gt = torch.randn(512, 3)
    weights = torch.rand(512, 1)
    expected = (weights.expand_as(pred) * (pred - gt) ** 2).sum() / weights.expand_as(pred).sum()
    assert torch.allclose(loss_fn(pred, gt, weights), expected)


def test_hdr_peak_metrics_identity_and_scale():
    from reni.utils.hdr_metrics import compute_hdr_peak_metrics

    gt = _random_hdr((64, 128, 3)).float()
    metrics = compute_hdr_peak_metrics(gt.clone(), gt, alignment="none")
    assert metrics["log_rmse_hdr"].item() < 1e-5
    assert metrics["lum_weighted_rmse_hdr"].item() < 1e-4
    assert metrics["peak_intensity_rel_error"].item() < 1e-5
    assert metrics["peak_angle_error_deg"].item() < 1e-2
    assert metrics["peak_argmax_angle_error_deg"].item() < 1e-2

    # A global exposure error should be forgiven under median_ratio alignment.
    metrics_scaled = compute_hdr_peak_metrics(gt * 3.0, gt, alignment="median_ratio")
    assert metrics_scaled["log_rmse_hdr"].item() < 1e-4
    assert metrics_scaled["peak_intensity_rel_error"].item() < 1e-4
    # ... but penalised without alignment.
    metrics_unaligned = compute_hdr_peak_metrics(gt * 3.0, gt, alignment="none")
    assert metrics_unaligned["peak_intensity_rel_error"].item() > 1.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
