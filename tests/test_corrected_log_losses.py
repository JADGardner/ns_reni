"""Loss-correction pilot: invariance, weighting, and gradient regression tests."""
import pytest
import torch

from reni.model_components.losses import (
    ScaleInvariantLogLoss, per_image_exposure_loss, linear_rgb_cosine_from_logs,
)


def samples():
    target = torch.tensor([[.5, .2, .1], [.1, .3, .7], [.8, .4, .2], [.2, .1, .5]], dtype=torch.float64).log()
    return target, torch.tensor([10, 10, 27, 27])


def test_opposite_exposures():
    target, ids = samples()
    prediction = target + torch.tensor([1, 1, -1, -1])[:, None]
    assert ScaleInvariantLogLoss()(prediction, target).item() == pytest.approx(1)
    assert per_image_exposure_loss(prediction, target, ids).item() == pytest.approx(0, abs=1e-12)
    assert linear_rgb_cosine_from_logs(prediction, target).item() == pytest.approx(0, abs=1e-12)


def test_single_image_matches_legacy():
    target, ids = samples()
    prediction = target + torch.tensor([.1, -.2, .4])
    assert torch.allclose(per_image_exposure_loss(prediction, target, ids * 0), ScaleInvariantLogLoss()(prediction, target))


def test_colour_and_spatial_errors_retained():
    target, ids = samples()
    assert per_image_exposure_loss(target + torch.tensor([1, 0, -1]), target, ids) > .6
    assert per_image_exposure_loss(target + torch.tensor([1, -1, 0, 0])[:, None], target, ids) > .4
    assert linear_rgb_cosine_from_logs(target * 2, target) > .01


def test_weighted_ragged_shuffled_matches_explicit_groups():
    target, _ = samples()
    ids = torch.tensor([5, 8, 8, 8])
    weights = torch.tensor([1., 2., 0., 3.], dtype=torch.float64)
    prediction = target + torch.tensor([[1., 0., -1.], [2., 3., 4.], [4., 5., 6.], [1., 2., 3.]])
    expected = torch.stack([ScaleInvariantLogLoss()(prediction[ids == i], target[ids == i], weights[ids == i, None]) for i in [5, 8]]).mean()
    actual = per_image_exposure_loss(prediction, target, ids, weights)
    assert torch.allclose(actual, expected)
    order = torch.tensor([3, 0, 2, 1])
    assert torch.allclose(actual, per_image_exposure_loss(prediction[order], target[order], ids[order], weights[order]))


def test_zero_weight_groups():
    target, ids = samples()
    prediction = target + torch.tensor([1, -1, 2, 2])[:, None]
    assert per_image_exposure_loss(prediction, target, ids, torch.tensor([0, 0, 1, 1])).item() == pytest.approx(0)
    assert torch.isnan(per_image_exposure_loss(prediction, target, ids, torch.zeros(4)))


def test_white_and_large_exposure():
    target, _ = samples()
    assert linear_rgb_cosine_from_logs(target + 1000, target).item() == pytest.approx(0, abs=1e-12)
    assert linear_rgb_cosine_from_logs(target * 0, target * 0).item() == pytest.approx(0, abs=1e-12)


def test_gradients():
    target, ids = samples()
    prediction = (target + torch.tensor([.1, -.2, .4])).requires_grad_()
    assert torch.autograd.gradcheck(lambda p: per_image_exposure_loss(p, target, ids), (prediction,))
    assert torch.autograd.gradcheck(lambda p: linear_rgb_cosine_from_logs(p, target), (prediction,))


def test_exposure_gradient():
    target, ids = samples()
    ev = torch.tensor([1., -1.], dtype=torch.float64, requires_grad=True)
    prediction = target + ev.repeat_interleave(2)[:, None]
    loss = per_image_exposure_loss(prediction, target, ids) + linear_rgb_cosine_from_logs(prediction, target)
    assert torch.allclose(torch.autograd.grad(loss, ev)[0], torch.zeros_like(ev), atol=1e-12)
