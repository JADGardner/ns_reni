"""Invariant check for V8 counterfactual scene generation."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "sun_control"))
from synthetic_sky import erp_directions, render_sky, sample_params  # noqa: E402


def test_v8_nuisance_geometry_is_identical_across_suns():
    rng = np.random.default_rng(808)
    params = sample_params(rng)
    nuisance_seed = 123456
    directions = erp_directions(24, 48)

    images = []
    scenes = []
    for elevation, azimuth in ((10.0, -120.0), (80.0, 75.0), (-7.0, 15.0)):
        command = dict(params)
        command["sun_elevation_deg"] = elevation
        command["sun_azimuth_deg"] = azimuth
        image, _, scene = render_sky(
            24, 48,
            dirs=directions,
            nuisance_rng=np.random.default_rng(nuisance_seed),
            return_scene=True,
            **command,
        )
        images.append(image)
        scenes.append(scene)

    for key in scenes[0]:
        assert torch.equal(scenes[0][key], scenes[1][key])
        assert torch.equal(scenes[0][key], scenes[2][key])
    assert not torch.equal(images[0], images[1])
    assert not torch.equal(images[0], images[2])


if __name__ == "__main__":
    test_v8_nuisance_geometry_is_identical_across_suns()
    print("V8 nuisance-geometry checks passed")
