"""Focused tests for structurally shared synthetic sun latents."""

import json
import tempfile
from pathlib import Path

import torch

from reni.illumination_fields.reni_illumination_field import (
    RENIField,
    RENIFieldConfig,
    _load_structural_sun_layout,
)


def _labels(tmp_path):
    payload = {
        "g0_a.exr": {"group_id": 10, "sun_direction": [1.0, 0.0, 0.0]},
        "g0_b.exr": {"group_id": 10, "sun_direction": [0.0, 1.0, 0.0]},
        "g1_a.exr": {"group_id": 20, "sun_direction": [0.0, 0.0, 1.0]},
        "g1_b.exr": {"group_id": 20, "sun_direction": [-1.0, 0.0, 0.0]},
    }
    path = tmp_path / "sun_labels.json"
    path.write_text(json.dumps(payload))
    return path


def test_structural_layout_matches_dataparser_and_mirror_order():
    with tempfile.TemporaryDirectory() as directory:
        labels = _labels(Path(directory))
        representatives, directions, groups = _load_structural_sun_layout(
            labels, num_train_data=8)

        assert groups == 4
        assert representatives.tolist() == [0, 0, 2, 2, 4, 4, 6, 6]
        assert torch.equal(directions[4:, 0], -directions[:4, 0])
        assert torch.equal(directions[4:, 1:], directions[:4, 1:])
        assert torch.allclose(directions.norm(dim=-1), torch.ones(8))


def test_structural_sampling_shares_content_and_hard_sets_sun():
    with tempfile.TemporaryDirectory() as directory:
        config = RENIFieldConfig(
            latent_dim=4,
            structural_sun_labels=_labels(Path(directory)),
            structural_sun_channel=1,
            invariant_function="Norms",
            conditioning="Concat",
            encoded_input="None",
            hidden_layers=1,
            hidden_features=8,
        )
        field = RENIField(
            config, num_train_data=8, num_eval_data=1,
            normalisations=None)

        indices = torch.tensor([0, 1, 4, 5])
        _, mu, _ = field._sample_structural_train_latent(
            indices, stochastic=False)
        keep = torch.tensor([0, 2, 3])

        assert torch.equal(mu[0, keep], mu[1, keep])
        assert torch.equal(mu[2, keep], mu[3, keep])
        assert torch.equal(mu[:, 1], field.structural_sun_directions[indices])

        mu[:, keep].sum().backward()
        assert field.train_mu.grad is not None
        assert field.train_mu.grad[0, keep].abs().sum() > 0
        assert field.train_mu.grad[1].abs().sum() == 0
        assert field.train_mu.grad[:, 1].abs().sum() == 0


def test_reset_and_checkpoint_materialisation_preserve_contract():
    with tempfile.TemporaryDirectory() as directory:
        config = RENIFieldConfig(
            latent_dim=4,
            structural_sun_labels=_labels(Path(directory)),
            structural_sun_channel=1,
            invariant_function="Norms",
            conditioning="Concat",
            encoded_input="None",
            hidden_layers=1,
            hidden_features=8,
        )
        field = RENIField(
            config, num_train_data=8, num_eval_data=1,
            normalisations=None)
        field.reset_train_latents_to_zero()

        _, mu, _ = field._sample_structural_train_latent(
            torch.arange(8), stochastic=False)
        keep = torch.tensor([0, 2, 3])
        assert torch.count_nonzero(mu[:, keep]) == 0
        assert torch.equal(mu[:, 1], field.structural_sun_directions)

        saved = field.state_dict()
        bank = saved["train_mu"]
        assert torch.equal(bank[0, keep], bank[1, keep])
        assert torch.equal(bank[:, 1], field.structural_sun_directions)


if __name__ == "__main__":
    test_structural_layout_matches_dataparser_and_mirror_order()
    test_structural_sampling_shares_content_and_hard_sets_sun()
    test_reset_and_checkpoint_materialisation_preserve_contract()
    print("structural sun-group checks passed")
