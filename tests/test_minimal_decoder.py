"""Unit tests for the PyTorch-only thesis RENI++ decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


MINIMAL_ROOT = Path(__file__).resolve().parents[1] / "examples" / "minimal_inference"
sys.path.insert(0, str(MINIMAL_ROOT))

from reni_decoder import (  # noqa: E402
    ARTIFACT_FORMAT_VERSION,
    MODEL_TYPE,
    ReniDecoder,
    equirectangular_directions,
)


def test_artifact_roundtrip_and_latent_gradient(tmp_path: Path) -> None:
    torch.manual_seed(3)
    source = ReniDecoder().eval()
    artifact = tmp_path / "decoder.pt"
    torch.save(
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "model_type": MODEL_TYPE,
            "config": source.config.to_dict(),
            "state_dict": source.state_dict(),
        },
        artifact,
    )
    restored = ReniDecoder.from_artifact(artifact)
    latent = torch.randn(100, 3)
    directions = torch.randn(13, 3)
    torch.testing.assert_close(
        source(latent, directions, chunk_size=5),
        restored(latent, directions, chunk_size=5),
    )

    trainable = ReniDecoder()
    trainable.load_state_dict(restored.state_dict())
    latent.requires_grad_(True)
    trainable(latent, directions).mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().max() > 0


def test_batched_inputs_broadcast() -> None:
    model = ReniDecoder().eval()
    latents = torch.randn(2, 100, 3)
    directions = torch.randn(7, 3)
    output = model(latents, directions, chunk_size=3)
    assert output.shape == (2, 7, 3)
    torch.testing.assert_close(
        output[0],
        model(latents[0], directions),
        atol=3e-4,
        rtol=1e-5,
    )


def test_equirectangular_directions_are_unit_length() -> None:
    directions = equirectangular_directions(8)
    assert directions.shape == (128, 3)
    torch.testing.assert_close(
        directions.norm(dim=-1),
        torch.ones(128),
        atol=1e-6,
        rtol=1e-6,
    )
