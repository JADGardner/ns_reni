#!/usr/bin/env python3
"""Compare the lightweight decoder with the original Nerfstudio field."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "minimal_inference"))

from reni_decoder import (  # noqa: E402
    ReniDecoder,
    equirectangular_directions,
    two_bracket_to_linear,
)
from reni.illumination_fields.reni_illumination_field import (  # noqa: E402
    RENIField,
    RENIFieldConfig,
)
from reni.model_components.illumination_samplers import (  # noqa: E402
    EquirectangularSamplerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=257)
    parser.add_argument("--erp-width", type=int, default=32)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def original_field(checkpoint_path: Path) -> RENIField:
    field = RENIField(
        RENIFieldConfig(
            equivariance="SO2",
            axis_of_invariance="z",
            invariant_function="VNJoint",
            canonical_frame_orthonormalise=True,
            conditioning="Attention",
            positional_encoding="NeRF",
            encoded_input="Directions",
            latent_dim=100,
            hidden_features=128,
            num_attention_heads=8,
            num_attention_layers=6,
            out_features=6,
            output_activation="sigmoid",
            last_layer_linear=True,
            fixed_decoder=True,
        )
    ).eval()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source = checkpoint["pipeline"]
    state_dict = {
        key.removeprefix("_model.field."): value
        for key, value in source.items()
        if key.startswith(("_model.field.network.", "_model.field.vn_joint_frame."))
    }
    missing, unexpected = field.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "min_max",
        "log_domain",
        "log_domain_min",
        "log_domain_max",
    }
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(
            f"could not restore reference field: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return field


def main() -> None:
    args = parse_args()
    torch.manual_seed(17)
    reference = original_field(args.checkpoint.expanduser().resolve())
    minimal = ReniDecoder.from_artifact(
        args.decoder.expanduser().resolve(),
    )
    latent = torch.randn(args.samples, 100, 3)
    directions = torch.randn(args.samples, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        directional, conditioning = reference.vn_joint_invariance(
            latent,
            directions,
            equivariance="SO2",
            axis_of_invariance=2,
        )
        directional, conditioning = reference.apply_positional_encoding(
            directional,
            conditioning,
        )
        reference_brackets = reference.network(directional, conditioning)
        minimal_brackets = minimal.decode_brackets(
            latent,
            directions.unsqueeze(1),
        ).squeeze(1)
        reference_hdr = two_bracket_to_linear(reference_brackets)
        minimal_hdr = minimal(latent, directions.unsqueeze(1)).squeeze(1)

    bracket_error = (reference_brackets - minimal_brackets).abs().max().item()
    hdr_error = (reference_hdr - minimal_hdr).abs().max().item()
    torch.testing.assert_close(
        minimal_brackets,
        reference_brackets,
        atol=args.atol,
        rtol=args.rtol,
    )
    torch.testing.assert_close(
        minimal_hdr,
        reference_hdr,
        atol=args.atol,
        rtol=args.rtol,
    )

    reference_erp = (
        EquirectangularSamplerConfig(width=args.erp_width)
        .setup()
        .generate_direction_samples()
        .frustums.directions
    )
    minimal_erp = equirectangular_directions(args.erp_width // 2)
    erp_error = (reference_erp - minimal_erp).abs().max().item()
    torch.testing.assert_close(
        minimal_erp,
        reference_erp,
        atol=1e-6,
        rtol=1e-6,
    )

    differentiable = ReniDecoder(minimal.config)
    differentiable.load_state_dict(minimal.state_dict())
    fitted_latent = torch.randn(100, 3, requires_grad=True)
    loss = differentiable(
        fitted_latent,
        directions[:8],
        chunk_size=4,
    ).mean()
    loss.backward()
    if fitted_latent.grad is None or not torch.isfinite(fitted_latent.grad).all():
        raise RuntimeError("latent-code gradients are missing or non-finite")
    if fitted_latent.grad.abs().max().item() == 0.0:
        raise RuntimeError("latent-code gradients are identically zero")

    print(f"Bracket max absolute error: {bracket_error:.3e}")
    print(f"HDR max absolute error:     {hdr_error:.3e}")
    print(f"ERP direction max error:    {erp_error:.3e}")
    print("Latent-code gradient check: passed")


if __name__ == "__main__":
    main()
