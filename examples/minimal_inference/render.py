#!/usr/bin/env python3
"""Render an HDR environment map with the lightweight RENI++ decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from reni_decoder import ReniDecoder, equirectangular_directions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("decoder.pt"),
        help="Decoder-only artifact from the RENI Models release",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("render"))
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exposure", type=float, default=0.0, help="Display EV")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser.parse_args()


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    linear = linear.clamp_min(0.0)
    return torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )


def main() -> None:
    args = parse_args()
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model = ReniDecoder.from_artifact(args.weights, device=device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn(
        model.config.latent_dim,
        3,
        generator=generator,
    ).to(device)
    directions = equirectangular_directions(
        args.height,
        device=device,
    )
    with torch.no_grad():
        hdr = model(latent, directions, args.chunk_size).reshape(
            args.height, 2 * args.height, 3
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hdr.cpu(), args.output_dir / "environment_linear_hdr.pt")
    torch.save(latent.cpu(), args.output_dir / "latent.pt")

    display = hdr * (2.0**args.exposure)
    display = display / (1.0 + display.clamp_min(0.0))
    display = (
        linear_to_srgb(display).clamp(0.0, 1.0).mul(255.0).byte().cpu().contiguous()
    )
    preview = Image.frombytes(
        "RGB",
        (display.shape[1], display.shape[0]),
        bytes(display.flatten()),
    )
    preview.save(args.output_dir / "environment_preview.png")
    print(f"Wrote {args.output_dir / 'environment_linear_hdr.pt'}")
    print(f"Wrote {args.output_dir / 'environment_preview.png'}")


if __name__ == "__main__":
    main()
