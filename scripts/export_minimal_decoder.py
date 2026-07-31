#!/usr/bin/env python3
"""Export the thesis RENI++ decoder without Nerfstudio or optimiser state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_ROOT = PROJECT_ROOT / "examples" / "minimal_inference"
sys.path.insert(0, str(MINIMAL_ROOT))

from reni_decoder import (  # noqa: E402
    ARTIFACT_FORMAT_VERSION,
    MODEL_TYPE,
    ReniDecoder,
    ReniDecoderConfig,
)


class ConfigLoader(yaml.SafeLoader):
    """Load Nerfstudio object YAML as plain mappings without importing it."""


def _construct_python(
    loader: yaml.Loader,
    tag_suffix: str,
    node: yaml.Node,
) -> Any:
    if tag_suffix.startswith(("name:", "module:")):
        return tag_suffix.split(":", 1)[1]
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    try:
        return loader.construct_scalar(node)
    except Exception:
        return None


ConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    _construct_python,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=ConfigLoader)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a config mapping")
    return config


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def validate_and_extract_config(config: dict[str, Any]) -> ReniDecoderConfig:
    field = nested(config, "pipeline", "model", "field")
    dataparser = nested(config, "pipeline", "datamanager", "dataparser")
    expected = {
        "axis_of_invariance": "z",
        "canonical_frame_orthonormalise": True,
        "conditioning": "Attention",
        "encoded_input": "Directions",
        "equivariance": "SO2",
        "invariant_function": "VNJoint",
        "latent_dim": 100,
        "hidden_features": 128,
        "num_attention_heads": 8,
        "num_attention_layers": 6,
        "out_features": 6,
        "output_activation": "sigmoid",
        "positional_encoding": "NeRF",
    }
    mismatches = {
        name: (field.get(name), required)
        for name, required in expected.items()
        if field.get(name) != required
    }
    if dataparser.get("tonemap_targets") is not True:
        mismatches["tonemap_targets"] = (
            dataparser.get("tonemap_targets"),
            True,
        )
    if dataparser.get("fixed_gauge_normalisation") is not True:
        mismatches["fixed_gauge_normalisation"] = (
            dataparser.get("fixed_gauge_normalisation"),
            True,
        )
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {required!r})"
            for name, (actual, required) in sorted(mismatches.items())
        )
        raise ValueError(f"unsupported RENI architecture: {details}")
    return ReniDecoderConfig(
        latent_dim=field["latent_dim"],
        hidden_features=field["hidden_features"],
        num_attention_heads=field["num_attention_heads"],
        num_attention_layers=field["num_attention_layers"],
        out_features=field["out_features"],
        m_ldr=float(dataparser["tonemap_m_ldr"]),
        m_log=float(dataparser["tonemap_m_log"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default="thesis-vnjoint-ortho-so2-d100",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    decoder_config = validate_and_extract_config(load_config(config_path))
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    pipeline = checkpoint.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError(f"{checkpoint_path} has no pipeline state dictionary")

    prefixes = ("_model.field.network.", "_model.field.vn_joint_frame.")
    state_dict = {
        key.removeprefix("_model.field."): value.detach().cpu()
        for key, value in pipeline.items()
        if key.startswith(prefixes)
    }
    model = ReniDecoder(decoder_config)
    model.load_state_dict(state_dict, strict=True)

    artifact_path = output_dir / "decoder.pt"
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "config": decoder_config.to_dict(),
        "state_dict": state_dict,
        "source": {
            "model_id": args.model_id,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "step": checkpoint.get("step"),
        },
    }
    torch.save(payload, artifact_path)

    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "config": decoder_config.to_dict(),
        "source": payload["source"],
        "artifact": {
            "path": artifact_path.name,
            "size_bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        },
    }
    (output_dir / "decoder_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {artifact_path} ({artifact_path.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
