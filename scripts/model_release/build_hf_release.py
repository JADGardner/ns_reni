#!/usr/bin/env python3
"""Build the allowlisted Hugging Face release for RENI model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RELEASE_VERSION = "1.1"


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    groups: tuple[str, ...]
    source_run: str
    destination: str
    checkpoint: str = "step-000100001.ckpt"
    released_checkpoint: str | None = None
    notes: str = ""


RUNS = (
    RunSpec(
        "thesis-vnjoint-ortho-so2-d100",
        ("core", "thesis"),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho/reni/2026-07-15_131259",
        "thesis/vnjoint-ortho/so2/d100",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so2-d9",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d9_two_bracket_ldrw3_2cyc_vnjoint_ortho/reni/2026-07-16_111806",
        "thesis/vnjoint-ortho/so2/d9",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so2-d36",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d36_two_bracket_ldrw3_2cyc_vnjoint_ortho/reni/2026-07-16_114245",
        "thesis/vnjoint-ortho/so2/d36",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so2-d49",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d49_two_bracket_ldrw3_2cyc_vnjoint_ortho/reni/2026-07-16_120704",
        "thesis/vnjoint-ortho/so2/d49",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so3-d9",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d9_two_bracket_ldrw3_2cyc_vnjoint_ortho_eqso3/reni/2026-07-16_162052",
        "thesis/vnjoint-ortho/so3/d9",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so3-d36",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d36_two_bracket_ldrw3_2cyc_vnjoint_ortho_eqso3/reni/2026-07-16_164535",
        "thesis/vnjoint-ortho/so3/d36",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so3-d49",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d49_two_bracket_ldrw3_2cyc_vnjoint_ortho_eqso3/reni/2026-07-16_170955",
        "thesis/vnjoint-ortho/so3/d49",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-so3-d100",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho_eqso3/reni/2026-07-16_130119",
        "thesis/vnjoint-ortho/so3/d100",
    ),
    RunSpec(
        "thesis-none-d9",
        ("thesis",),
        "outputs/reni/reni_equivariance_none_two_bracket_2cyc_d9/reni/2026-07-14_two_bracket_2cycles",
        "thesis/equivariance-none/d9",
    ),
    RunSpec(
        "thesis-none-d36",
        ("thesis",),
        "outputs/reni/reni_equivariance_none_two_bracket_2cyc_d36/reni/2026-07-14_two_bracket_2cycles",
        "thesis/equivariance-none/d36",
    ),
    RunSpec(
        "thesis-none-d49",
        ("thesis",),
        "outputs/reni/reni_equivariance_none_two_bracket_2cyc_d49/reni/2026-07-14_two_bracket_2cycles",
        "thesis/equivariance-none/d49",
    ),
    RunSpec(
        "thesis-none-d100",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho_eqnone/reni/2026-07-16_123144",
        "thesis/equivariance-none/d100",
    ),
    RunSpec(
        "neusky-prior",
        ("neusky-prior",),
        "outputs/reni/reni_latent_reset_d100_two_bracket_ldrw3_2cyc/reni/2026-07-04_2cycles",
        "neusky-prior",
        checkpoint="step-000100000.ckpt",
        released_checkpoint="step-000050000.ckpt",
        notes=(
            "The compatibility filename preserves the step recorded in released "
            "NeuSky configs; the source is the final 100000-step checkpoint."
        ),
    ),
    RunSpec(
        "thesis-channelwise-vn-seed42",
        ("thesis",),
        "outputs/reni/reni_latent_reset_d100_two_bracket_ldrw3_2cyc/reni/2026-07-04_2cycles",
        "thesis/invariant/channelwise-vn/seed42",
    ),
    RunSpec(
        "thesis-channelwise-vn-seed1234",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vn_seed1234/reni/2026-07-16_151057",
        "thesis/invariant/channelwise-vn/seed1234",
    ),
    RunSpec(
        "thesis-channelwise-vn-seed5678",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vn_seed5678/reni/2026-07-16_155035",
        "thesis/invariant/channelwise-vn/seed5678",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-seed1234",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho_seed1234/reni/2026-07-16_134520",
        "thesis/invariant/vnjoint-ortho/seed1234",
    ),
    RunSpec(
        "thesis-vnjoint-ortho-seed5678",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho_seed5678/reni/2026-07-16_142933",
        "thesis/invariant/vnjoint-ortho/seed5678",
    ),
    RunSpec(
        "thesis-invariant-norms",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_norms/reni/2026-07-16_094218",
        "thesis/invariant/norms",
    ),
    RunSpec(
        "thesis-invariant-vncanonical",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vncanonical/reni/2026-07-15_104134",
        "thesis/invariant/vncanonical",
    ),
    RunSpec(
        "thesis-invariant-vncanonical-ortho",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vncanonical_ortho/reni/2026-07-15_122650",
        "thesis/invariant/vncanonical-ortho",
    ),
    RunSpec(
        "thesis-invariant-vnjoint-collapsed",
        ("thesis",),
        "outputs/reni/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint/reni/2026-07-13_2cycles_vnjoint",
        "thesis/invariant/vnjoint-collapsed",
    ),
    RunSpec(
        "thesis-log-domain-four-cycle",
        ("thesis",),
        "code/ns_reni/outputs/reni_latent_reset_4_rerun/reni/2026-07-01_4cycles_rerun",
        "thesis/latent-reset/log-domain-four-cycle",
        checkpoint="step-000200003.ckpt",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def stage_run(
    phd_root: Path,
    output: Path,
    spec: RunSpec,
) -> dict[str, object]:
    source_run = phd_root / spec.source_run
    destination = output / spec.destination
    files = []
    for name in ("config.yml", "dataparser_transforms.json"):
        source = source_run / name
        if source.is_file():
            files.append(copy_file(source, destination / name))
    checkpoint_name = spec.released_checkpoint or spec.checkpoint
    files.append(
        copy_file(
            source_run / "nerfstudio_models" / spec.checkpoint,
            destination / "nerfstudio_models" / checkpoint_name,
        )
    )
    for info in files:
        info["path"] = Path(info["path"]).relative_to(output).as_posix()
    return {
        "id": spec.model_id,
        "groups": list(spec.groups),
        "source_run": spec.source_run,
        "source_checkpoint": spec.checkpoint,
        "notes": spec.notes,
        "files": files,
    }


def stage_published(
    model_storage: Path,
    output: Path,
) -> list[dict[str, object]]:
    source_root = model_storage / "reni_paper_models"
    models = []
    for config in sorted(source_root.rglob("config.yml")):
        run_dir = config.parent
        checkpoints = sorted((run_dir / "nerfstudio_models").glob("step-*.ckpt"))
        if not checkpoints:
            continue
        checkpoint = checkpoints[-1]
        relative = run_dir.relative_to(source_root)
        destination = output / "published" / relative
        files = [
            copy_file(config, destination / "config.yml"),
            copy_file(
                checkpoint,
                destination / "nerfstudio_models" / checkpoint.name,
            ),
        ]
        for info in files:
            info["path"] = Path(info["path"]).relative_to(output).as_posix()
        models.append(
            {
                "id": "published-" + "-".join(relative.parts).replace("_", "-"),
                "groups": ["published"],
                "source_run": str(run_dir),
                "source_checkpoint": checkpoint.name,
                "notes": "Final checkpoint from the archived RENI++ paper run.",
                "files": files,
            }
        )
    if not models:
        raise RuntimeError(f"no published model runs found under {source_root}")
    return models


def stage_minimal_decoder(
    phd_root: Path,
    output: Path,
) -> dict[str, object]:
    """Export and stage the PyTorch-only form of the headline decoder."""
    spec = RUNS[0]
    source_run = phd_root / spec.source_run
    destination = output / "minimal"
    exporter = (
        phd_root
        / "code"
        / "ns_reni"
        / "scripts"
        / "export_minimal_decoder.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(exporter),
            "--checkpoint",
            str(source_run / "nerfstudio_models" / spec.checkpoint),
            "--config",
            str(source_run / "config.yml"),
            "--output-dir",
            str(destination),
            "--model-id",
            spec.model_id,
        ],
        check=True,
    )

    example_root = (
        phd_root / "code" / "ns_reni" / "examples" / "minimal_inference"
    )
    for name in (
        "README.md",
        "pyproject.toml",
        "render.py",
        "reni_decoder.py",
        "uv.lock",
    ):
        copy_file(example_root / name, destination / name)
    copy_file(
        phd_root / "code" / "ns_reni" / "LICENSE",
        destination / "LICENSE",
    )

    files = []
    for path in sorted(destination.iterdir()):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "id": "thesis-vnjoint-ortho-so2-d100-minimal",
        "groups": ["minimal"],
        "source_run": spec.source_run,
        "source_checkpoint": spec.checkpoint,
        "notes": (
            "Decoder-only PyTorch reference artifact for inference and "
            "latent fitting; excludes Nerfstudio, latent banks and optimiser "
            "state."
        ),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phd-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument(
        "--model-storage",
        type=Path,
        default=Path("~/model-storage").expanduser(),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phd_root = args.phd_root.expanduser().resolve()
    model_storage = args.model_storage.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing release: {output}")
    output.mkdir(parents=True)

    models = [stage_run(phd_root, output, spec) for spec in RUNS]
    models.extend(stage_published(model_storage, output))
    models.append(stage_minimal_decoder(phd_root, output))
    shutil.copy2(Path(__file__).with_name("MODEL_CARD.md"), output / "README.md")
    shutil.copy2(
        phd_root / "code" / "ns_reni" / "LICENSE",
        output / "LICENSE",
    )

    manifest = {
        "schema_version": 1,
        "release_version": RELEASE_VERSION,
        "repository": "jadgardner/reni-models",
        "models": models,
    }
    (output / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_lines = sorted(
        f"{file_info['sha256']}  {file_info['path']}"
        for model in models
        for file_info in model["files"]
    )
    (output / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Staged {len(models)} models at {output}")


if __name__ == "__main__":
    main()
