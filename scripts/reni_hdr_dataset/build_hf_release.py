#!/usr/bin/env python3
"""Build the allowlisted Hugging Face release of the RENI HDR dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

SPLIT_COUNTS = {"train": 1673, "val": 10, "test": 21}
MASK_COUNT = 5
EXR_MAGIC = struct.pack("<I", 20000630)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_c_string(handle) -> str:
    value = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise ValueError("unexpected end of EXR header")
        if byte == b"\0":
            return value.decode("ascii")
        value.extend(byte)


def exr_metadata(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        if handle.read(4) != EXR_MAGIC:
            raise ValueError(f"not a valid OpenEXR file: {path}")
        version = struct.unpack("<I", handle.read(4))[0]
        attributes: dict[str, tuple[str, bytes]] = {}
        while True:
            name = read_c_string(handle)
            if not name:
                break
            attribute_type = read_c_string(handle)
            size = struct.unpack("<I", handle.read(4))[0]
            attributes[name] = (attribute_type, handle.read(size))

    if "dataWindow" not in attributes:
        raise ValueError(f"EXR has no dataWindow: {path}")
    _, data_window = attributes["dataWindow"]
    if len(data_window) != 16:
        raise ValueError(f"invalid EXR dataWindow: {path}")
    x_min, y_min, x_max, y_max = struct.unpack("<4i", data_window)
    return {
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
        "version": version,
    }


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def expected_names(count: int, suffix: str) -> list[str]:
    return [f"{index:05d}{suffix}" for index in range(1, count + 1)]


def release_artifacts(data_root: Path) -> list[tuple[Path, Path]]:
    artifacts: list[tuple[Path, Path]] = []
    for split in SPLIT_COUNTS:
        for source in sorted((data_root / split).glob("*.exr")):
            artifacts.append(
                (source, Path("RENI_HDR") / split / source.name)
            )
    for source in sorted((data_root / "masks").glob("*.png")):
        artifacts.append(
            (source, Path("RENI_HDR") / "masks" / source.name)
        )
    return artifacts


def validate_dataset(data_root: Path) -> dict[str, object]:
    split_stats: dict[str, dict[str, object]] = {}
    for split, count in SPLIT_COUNTS.items():
        files = sorted((data_root / split).glob("*.exr"))
        names = [path.name for path in files]
        if names != expected_names(count, ".exr"):
            missing = sorted(set(expected_names(count, ".exr")) - set(names))
            extra = sorted(set(names) - set(expected_names(count, ".exr")))
            raise ValueError(
                f"{split}: expected {count} sequential EXRs; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        resolutions = Counter()
        for path in files:
            metadata = exr_metadata(path)
            resolutions[(metadata["width"], metadata["height"])] += 1
        split_stats[split] = {
            "images": len(files),
            "resolutions": [
                {"width": width, "height": height, "count": resolution_count}
                for (width, height), resolution_count in sorted(resolutions.items())
            ],
        }

    masks = sorted((data_root / "masks").glob("*.png"))
    expected_masks = [f"{index:02d}.png" for index in range(MASK_COUNT)]
    if [path.name for path in masks] != expected_masks:
        raise ValueError(
            f"masks: expected {expected_masks}, found {[path.name for path in masks]}"
        )
    mask_resolutions = Counter(png_size(path) for path in masks)

    artifacts = release_artifacts(data_root)
    if len(artifacts) != sum(SPLIT_COUNTS.values()) + MASK_COUNT:
        raise ValueError("release artifact count does not match the explicit allowlist")
    symlinks = [source for source, _ in artifacts if source.is_symlink()]
    if symlinks:
        raise ValueError(f"symbolic links are not permitted: {symlinks}")

    return {
        "splits": split_stats,
        "masks": {
            "count": len(masks),
            "resolutions": [
                {"width": width, "height": height, "count": resolution_count}
                for (width, height), resolution_count in sorted(
                    mask_resolutions.items()
                )
            ],
        },
    }


def archive_dataset(
    data_root: Path,
    output: Path,
    mode: str,
) -> None:
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("building the release requires GNU tar and zstd")

    archive_dir = output / "archives"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / "reni-hdr.tar.zst"

    with tempfile.TemporaryDirectory(prefix=".reni-hdr-", dir=output) as temporary:
        temporary_root = Path(temporary)
        for source, relative in release_artifacts(data_root):
            link_or_copy(source, temporary_root / relative, mode)
        subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=@0",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "--format=gnu",
                "--use-compress-program=zstd -T0 -6 --no-progress",
                "-cf",
                str(archive_path),
                "-C",
                str(temporary_root),
                "RENI_HDR",
            ],
            check=True,
        )


def copy_release_metadata(
    output: Path,
    repo_id: str,
    version: str,
    commit: str,
) -> None:
    card = (SCRIPT_DIR / "DATASET_CARD.md").read_text()
    replacements = {
        "@@NS_RENI_COMMIT@@": commit,
        "@@RELEASE_DATE@@": date.today().isoformat(),
        "@@RELEASE_VERSION@@": version,
        "@@REPO_ID@@": repo_id,
    }
    for placeholder, value in replacements.items():
        card = card.replace(placeholder, value)
    if "@@" in card:
        raise ValueError("unresolved placeholder in dataset card")
    (output / "README.md").write_text(card)
    shutil.copy2(SCRIPT_DIR / "DATASET_LICENSE.md", output / "LICENSE.md")
    shutil.copy2(SCRIPT_DIR / "DATASET_SOURCES.md", output / "SOURCES.md")


def write_manifests(
    output: Path,
    data_root: Path,
    version: str,
    commit: str,
    dataset_stats: dict[str, object],
) -> None:
    contents = []
    for source, relative in release_artifacts(data_root):
        contents.append(
            {
                "path": relative.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    content_manifest = {
        "schema_version": 1,
        "dataset": "RENI HDR",
        "release_version": version,
        "generator_commit": commit,
        "file_count": len(contents),
        "total_bytes": sum(entry["size_bytes"] for entry in contents),
        "files": contents,
    }
    (output / "CONTENTS.json").write_text(
        json.dumps(content_manifest, indent=2) + "\n"
    )

    stats = {
        "schema_version": 1,
        "dataset": "RENI HDR",
        "release_version": version,
        "generator_commit": commit,
        **dataset_stats,
        "totals": {
            "environment_maps": sum(SPLIT_COUNTS.values()),
            "training": SPLIT_COUNTS["train"],
            "validation": SPLIT_COUNTS["val"],
            "test": SPLIT_COUNTS["test"],
            "completion_masks": MASK_COUNT,
        },
    }
    (output / "DATASET_STATS.json").write_text(json.dumps(stats, indent=2) + "\n")

    entries = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "SHA256SUMS"}:
            continue
        entries.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": "RENI HDR",
        "release_version": version,
        "generated_on": date.today().isoformat(),
        "generator_repository": "https://github.com/JADGardner/ns_reni",
        "generator_commit": commit,
        "licence": "CC0-1.0",
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(
                f"{sha256(path)}  {path.relative_to(output).as_posix()}"
            )
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


def parse_args() -> argparse.Namespace:
    default_data = Path(
        os.environ.get("RENI_HDR_ROOT", Path.home() / "data" / "RENI_HDR")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data,
        help="Working RENI_HDR directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="jadgardner/reni-hdr",
        help="Hugging Face dataset repository ID written into the card",
    )
    parser.add_argument("--version", default="1.0")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to stage files before archiving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    commit = git_commit()
    print("[validate] RENI HDR")
    dataset_stats = validate_dataset(data_root)
    print("[archive] RENI HDR")
    archive_dataset(data_root, output, args.mode)
    copy_release_metadata(output, args.repo_id, args.version, commit)
    print("[hash] release files")
    write_manifests(
        output=output,
        data_root=data_root,
        version=args.version,
        commit=commit,
        dataset_stats=dataset_stats,
    )

    forbidden_names = {
        "sun_labels_pseudo.json",
        "sun_labels_pseudo_v1.json",
        "house.jpg",
    }
    forbidden_parts = {"3d_models", "irn_test"}
    forbidden = [
        path
        for path in output.rglob("*")
        if path.name in forbidden_names
        or any(part in forbidden_parts for part in path.parts)
    ]
    if forbidden:
        raise ValueError(f"forbidden release paths: {forbidden}")

    manifest = json.loads((output / "MANIFEST.json").read_text())
    print(
        f"[done] {manifest['file_count']} release files, "
        f"{manifest['total_bytes'] / 2**20:.2f} MiB at {output}"
    )


if __name__ == "__main__":
    main()
