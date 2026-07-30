#!/usr/bin/env python3
"""Download and verify released RENI checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import requests


REPO_ID = "jadgardner/reni-models"
RELEASE_VERSION = "1.0"
REVISION = f"v{RELEASE_VERSION}"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}"
MANIFEST_NAME = "MODEL_MANIFEST.json"
DEFAULT_GROUP = "core"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    resume_at = temporary.stat().st_size if temporary.is_file() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(10, 300),
    ) as response:
        response.raise_for_status()
        append = resume_at > 0 and response.status_code == 206
        if not append:
            resume_at = 0
        total = int(response.headers.get("content-length", 0)) + resume_at
        downloaded = resume_at
        with temporary.open("ab" if append else "wb") as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                downloaded += len(block)
                if total:
                    print(
                        f"\r{destination.name}: {downloaded / 2**20:.1f} / "
                        f"{total / 2**20:.1f} MiB",
                        end="",
                        flush=True,
                    )
    if total:
        print()
    temporary.replace(destination)


def fetch_manifest() -> dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/{MANIFEST_NAME}?download=true",
        timeout=(10, 60),
    )
    response.raise_for_status()
    manifest = response.json()
    if manifest.get("release_version") != RELEASE_VERSION:
        raise RuntimeError(
            f"expected release {RELEASE_VERSION}, found "
            f"{manifest.get('release_version')!r}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("model-storage/reni"),
        help="Download root (default: model-storage/reni)",
    )
    parser.add_argument(
        "--group",
        action="append",
        help=(
            "Release group to download; repeat to combine groups "
            f"(default: {DEFAULT_GROUP})"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Exact model id to download; repeat to select several models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available groups and model ids without downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = fetch_manifest()
    models = manifest["models"]

    if args.list:
        groups = sorted({group for model in models for group in model["groups"]})
        print("Groups:")
        for group in groups:
            print(f"  {group}")
        print("\nModels:")
        for model in models:
            print(
                f"  {model['id']:<48} "
                f"[{', '.join(model['groups'])}]"
            )
        return

    requested_groups = set(args.group or ([] if args.model else [DEFAULT_GROUP]))
    requested_models = set(args.model or [])
    known_groups = {group for model in models for group in model["groups"]}
    known_models = {model["id"] for model in models}
    unknown_groups = requested_groups - known_groups
    unknown_models = requested_models - known_models
    if unknown_groups:
        raise SystemExit(f"unknown group(s): {', '.join(sorted(unknown_groups))}")
    if unknown_models:
        raise SystemExit(f"unknown model(s): {', '.join(sorted(unknown_models))}")

    selected = [
        model
        for model in models
        if requested_models.intersection({model["id"]})
        or requested_groups.intersection(model["groups"])
    ]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for model in selected:
        print(f"\n{model['id']}")
        for file_info in model["files"]:
            relative = Path(file_info["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe release path: {relative}")
            destination = output / relative
            expected = file_info["sha256"]
            if destination.is_file() and sha256(destination) == expected:
                print(f"  verified {relative}")
                continue
            if destination.exists():
                destination.unlink()
            download(
                f"{BASE_URL}/{relative.as_posix()}?download=true",
                destination,
            )
            actual = sha256(destination)
            if actual != expected:
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"checksum mismatch for {relative}: "
                    f"expected {expected}, found {actual}"
                )
            print(f"  SHA256 {actual}")

    print(f"\nDownloaded {len(selected)} model(s) to {output}")


if __name__ == "__main__":
    main()
