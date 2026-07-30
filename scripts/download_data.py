#!/usr/bin/env python3
"""Download and verify the public RENI HDR dataset release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import requests


RELEASE_VERSION = "1.0"
ARCHIVE_NAME = f"reni-hdr-v{RELEASE_VERSION}.tar.zst"
ARCHIVE_SHA256 = "32f902e94d0844d9a909121dc61b3b3f0a823194d6328d740286f0c799d90ff8"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/jadgardner/reni-hdr/"
    f"resolve/v{RELEASE_VERSION}/archives/reni-hdr.tar.zst?download=true"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                downloaded += len(block)
                if total:
                    print(
                        f"\rDownloaded {downloaded / 2**20:.1f} / "
                        f"{total / 2**20:.1f} MiB",
                        end="",
                        flush=True,
                    )
    if total:
        print()
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        help="Data root into which RENI_HDR/ will be extracted",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive after successful extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ARCHIVE_NAME

    if archive.is_file() and sha256(archive) == ARCHIVE_SHA256:
        print(f"Using verified archive: {archive}")
    else:
        if archive.exists():
            archive.unlink()
        print(f"Downloading RENI HDR v{RELEASE_VERSION} from Hugging Face...")
        download(ARCHIVE_URL, archive)

    actual_sha256 = sha256(archive)
    if actual_sha256 != ARCHIVE_SHA256:
        raise RuntimeError(
            "RENI HDR archive checksum mismatch: "
            f"expected {ARCHIVE_SHA256}, found {actual_sha256}"
        )
    print(f"Verified SHA256: {actual_sha256}")

    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("extraction requires GNU tar and zstd")
    subprocess.run(
        ["tar", "--zstd", "-xf", str(archive), "-C", str(output)],
        check=True,
    )

    dataset = output / "RENI_HDR"
    if not dataset.is_dir():
        raise RuntimeError(f"archive did not create the expected directory: {dataset}")
    if not args.keep_archive:
        archive.unlink()
    print(f"RENI HDR is ready at {dataset}")


if __name__ == "__main__":
    main()
