"""Release invariants for the public RENI model bundle."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MINIMAL_ROOT = ROOT / "examples" / "minimal_inference"
RELEASE_ROOT = ROOT / "scripts" / "model_release"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_minimal_environment_uses_patched_dependencies() -> None:
    project = tomllib.loads((MINIMAL_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((MINIMAL_ROOT / "uv.lock").read_text())
    locked = {package["name"]: package["version"] for package in lock["package"]}

    assert project["project"]["dependencies"] == [
        "numpy==2.2.6",
        "pillow==12.3.0",
        "torch==2.13.0+cpu",
    ]
    assert project["tool"]["uv"]["find-links"] == [
        "https://download.pytorch.org/whl/cpu/torch/"
    ]
    assert "extra-index-url" not in project["tool"]["uv"]
    assert locked["pillow"] == "12.3.0"
    assert locked["torch"] == "2.13.0+cpu"
    assert locked["setuptools"] == "83.0.0"


def test_model_release_tools_target_the_same_revision() -> None:
    builder = _load_module(
        "build_reni_model_release",
        RELEASE_ROOT / "build_hf_release.py",
    )
    downloader = _load_module(
        "download_reni_models",
        ROOT / "scripts" / "download_models.py",
    )
    card = (RELEASE_ROOT / "MODEL_CARD.md").read_text()

    assert builder.RELEASE_VERSION == "1.2"
    assert downloader.RELEASE_VERSION == builder.RELEASE_VERSION
    assert "--revision v1.2" in card
