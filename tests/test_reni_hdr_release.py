"""Release invariants for the RENI HDR dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path


GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "reni_hdr_dataset"


def _load_release_builder():
    path = GENERATOR / "build_hf_release.py"
    spec = importlib.util.spec_from_file_location("build_reni_hdr_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_uses_the_curated_core_dataset_only():
    builder = _load_release_builder()

    assert builder.SPLIT_COUNTS == {"train": 1673, "val": 10, "test": 21}
    assert builder.MASK_COUNT == 5
    assert sum(builder.SPLIT_COUNTS.values()) == 1704


def test_dataset_card_records_scope_and_provenance_limit():
    card = (GENERATOR / "DATASET_CARD.md").read_text()

    assert "license: cc0-1.0" in card
    assert "1,704" in card
    assert "pseudo-sun" in card
    assert "per-image source-name mapping is not available" in card
    assert "Egger, Bernhard" in card
    assert "Advances in Neural Information Processing Systems" in card
    assert "Scale-Invariant" in card
    assert "@@REPO_ID@@" in card
    assert "--revision v@@RELEASE_VERSION@@" in card
    assert "/resolve/v@@RELEASE_VERSION@@/" in card
