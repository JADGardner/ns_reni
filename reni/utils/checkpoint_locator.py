"""Single checkpoint-path resolver for figure/eval/comparison scripts.

All scripts reference checkpoints by a relative name (the historical
``checkpoints/...`` layout) and this module maps that name onto whichever
storage root actually holds it. No symlinks, one lookup order:

    1. $RENI_CHECKPOINT_ROOTS (colon-separated, overrides everything)
    2. <repo>/checkpoints                       repo-local extras (SOLD_Net,
                                                inverserendernet, reni_original)
    3. $RENI_PAPER_MODELS                       explicit paper-archive copy
    4. ~/model-storage/reni_paper_models        canonical paper model archive
    5. <repo>/checkpoints/paper_models          scripts/download_models.py target
    6. ~/model-storage/reni_paper_baselines[/checkpoints]
                                                baselines archive (IRN_v2 etc.)

Archives are downloadable via scripts/download_models.py (paper models) and
the reni_plus_plus_baselines_comparisons.zip companion archive.
"""

import os
from pathlib import Path
from typing import Union

REPO_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_roots():
    env = os.environ.get("RENI_CHECKPOINT_ROOTS")
    if env:
        return [Path(p) for p in env.split(":") if p]
    roots = [REPO_ROOT / "checkpoints"]
    if os.environ.get("RENI_PAPER_MODELS"):
        roots.append(Path(os.environ["RENI_PAPER_MODELS"]))
    roots += [
        Path.home() / "model-storage" / "reni_paper_models",
        REPO_ROOT / "checkpoints" / "paper_models",
        Path.home() / "model-storage" / "reni_paper_baselines",
        Path.home() / "model-storage" / "reni_paper_baselines" / "checkpoints",
    ]
    return roots


def find_checkpoint(rel: Union[str, Path]) -> Path:
    """Resolve a checkpoint path. Accepts absolute paths (returned if they
    exist), legacy 'checkpoints/<rel>' names, or bare '<rel>' names."""
    p = Path(rel)
    if p.is_absolute():
        if p.exists():
            return p
        rel = os.path.relpath(p, REPO_ROOT)  # container/legacy absolute path
    rel = str(rel)
    if rel.startswith("checkpoints/"):
        rel = rel[len("checkpoints/"):]
    for root in checkpoint_roots():
        candidate = root / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Checkpoint '{rel}' not found under any root:\n  "
        + "\n  ".join(str(r) for r in checkpoint_roots())
        + "\nDownload the archives with scripts/download_models.py")
