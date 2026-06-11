import torch
from torch import sin, cos, atan2, acos
from pathlib import Path


def find_nerfstudio_project_root(start_dir: Path = Path(".")) -> Path:
    """Find the ns_reni repo root.

    Prefer the repo's own markers (the reni package next to pyproject.toml);
    fall back to the historical heuristic of a directory containing
    'nerfstudio' (a vendored checkout), which mis-resolves on machines where
    a sibling nerfstudio clone exists above the repo.
    """
    start_dir = Path(start_dir).resolve()
    for path in [start_dir, *start_dir.parents]:
        if (path / "reni").is_dir() and (path / "pyproject.toml").exists():
            return path
    for path in [start_dir, *start_dir.parents]:
        if (path / "nerfstudio").exists():
            return path
    raise ValueError("Project root not found.")

# https://github.com/lucidrains/VN-transformer/blob/main/VN_transformer/rotations.py

def rot_z(gamma: torch.Tensor):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)