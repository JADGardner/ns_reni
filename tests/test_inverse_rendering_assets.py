"""Tests for the reproducible inverse-rendering normal-map builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "inverse_rendering_assets"
    / "build_normal_maps.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_normal_maps", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sources_are_checksum_pinned():
    builder = _load_builder()

    assert set(builder.SOURCES) == {"bunny", "teapot"}
    assert all(len(source.sha256) == 64 for source in builder.SOURCES.values())
    assert "bunny2.ply" in builder.SOURCES["bunny"].url
    assert "2eb01f5cd4c2dae4e1ef9912ca27a93083bb6ef4" in (
        builder.SOURCES["teapot"].url
    )


def test_y_up_meshes_are_rotated_into_the_z_up_render_world():
    builder = _load_builder()
    basis = np.eye(3) @ builder.Y_UP_TO_Z_UP.T

    np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(basis[1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(basis[2], [0.0, -1.0, 0.0])


def test_angle_weighted_normals_are_unit_length():
    builder = _load_builder()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    normals = builder.angle_weighted_vertex_normals(vertices, faces)

    np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]] * 4)


def test_default_camera_metadata_matches_the_accepted_assets():
    builder = _load_builder()
    metadata = builder.camera_metadata(1000)

    assert metadata["fl_x"] == builder.REFERENCE_FOCAL_LENGTH
    assert metadata["fl_y"] == builder.REFERENCE_FOCAL_LENGTH
    assert metadata["cx"] == 500.0
    assert metadata["cy"] == 500.0
    assert [frame["file_path"] for frame in metadata["frames"]] == [
        "bunny_normals.exr",
        "teapot_normals.exr",
    ]
