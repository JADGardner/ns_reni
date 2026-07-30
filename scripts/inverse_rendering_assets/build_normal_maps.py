#!/usr/bin/env python3
"""Rebuild the bunny and teapot normal maps used by the inverse task.

The source meshes are downloaded from immutable or checksum-pinned public
locations.  No local OBJ file or Blender scene is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REFERENCE_RESOLUTION = 1000
REFERENCE_FOCAL_LENGTH = 1388.888888888889
CAMERA_TO_WORLD = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -4.371138828673793e-8, -1.0, -3.0],
    [0.0, 1.0, -4.371138828673793e-8, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# The public meshes are y-up. This is the object transform used to render them
# in the z-up world represented by CAMERA_TO_WORLD.
Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# bunny2.ply has the exact 72,027 vertices and 144,046 faces of the research
# asset. These constants recover its established scale, pose, and framing.
BUNNY_SCALE = 10.499626603015894
BUNNY_ROTATION = np.array(
    [
        [1.0000000000000000, -0.0000001459356370, 0.0000005327567140],
        [0.0000000999168185, 0.9963485930000000, 0.0853784608000000],
        [-0.0000005432711630, -0.0853784608000000, 0.9963485930000000],
    ],
    dtype=np.float64,
)
BUNNY_TRANSLATION = np.array(
    [0.19999987, -0.98869257, -0.01785992],
    dtype=np.float64,
)
TEAPOT_TRANSLATION = np.array([-0.06, -0.5, -0.02], dtype=np.float64)


@dataclass(frozen=True)
class SourceAsset:
    filename: str
    url: str
    sha256: str


SOURCES = {
    "bunny": SourceAsset(
        filename="bunny2.ply",
        url="https://pixl.cs.princeton.edu/proj/sugcon/models/bunny2.ply",
        sha256="b0d6c74b937db46d0684a54c959dda1eb0cc2a16bf4bca0247c8b0da03df031a",
    ),
    "teapot": SourceAsset(
        filename="utah_teapot.obj",
        url=(
            "https://raw.githubusercontent.com/PixarAnimationStudios/OpenUSD/"
            "2eb01f5cd4c2dae4e1ef9912ca27a93083bb6ef4/"
            "extras/usd/examples/usdObj/teapot.obj"
        ),
        sha256="e52b2ae40e9e3b8e7af7e9a8bfa95f471c610e853632bf2fe77e7272124edaa2",
    ),
}


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    corner_normals: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source(asset: SourceAsset, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / asset.filename

    if destination.exists():
        actual = sha256_file(destination)
        if actual != asset.sha256:
            raise RuntimeError(
                f"Checksum mismatch for cached {destination}: "
                f"expected {asset.sha256}, found {actual}. Remove the file and retry."
            )
        print(f"Using cached source: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "ns-reni-inverse-assets/1.0"},
    )
    print(f"Downloading {asset.url}")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

        actual = sha256_file(partial)
        if actual != asset.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {asset.url}: "
                f"expected {asset.sha256}, found {actual}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)

    return destination


def angle_weighted_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    triangles = vertices[faces]
    face_cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_normals = face_cross / np.linalg.norm(face_cross, axis=-1, keepdims=True)
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)

    for corner in range(3):
        edge_a = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        edge_b = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        cosine = np.sum(edge_a * edge_b, axis=-1) / (
            np.linalg.norm(edge_a, axis=-1)
            * np.linalg.norm(edge_b, axis=-1)
        )
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        np.add.at(
            vertex_normals,
            faces[:, corner],
            face_normals * angles[:, None],
        )

    lengths = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("The mesh contains a vertex without a valid normal")
    return vertex_normals / lengths


def load_bunny(path: Path) -> Mesh:
    data = path.read_bytes()
    header_marker = b"end_header\n"
    header_end = data.find(header_marker)
    if header_end < 0:
        raise ValueError(f"{path} is not a supported binary PLY file")
    header_end += len(header_marker)
    header = data[:header_end].decode("ascii")

    expected_lines = {
        "format binary_little_endian 1.0",
        "element vertex 72027",
        "element tristrips 1",
        "property list int int vertex_indices",
    }
    if not expected_lines.issubset(set(header.splitlines())):
        raise ValueError(f"{path} does not match the pinned bunny2 PLY layout")

    vertex_count = 72027
    vertices = np.frombuffer(
        data,
        dtype="<f4",
        count=vertex_count * 3,
        offset=header_end,
    ).reshape(vertex_count, 3).astype(np.float64)

    strip_offset = header_end + vertex_count * 3 * np.dtype("<f4").itemsize
    index_count = struct.unpack_from("<i", data, strip_offset)[0]
    strip_indices = np.frombuffer(
        data,
        dtype="<i4",
        count=index_count,
        offset=strip_offset + np.dtype("<i4").itemsize,
    )

    faces: list[list[int]] = []
    strip: list[int] = []
    for value in strip_indices:
        if value < 0:
            strip.clear()
            continue
        strip.append(int(value))
        if len(strip) < 3:
            continue
        triangle = strip[-3:].copy()
        if (len(strip) - 3) % 2:
            triangle[0], triangle[1] = triangle[1], triangle[0]
        if len(set(triangle)) == 3:
            faces.append(triangle)

    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.shape != (144046, 3):
        raise ValueError(
            f"Expected 144046 bunny triangles, found {len(face_array)}"
        )

    vertices = BUNNY_SCALE * vertices @ BUNNY_ROTATION + BUNNY_TRANSLATION
    normals = angle_weighted_vertex_normals(vertices, face_array)
    vertices = vertices @ Y_UP_TO_Z_UP.T
    normals = normals @ Y_UP_TO_Z_UP.T
    return Mesh(
        vertices=vertices,
        faces=face_array,
        corner_normals=normals[face_array],
    )


def load_teapot(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    faces: list[list[int]] = []
    normal_indices: list[list[int]] = []

    with path.open(encoding="ascii") as source:
        for line in source:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("vn "):
                normals.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                face_vertices: list[int] = []
                face_normals: list[int] = []
                for element in line.split()[1:]:
                    fields = element.split("/")
                    face_vertices.append(int(fields[0]) - 1)
                    face_normals.append(int(fields[-1]) - 1)
                if len(face_vertices) != 3:
                    raise ValueError(f"{path} contains a non-triangular face")
                faces.append(face_vertices)
                normal_indices.append(face_normals)

    vertex_array = np.asarray(vertices, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    normal_index_array = np.asarray(normal_indices, dtype=np.int64)
    if vertex_array.shape != (1292, 3) or face_array.shape != (2464, 3):
        raise ValueError(
            "The teapot source does not have the expected "
            "1292 vertices and 2464 triangles"
        )

    vertex_array = (vertex_array + TEAPOT_TRANSLATION) @ Y_UP_TO_Z_UP.T
    normal_array = normal_array @ Y_UP_TO_Z_UP.T
    normal_array /= np.linalg.norm(normal_array, axis=-1, keepdims=True)
    return Mesh(
        vertices=vertex_array,
        faces=face_array,
        corner_normals=normal_array[normal_index_array],
    )


def render_normal_map(mesh: Mesh, resolution: int) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "open3d is required to render the normal maps. "
            "Install ns_reni in its documented environment."
        ) from error

    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(mesh.vertices.astype(np.float32)),
        o3d.core.Tensor(mesh.faces.astype(np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    scale = resolution / REFERENCE_RESOLUTION
    focal_length = REFERENCE_FOCAL_LENGTH * scale
    principal_point = resolution / 2.0
    pixels = np.arange(resolution, dtype=np.float32) + 0.5
    camera_x, camera_y = np.meshgrid(
        (pixels - principal_point) / focal_length,
        -(pixels - principal_point) / focal_length,
    )
    directions = np.stack(
        [camera_x, np.ones_like(camera_x), camera_y],
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.zeros_like(directions)
    origins[..., 1] = -3.0
    rays = np.concatenate([origins, directions], axis=-1)

    intersections = scene.cast_rays(o3d.core.Tensor(rays))
    primitive_ids = intersections["primitive_ids"].numpy()
    primitive_uvs = intersections["primitive_uvs"].numpy()
    hit = primitive_ids != np.iinfo(np.uint32).max

    normal_map = np.zeros((resolution, resolution, 3), dtype=np.float32)
    triangle_ids = primitive_ids[hit].astype(np.int64)
    uv = primitive_uvs[hit]
    barycentric = np.column_stack([1.0 - uv.sum(axis=-1), uv])
    values = np.sum(
        mesh.corner_normals[triangle_ids] * barycentric[..., None],
        axis=1,
    )
    values /= np.linalg.norm(values, axis=-1, keepdims=True)

    # The accepted Blender maps stored interior normals at half precision in
    # float channels. Preserve that numerical convention.
    normal_map[hit] = values.astype(np.float16).astype(np.float32)
    return normal_map


def write_exr(path: Path, normal_map: np.ndarray) -> None:
    try:
        import Imath
        import pyexr
    except ImportError as error:
        raise RuntimeError(
            "pyexr and OpenEXR are required to write the normal maps. "
            "Install ns_reni in its documented environment."
        ) from error

    pyexr.write(
        path,
        normal_map,
        precision=Imath.PixelType(Imath.PixelType.FLOAT),
        compression=Imath.Compression(Imath.Compression.ZIP_COMPRESSION),
        extra_headers={"xDensity": 72.0},
    )


def camera_metadata(resolution: int) -> dict:
    scale = resolution / REFERENCE_RESOLUTION
    focal_length = REFERENCE_FOCAL_LENGTH * scale
    principal_point = resolution / 2.0
    return {
        "camera_angle_x": 0.6911112070083618,
        "camera_angle_y": 0.4710899591445923,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": principal_point,
        "cy": principal_point,
        "w": float(resolution),
        "h": float(resolution),
        "aabb_scale": 4,
        "frames": [
            {
                "file_path": "bunny_normals.exr",
                "transform_matrix": CAMERA_TO_WORLD,
            },
            {
                "file_path": "teapot_normals.exr",
                "transform_matrix": CAMERA_TO_WORLD,
            },
        ],
    }


def verify_normal_map(
    generated: np.ndarray,
    reference_path: Path,
) -> tuple[float, float, float]:
    try:
        import pyexr
    except ImportError as error:
        raise RuntimeError("pyexr is required for reference verification") from error

    reference = pyexr.read(reference_path)
    if generated.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for {reference_path}: "
            f"generated {generated.shape}, reference {reference.shape}"
        )

    generated_norm = np.linalg.norm(generated, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    generated_mask = generated_norm > 0
    reference_mask = reference_norm > 0
    intersection = generated_mask & reference_mask
    union = generated_mask | reference_mask
    mask_iou = float(intersection.sum() / union.sum())

    generated_unit = generated[intersection] / generated_norm[intersection, None]
    reference_unit = reference[intersection] / reference_norm[intersection, None]
    cosine = np.sum(generated_unit * reference_unit, axis=-1)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    mean_angle = float(np.mean(angles))
    percentile_95 = float(np.percentile(angles, 95))
    return mask_iou, mean_angle, percentile_95


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "RENI_HDR" / "3d_models" / "normal_maps",
        help="Directory for the EXR files and normal_cam_transforms.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "ns_reni" / "inverse_rendering_assets",
        help="Persistent cache for the checksum-verified upstream meshes.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=REFERENCE_RESOLUTION,
        help="Square output resolution. The established assets use 1000.",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Optional directory containing reference EXRs to compare.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    output_paths = {
        name: args.output_dir / f"{name}_normals.exr"
        for name in SOURCES
    }
    metadata_path = args.output_dir / "normal_cam_transforms.json"
    existing = [path for path in [*output_paths.values(), metadata_path] if path.exists()]
    if existing and not args.force:
        paths = "\n".join(f"  {path}" for path in existing)
        print(
            "Refusing to overwrite existing outputs without --force:\n" + paths,
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = {"bunny": load_bunny, "teapot": load_teapot}
    generated: dict[str, np.ndarray] = {}

    for name, asset in SOURCES.items():
        source_path = fetch_source(asset, args.cache_dir)
        mesh = loaders[name](source_path)
        print(
            f"Rendering {name}: {len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} triangles"
        )
        normal_map = render_normal_map(mesh, args.resolution)
        write_exr(output_paths[name], normal_map)
        generated[name] = normal_map
        coverage = np.count_nonzero(np.linalg.norm(normal_map, axis=-1)) / (
            args.resolution * args.resolution
        )
        print(f"Wrote {output_paths[name]} ({coverage:.2%} foreground)")

    metadata_path.write_text(
        json.dumps(camera_metadata(args.resolution), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {metadata_path}")

    if args.verify_against:
        failed = False
        for name, normal_map in generated.items():
            reference_path = args.verify_against / f"{name}_normals.exr"
            mask_iou, mean_angle, percentile_95 = verify_normal_map(
                normal_map,
                reference_path,
            )
            print(
                f"{name} reference check: mask IoU={mask_iou:.6f}, "
                f"mean angle={mean_angle:.4f} deg, "
                f"p95 angle={percentile_95:.4f} deg"
            )
            failed |= mask_iou < 0.9998 or mean_angle > 0.15
        if failed:
            print("Reference fidelity check failed", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
