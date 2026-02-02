"""
Render normal maps for additional 3D objects using trimesh ray casting.

Generates world-space normal maps matching the camera setup in
data/RENI_HDR/3d_models/normal_maps/normal_cam_transforms.json.

Usage:
    python notebooks/render_normal_maps.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyexr
import trimesh


def create_camera_rays(width, height, fx, fy, cx, cy, cam_to_world):
    """Generate world-space rays matching the existing camera setup.

    The camera transform is a 4x4 matrix where:
    - Column 0 = right (+X in camera space)
    - Column 1 = up (but ~+Z in world due to Blender convention)
    - Column 2 = -forward (-look direction)
    - Column 3 = position

    The transform matrix is:
        [1,  0,  0,  0 ]
        [0, ~0, -1, -3 ]
        [0,  1, ~0,  0 ]
    Camera at (0, -3, 0), looking along +Y, up = +Z.
    """
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    # Camera-space directions
    # The transform matrix uses Blender/NeRF convention where -Z is forward.
    # So camera-space direction for pixel (u,v) is ((u-cx)/fx, (v-cy)/fy, -1)
    dirs_cam = np.stack(
        [(uu - cx) / fx, (vv - cy) / fy, -np.ones_like(uu)], axis=-1
    )

    # Normalize
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # Extract rotation (3x3) and translation from cam_to_world
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]

    # Transform directions to world space
    dirs_world = dirs_cam @ R.T  # (H, W, 3)

    # All rays originate from the camera position
    origins = np.broadcast_to(t, dirs_world.shape).copy()

    return origins.reshape(-1, 3), dirs_world.reshape(-1, 3)


def render_normal_map(mesh, ray_origins, ray_directions, width, height):
    """Ray-cast mesh and extract face normals."""
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
    )

    normal_map = np.zeros((height * width, 3), dtype=np.float32)

    if len(index_ray) > 0:
        # Use face normals
        normals = mesh.face_normals[index_tri].astype(np.float32)
        # Ensure normals point toward camera (dot with -ray_direction > 0)
        dots = np.sum(normals * (-ray_directions[index_ray]), axis=-1)
        normals[dots < 0] *= -1
        # Normalize
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        normal_map[index_ray] = normals

    return normal_map.reshape(height, width, 3)


def create_meshes():
    """Create meshes using trimesh.creation."""
    meshes = {}

    # Sphere (high subdivision for smooth normals)
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    meshes["sphere"] = sphere

    # Torus
    torus = trimesh.creation.torus(major_radius=0.7, minor_radius=0.3, major_sections=128, minor_sections=64)
    meshes["torus"] = torus

    # Knot (trefoil knot — complex self-occluding geometry)
    # Create via revolution of a circle along a trefoil path
    from shapely.geometry import Point
    n_path = 500
    n_section = 32
    t = np.linspace(0, 2 * np.pi, n_path, endpoint=False)
    # Trefoil knot parametric equations
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    knot_path = np.column_stack([x, y, z]) * 0.25
    # Cross-section circle as a shapely polygon
    cross_section = Point(0, 0).buffer(0.12, resolution=n_section)
    knot = trimesh.creation.sweep_polygon(
        polygon=cross_section,
        path=knot_path,
    )
    meshes["knot"] = knot

    return meshes


def prepare_mesh(mesh, target_extent=1.2):
    """Center and scale mesh to fit in frame."""
    mesh.vertices -= mesh.centroid
    current_extent = mesh.bounding_box.extents.max()
    mesh.apply_scale(target_extent / current_extent)
    # Recompute normals
    mesh.fix_normals()
    return mesh


def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "RENI_HDR" / "3d_models" / "normal_maps"
    json_path = output_dir / "normal_cam_transforms.json"

    # Load camera config
    with open(json_path) as f:
        cam_config = json.load(f)

    width = int(cam_config["w"])
    height = int(cam_config["h"])
    fx = cam_config["fl_x"]
    fy = cam_config["fl_y"]
    cx = cam_config["cx"]
    cy = cam_config["cy"]

    # Camera-to-world from the existing transform (same for all objects)
    cam_to_world = np.array(cam_config["frames"][0]["transform_matrix"], dtype=np.float64)

    print(f"Camera: {width}x{height}, f=({fx:.1f}, {fy:.1f}), c=({cx:.1f}, {cy:.1f})")
    print(f"Camera position: {cam_to_world[:3, 3]}")

    # Generate rays
    print("Generating camera rays...")
    ray_origins, ray_directions = create_camera_rays(
        width, height, fx, fy, cx, cy, cam_to_world
    )

    # Create meshes
    print("Creating meshes...")
    meshes = create_meshes()

    # Render each mesh
    new_frames = []
    for name, mesh in meshes.items():
        print(f"\nRendering {name}...")
        mesh = prepare_mesh(mesh)
        print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

        normal_map = render_normal_map(mesh, ray_origins, ray_directions, width, height)

        # Check coverage
        mask = np.linalg.norm(normal_map, axis=-1) > 0.5
        coverage = mask.sum() / mask.size * 100
        print(f"  Coverage: {coverage:.1f}%")

        # Save EXR
        out_path = output_dir / f"{name}_normals.exr"
        pyexr.write(str(out_path), normal_map)
        print(f"  Saved: {out_path}")

        new_frames.append(
            {
                "file_path": f"{name}_normals.exr",
                "transform_matrix": cam_config["frames"][0]["transform_matrix"],
            }
        )

        # Quick visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # Show normal map as RGB (remap [-1,1] to [0,1])
        vis = (normal_map + 1) / 2
        vis[~mask] = 1.0  # white background
        axes[0].imshow(vis)
        axes[0].set_title(f"{name} — Normal Map")
        axes[0].axis("off")
        # Show mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"{name} — Mask ({coverage:.1f}%)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_preview.png", dpi=100)
        plt.close()

    # Update JSON
    cam_config["frames"].extend(new_frames)
    with open(json_path, "w") as f:
        json.dump(cam_config, f, indent=2)
    print(f"\nUpdated {json_path}")

    print("\nDone! Generated normal maps for:", list(meshes.keys()))


if __name__ == "__main__":
    main()
