# Inverse-Rendering Normal Maps

This script rebuilds the two world-space normal maps used by the RENI inverse
rendering task without retaining private OBJ files or a Blender scene:

```bash
python scripts/inverse_rendering_assets/build_normal_maps.py \
  --output-dir data/RENI_HDR/3d_models/normal_maps
```

Use `--force` to replace existing files. The command downloads and verifies
the source meshes, then writes:

- `bunny_normals.exr`
- `teapot_normals.exr`
- `normal_cam_transforms.json`

## Sources

The bunny is
[`bunny2.ply`](https://pixl.cs.princeton.edu/proj/sugcon/models/bunny2.ply)
from the
[Princeton Suggestive Contour Gallery](https://pixl.cs.princeton.edu/proj/sugcon/models/).
It is the 144,046-triangle, VRIP-merged and hole-filled reconstruction made
from the original high-resolution scans in the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
The script pins SHA256
`b0d6c74b937db46d0684a54c959dda1eb0cc2a16bf4bca0247c8b0da03df031a`.
Stanford and Princeton describe this model as available for research and
non-commercial use with acknowledgement.

The teapot is the 1,292-vertex, 2,464-triangle Utah Teapot mesh distributed
with [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD), pinned to
commit `2eb01f5cd4c2dae4e1ef9912ca27a93083bb6ef4` and SHA256
`e52b2ae40e9e3b8e7af7e9a8bfa95f471c610e853632bf2fe77e7272124edaa2`.
The mesh is covered by OpenUSD's modified Apache 2.0 licence. Its historical
origin is the
[Utah Teapot](https://graphics.cs.utah.edu/teapot/); the University of Utah
page permits any use and asks that derived models retain that identification
and acknowledgement.

## Fidelity

Both upstream meshes are transformed to the scale, pose and z-up world used
by the accepted inverse-rendering assets. Normals are interpolated from smooth
vertex normals and stored in float EXR channels after half-precision
quantisation, matching the numerical convention of the original Blender
renders.

The old EXRs applied Blender's subpixel filter at the silhouette. The rebuilt
maps use deterministic pixel-centre visibility. This changes only a few
boundary samples and has no effect after the dataset's existing nonzero mask
and normalisation. To compare a rebuild with the accepted files:

```bash
python scripts/inverse_rendering_assets/build_normal_maps.py \
  --output-dir /tmp/reni-inverse-normal-maps \
  --verify-against data/RENI_HDR/3d_models/normal_maps
```

At 1000 by 1000, the check requires mask IoU above `0.9998` and mean angular
normal error below `0.15` degrees.
