# RENI HDR Release

This directory defines the public, allowlisted release of the RENI HDR
illumination corpus.

Included:

- `train/*.exr`
- `val/*.exr`
- `test/*.exr`
- `masks/*.png`

Excluded:

- pseudo-sun labels created by later experiments;
- `3d_models/` and `irn_test/`;
- Blender, mesh, rendered inverse-task and unrelated image assets.

Build the release:

```bash
python scripts/reni_hdr_dataset/build_hf_release.py \
  --data-root ~/data/RENI_HDR \
  --output /path/to/reni-hdr-v1
```

The output contains one deterministic archive, split statistics, an
extracted-file manifest, release checksums, the Hugging Face dataset card,
source notes and the CC0 licence.
