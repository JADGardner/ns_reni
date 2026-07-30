---
pretty_name: RENI HDR
license: cc0-1.0
size_categories:
  - 1K<n<10K
tags:
  - image
  - computer-vision
  - high-dynamic-range
  - environment-maps
  - illumination
  - inverse-rendering
  - neural-rendering
  - outdoor-scenes
---

# RENI HDR

RENI HDR is a curated dataset of outdoor, natural high-dynamic-range
illumination environments used to train and evaluate
[RENI and RENI++](https://github.com/JADGardner/ns_reni). Each image is a
full equirectangular environment map in linear RGB radiance.

| Property | Value |
|---|---:|
| Environment maps | 1,704 |
| Training | 1,673 |
| Validation | 10 |
| Test | 21 |
| Fixed completion masks | 5 |
| Licence | CC0 1.0 |
| Release | @@RELEASE_VERSION@@ |
| Release date | @@RELEASE_DATE@@ |
| Release builder | [`@@NS_RENI_COMMIT@@`](https://github.com/JADGardner/ns_reni/tree/@@NS_RENI_COMMIT@@) |

The source panoramas were obtained under CC0 and manually curated. Images
containing personally identifiable information, offensive content, or
predominantly unnatural light sources were removed.

## Contents

The archive extracts to the directory expected by `ns_reni`:

```text
RENI_HDR/
  train/       # 1,673 EXRs at 128 x 64
  val/         # 10 high-resolution EXRs at 1024 x 512
  test/        # 21 held-out EXRs at 128 x 64
  masks/       # five fixed 512 x 256 completion masks
```

The test images are held out from decoder training and are used by fitting
only their latent codes. The validation maps retain their high-resolution
versions for qualitative evaluation; the loader resamples them when a fixed
query resolution is requested.

This release deliberately excludes the working directory's later pseudo-sun
labels, Blender files, object meshes, inverse-rendering renders and unrelated
image assets. Those are not part of the curated illumination corpus.

## Download

Download and verify the tagged release:

```bash
hf download @@REPO_ID@@ \
  --repo-type dataset \
  --revision v@@RELEASE_VERSION@@ \
  --local-dir reni-hdr

(cd reni-hdr && sha256sum -c SHA256SUMS)
tar --zstd -xf reni-hdr/archives/reni-hdr.tar.zst -C /path/to/data
```

The result is `/path/to/data/RENI_HDR`, which can be passed directly to
RENI++:

```bash
ns-train reni --data /path/to/data/RENI_HDR
```

The archive can also be downloaded without the Hugging Face CLI:

```bash
curl -L \
  "https://huggingface.co/datasets/@@REPO_ID@@/resolve/v@@RELEASE_VERSION@@/archives/reni-hdr.tar.zst?download=true" \
  -o reni-hdr.tar.zst
```

`CONTENTS.json` records the release-relative path, size and SHA256 of all
1,709 extracted files. `DATASET_STATS.json` records the validated split
counts and resolutions. `MANIFEST.json` and `SHA256SUMS` cover the
downloadable release files.

## Provenance

The environments were curated from the CC0 sources listed in `SOURCES.md`.
The collected files were sequentially renamed during the original curation,
so a per-image source-name mapping is not available. This limitation is
recorded explicitly rather than inferring provenance after the fact.

## Licence

The dataset is released under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
See `LICENSE.md`.

## Citation

Please cite the RENI and RENI++ publications:

```bibtex
@inproceedings{gardner2022reni,
  title     = {Rotation-Equivariant Conditional Spherical Neural Fields for
               Learning a Natural Illumination Prior},
  author    = {Gardner, James A. D. and Egger, Bernhard and
               Smith, William A. P.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {26309--26323},
  year      = {2022}
}

@article{gardner2026renipp,
  title   = {{RENI++}: A Rotation-Equivariant, Scale-Invariant, Natural
             Illumination Prior},
  author  = {Gardner, James A. D. and Egger, Bernhard and
             Smith, William A. P.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  doi     = {10.1109/TPAMI.2026.3691593}
}
```
