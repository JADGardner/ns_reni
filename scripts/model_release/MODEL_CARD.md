---
license: apache-2.0
library_name: nerfstudio
tags:
- inverse-rendering
- neural-fields
- illumination
- hdr
- nerfstudio
---

# RENI and RENI++ Checkpoints

This repository contains the released checkpoints for
[RENI++](https://github.com/JADGardner/ns_reni) and the RENI experiments in
*Learning Representations for Incomplete Spherical Scene Signals*.

The release is deliberately divided into groups:

- `minimal`: the headline decoder as a self-contained, PyTorch-only
  reference artifact and locked CPU rendering example.
- `core`: the thesis headline model, with joint Gram-Schmidt invariant
  conditioning, two HDR brackets and two latent-reset cycles.
- `neusky-prior`: the channelwise two-bracket prior used by every released
  NeuSky checkpoint. This is intentionally not the joint-frame thesis model.
- `thesis`: the size, equivariance, invariant-function, seed and latent-reset
  checkpoints needed for the thesis experiments.
- `published`: the final checkpoints from the published RENI and RENI++
  experiments, including SH/SG and inverse-rendering baselines.

`MODEL_MANIFEST.json` records the source run, selected checkpoint, byte size
and SHA256 of every released file. Only final, named checkpoints are included;
optimizer trajectories, failed runs, logs and Weights & Biases state are not.

## Download

From a clone of `ns_reni`, download the headline model:

```bash
python scripts/download_models.py model-storage/reni
```

Download another release group:

```bash
python scripts/download_models.py model-storage/reni --group minimal
python scripts/download_models.py model-storage/reni --group neusky-prior
python scripts/download_models.py model-storage/reni --group thesis
python scripts/download_models.py model-storage/reni --group published
```

List exact model identifiers or download one model:

```bash
python scripts/download_models.py --list
python scripts/download_models.py model-storage/reni \
  --model thesis-vnjoint-ortho-so2-d100
```

Files can also be retrieved directly with `curl` or the Hugging Face CLI.
For example:

```bash
hf download jadgardner/reni-models \
  --revision v1.1 \
  --include "minimal/*" \
  --local-dir model-storage/reni
```

## Minimal PyTorch Inference

The `minimal` group is the quickest way to evaluate the thesis model. It
contains only the 3.57 MiB decoder weights and the code needed to sample an
environment map:

```bash
python scripts/download_models.py model-storage/reni --group minimal
cd model-storage/reni/minimal
uv run render.py --weights decoder.pt --output-dir render
```

It requires no Nerfstudio, tiny-cuda-nn, COLMAP, CUDA, or `ns_reni`
installation. The artifact includes the joint Vector Neuron frame, attention
decoder, architecture metadata and two-bracket HDR constants. The full
checkpoint remains available in `core` for continued training and analysis of
the learned training latents.

## Data

The corresponding HDR panorama data are released separately as
[RENI HDR v1.0](https://huggingface.co/datasets/jadgardner/reni-hdr).

## Licence

The checkpoint release follows the Apache 2.0 licence in the RENI++
repository. Third-party source assets and datasets retain their own licences.
