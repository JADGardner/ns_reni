# Minimal RENI++ Inference

This directory evaluates the thesis RENI++ illumination prior with a small
CPU environment containing PyTorch, NumPy and Pillow. It does not install
Nerfstudio, tiny-cuda-nn, COLMAP, or the full `ns_reni` package.

Download the `minimal` release group from the repository root:

```bash
python scripts/download_models.py model-storage/reni --group minimal
```

Then render a deterministic random sample on CPU:

```bash
cd model-storage/reni/minimal
uv run render.py --weights decoder.pt --output-dir render
```

The release directory is self-contained: it includes the decoder artifact,
reference implementation, rendering example, Apache 2.0 licence and locked
CPU environment. From the source-tree copy of this example, pass the
downloaded `model-storage/reni/minimal/decoder.pt` path explicitly.

The command writes:

- `environment_linear_hdr.pt`, a `[H, 2H, 3]` linear-HDR tensor;
- `environment_preview.png`, a display-tonemapped preview;
- `latent.pt`, the `[100, 3]` latent used for the render.

The output is in the fixed luminance gauge used to train the two-bracket
model. Apply a scalar exposure in linear HDR when absolute scene exposure is
required.

`reni_decoder.py` is a release-specific reference implementation. It includes
the learned joint Vector Neuron frame, its Gram-Schmidt orthonormalisation, the
directional encoding, attention decoder, and two-bracket HDR reconstruction.
The full Nerfstudio checkpoint remains the source for continued training and
for reproducing analyses of the learned training latents.
