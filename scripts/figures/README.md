# RENI++ programmatic figures

Headless, deterministic, inference-driven regeneration of the RENI++ paper /
thesis figures (replaces `publication/figures_and_tables.ipynb`). Same
philosophy as `360anything/scripts/sphere_jepa/figures/`.

## Environment

Host conda env `reni++` (torch 2.1.2 + nerfstudio + this repo's `reni`
package via PYTHONPATH). All commands run from the repo root:

```bash
PY=/home/james/miniconda3/envs/reni++/bin/python
PYTHONPATH=. $PY scripts/figures/fig_comparison.py        # vs SH/SG grid
PYTHONPATH=. $PY scripts/figures/fig_interpolations.py    # latent lerps
PYTHONPATH=. $PY scripts/figures/fig_outpainting.py       # masked completion
PYTHONPATH=. $PY scripts/figures/fig_mirror.py            # O(3) negations
PYTHONPATH=. $PY scripts/figures/fig_teaser.py            # teaser overlay
PYTHONPATH=. $PY scripts/figures/fig_old_vs_new.py        # needs old ckpts!
PYTHONPATH=. $PY scripts/figures/fig_inverse_rendering.py # bunny/teapot inverse
PYTHONPATH=. $PY scripts/figures/make_tables.py           # PSNR/SSIM/LPIPS .tex
```

Outputs default to `publication/figures/<name>.{png,pdf}` and
`publication/tables/comparison.{tex,csv}` (`--output` overrides; `--svg`
adds SVG).

## Inputs

- Checkpoints: the COMPLETE paper model archive (single source of truth).
  Download: `python scripts/download_models.py` (the canonical Dropbox zip,
  ~1.1 GB) -> `checkpoints/paper_models/`. `_common.PAPER_MODELS` resolves
  `$RENI_PAPER_MODELS` -> `~/model-storage/reni_paper_models` ->
  `checkpoints/paper_models`. Contains reni_plus_plus_models (+ masked),
  old_reni_models, spherical_harmonics, spherical_gaussians, ablations,
  inverse_task.
- `checkpoints/` keeps only what the archive lacks: SOLD_Net and
  InverseRenderNet baseline checkpoints, and the historical (unusable)
  `reni_original/` wandb dumps.
- Data: `data/RENI_HDR/` (test split = 21 envmaps the fits index into;
  `3d_models/` bunny+teapot for the inverse figure).

## Checkpoint quirks handled by `_common.load_model`

These committed checkpoints are a heterogeneous mix of full training runs
and test-set fits; the loader normalises over the differences:

- nested run dirs (`<variant>/RENI_HDR/<method>/<timestamp>/`) are resolved
  to the latest run;
- the eval split is chosen by matching the checkpoint's per-image bank size
  to the dataset folders (test=21 / val=10), not the saved `test_mode`;
- per-image parameter banks are grafted at their checkpoint sizes when they
  disagree with the rebuilt datamanager's split sizes;
- SH/SG fits store fitted coefficients in the *train* banks (the fit used
  the test set as its training set); they are mirrored into the eval banks
  when the eval bank is clearly unfitted;
- container-absolute `/workspace/...` paths in saved configs are remapped.

## Outpainting mask modes

`fig_outpainting.py` defaults to `--mask_mode perspective`: each row masks the
GT envmap to a perspective camera footprint on the sphere (different viewing
direction per row, configurable `--hfov/--vfov/--fit_steps`) and refits a
latent on the visible pixels with the frozen D=100 decoder. This visually
matches SphereJEPA's crop-to-sphere completion setting. `--mask_mode dataset`
reproduces the original square-mask paper figure via the masked-fit
checkpoint.

## Original-RENI checkpoints (resolved 2026-06-10)

The full paper model archive lives at
`/home/james/model-storage/reni_paper_models/` (old_reni_models,
reni_plus_plus_models, ablations, inverse_task, spherical_harmonics,
spherical_gaussians). `checkpoints/old_reni_models` is a symlink into it and
is what `MODEL_DIRS["reni_old"]` uses; `fig_old_vs_new.py` and the table's
RENI column work from these.

CORRECTION: `checkpoints/reni_original/` (original-repo wandb dumps) ARE the
published original-RENI weights — `publication/generate_figures.py`
(`_load_old_format`) loads them correctly and reproduces the published
table values. The `old_implementation` path in the modern field (and the
naive manual decode) does NOT reproduce them — use the generate_figures
loader for old-format checkpoints. `convert_original_reni.py` is retained
but should be ported to that loader's conventions before trusting it.

All checkpoint paths resolve through `reni/utils/checkpoint_locator.py`
(repo extras → paper archive → baselines archive; `$RENI_CHECKPOINT_ROOTS`
override). No symlinks.
