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
PYTHONPATH=. $PY scripts/figures/fig_model_overview.py    # model diagram
PYTHONPATH=. $PY scripts/figures/fig_teaser.py            # teaser overlay
PYTHONPATH=. $PY scripts/figures/fig_old_vs_new.py        # needs old ckpts!
PYTHONPATH=. $PY scripts/figures/fig_inverse_rendering.py # bunny/teapot inverse
PYTHONPATH=. $PY scripts/figures/fig_car_relighting.py    # Blender car relighting
PYTHONPATH=. $PY scripts/figures/fig_latent_reset_tsne.py # latent-reset t-SNE
PYTHONPATH=. $PY scripts/figures/fig_baseline_comparison.py  # vs SOLD-Net + Hosek-Wilkie (+ Table 2.2)
PYTHONPATH=. $PY scripts/figures/fig_yi_comparison.py        # vs Yi et al. (SH) inverse rendering
PYTHONPATH=. $PY scripts/figures/fig_inversenet_comparison.py # vs InverseRenderNet outpainting
PYTHONPATH=. $PY scripts/figures/make_tables.py           # PSNR/SSIM/LPIPS .tex
```

The three external-baseline comparisons (`fig_baseline_comparison`,
`fig_yi_comparison`, `fig_inversenet_comparison`) reuse the reviewed evaluation
logic in `publication/{baseline,yi_et_al_vs_reni,inversenet_vs_reni}_comparison.py`
and re-render in this pattern (headless, seeded, `--labels` bakes the thesis
TikZ column headers, outputs to `publication/figures/`, and each writes a
`<name>_metrics.tex` alongside). Extra inputs beyond the paper archive:
`checkpoints/SOLD_Net/pretrained_model/`, `checkpoints/inverserendernet/model_ckpt.pth`,
and `thirdparty/Yi_et_al_relighting/InverseRendering/path/invrender.pth`.
`fig_baseline_comparison`/`fig_yi_comparison` read `data/RENI_HDR/{test,val}`;
`fig_inversenet_comparison` reads `data/RENI_HDR/test`.

In the Docker `research` container, `~/model-storage` is not on `$HOME`, so pass
`-e RENI_PAPER_MODELS=/home/james/model-storage/reni_paper_models` so the RENI++
checkpoint resolves, e.g.:

```bash
docker compose run --rm -e RENI_PAPER_MODELS=/home/james/model-storage/reni_paper_models \
  research bash -c "cd /workspace/phd/code/ns_reni && \
  PYTHONPATH=. python scripts/figures/fig_baseline_comparison.py --device cuda:0"
```

Outputs default to `publication/figures/<name>.{png,pdf}` and
`publication/tables/comparison.{tex,csv}` (`--output` overrides; `--svg`
adds SVG).

Text baked into regenerated figures uses a Times-compatible serif stack
(`Nimbus Roman`, `Times New Roman`, `Liberation Serif`, `STIXGeneral`) to match
the thesis TikZ overlays, which inherit the thesis `times`/`mathptmx` font.
Use `--height 512 --decode_chunk 32768` for the latent-only thesis figures
when the rendered envmaps should use the same 512x1024 source resolution as the
high-resolution RENI_HDR mapping.

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
  `3d_models/` bunny+teapot for the inverse figure, plus the optional
  `frazer_nash_super_sport_1929.blend` Blender asset for the car relighting
  figure).

### High-resolution RENI_HDR test images

The local anonymous `data/RENI_HDR/test/*.exr` split has been matched to the
provider-organised `data/RENI_HDR_512x1024/` archive. Use the confirmed
test-only mapping from the PhD repo root:

```bash
RENI_FIGURE_EVAL_MAPPING=../../artifacts/reni_hdr_test_highres_mapping.csv \
RENI_FIGURE_HIGH_ROOT=../../data/RENI_HDR_512x1024 \
RENI_FIGURE_EVAL_WIDTH=1024 \
PYTHONPATH=. $PY scripts/figures/fig_comparison.py
```

`RENI_FIGURE_EVAL_WIDTH` defaults to the mapped source width if unset; lowering
it is useful for CPU smoke tests. The validation split does not have confirmed
matches in `RENI_HDR_512x1024`; its nearest neighbours appear to be unrelated
newer Polyhaven/HDRI images, so do not use the validation candidates for paper
figures.

Some paper checkpoints rebuild the 21-image eval dataset with a stale
`split='train'` attribute even though the filenames are the test split.
`_common.eval_image_tensor` therefore falls back to the confirmed test mapping
by filename when the dataset split key does not match.

High-resolution labelled thesis regeneration:

```bash
RENI_FIGURE_EVAL_MAPPING=../../artifacts/reni_hdr_test_highres_mapping.csv \
RENI_FIGURE_HIGH_ROOT=../../data/RENI_HDR_512x1024 \
RENI_FIGURE_EVAL_WIDTH=1024 \
PYTHONPATH=. $PY scripts/figures/fig_comparison.py \
  --device cuda:0 --labels --output publication/figures/comparison_labeled

RENI_FIGURE_EVAL_MAPPING=../../artifacts/reni_hdr_test_highres_mapping.csv \
RENI_FIGURE_HIGH_ROOT=../../data/RENI_HDR_512x1024 \
RENI_FIGURE_EVAL_WIDTH=1024 \
PYTHONPATH=. $PY scripts/figures/fig_old_vs_new.py \
  --device cuda:0 --labels --output publication/figures/old_vs_new_labeled

PYTHONPATH=. $PY scripts/figures/fig_interpolations.py \
  --device cuda:0 --height 512 --decode_chunk 32768 --labels \
  --model_dir outputs/reni_latent_reset_4_rerun/reni/2026-07-01_4cycles_rerun \
  --random_source train_mu \
  --idx1 2 --idx2 8 \
  --output publication/figures/interpolations_and_random_samples_labeled

PYTHONPATH=. $PY scripts/figures/fig_mirror.py \
  --device cuda:0 --height 512 --decode_chunk 32768 --labels \
  --output publication/figures/mirror_labeled

PYTHONPATH=. $PY scripts/figures/fig_teaser.py \
  --device cuda:0 --height 512 --decode_chunk 32768 --labels \
  --output publication/figures/teaser_labeled

PYTHONPATH=. $PY scripts/figures/fig_latent_reset_tsne.py \
  --output publication/figures/latent_reset_train_latents_tsne

PYTHONPATH=. $PY scripts/figures/fig_model_overview.py \
  --envmap ../../data/RENI_HDR_512x1024/openfootage_hdri/00266_OpenfootageNET_Pinzgau_LOW.exr \
  --output publication/figures/model_overview

RENI_FIGURE_EVAL_MAPPING=../../artifacts/reni_hdr_test_highres_mapping.csv \
RENI_FIGURE_HIGH_ROOT=../../data/RENI_HDR_512x1024 \
PYTHONPATH=. $PY scripts/figures/fig_inverse_rendering.py \
  --device cuda:0 --labels --envmap_width 1024 --envmap_chunk 32768 \
  --output publication/figures/inverse_rendering_labeled
```

The inverse-rendering object's precomputed target RGB renders are 128x128 in
`data/RENI_HDR/3d_models/image/`; the figure keeps those at native resolution.
The fitted/GT environment maps in that figure use the high-resolution
512x1024 HDRI mapping and 1024-wide predicted envmap renders.

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
GT envmap to a perspective camera footprint on the sphere and refits a latent
on the visible pixels with the frozen D=100 decoder. The default perspective
footprint is 110 x 80 degrees. Row viewpoints are fixed as roll, pitch, yaw
triples in the script, matching the thesis figure; override them with
`--row_rpy ROW:ROLL,PITCH,YAW` to nudge a camera. `--mask_mode dataset`
reproduces the original square-mask paper figure via the masked-fit checkpoint.

High-resolution thesis outpainting preview:

```bash
RENI_FIGURE_EVAL_MAPPING=../../artifacts/reni_hdr_test_highres_mapping.csv \
RENI_FIGURE_HIGH_ROOT=../../data/RENI_HDR_512x1024 \
RENI_FIGURE_EVAL_WIDTH=1024 \
PYTHONPATH=. $PY scripts/figures/fig_outpainting.py \
  --device cuda:0 --labels --fit_height 0 --fit_rays_per_step 32768 --dpi 400 \
  --output publication/figures/outpainting_labeled
```

`--fit_height` controls the optimisation resolution for the fresh latent fit;
the figure is still rendered at the mapped image resolution. Use
`--fit_height 0` for native/full-resolution fitting. Use a smaller value such
as `128` only for smoke tests. `--fit_rays_per_step` samples visible rays from
the native-resolution mask each step; use `0` for all visible rays if GPU memory
allows it.

Default row poses (`roll,pitch,yaw`, degrees):

```text
1:  +3.820935, -6.460099,+175.683607
2: -30.532792,+21.338249, +68.085577
3: +22.934182, +7.193256, -10.635712
4:  +9.216508,-13.638064, -10.635712
5: +18.066142, -2.829290, +68.085577
6: -10.183182,-13.638064, -93.300564
```

For example, to raise row 4 slightly while preserving its roll/yaw:

```bash
--row_rpy 4:9.216508,-8.0,-10.635712
```

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
