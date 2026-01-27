# TPAMI Revision Plan: RENI++

> **Status**: Major Revision Required  
> **Priority**: Address ALL reviewer concerns to avoid rejection

---

## Executive Summary

The reviewers require **quantitative comparisons against non-SH/SG methods** as the primary concern. This document outlines the implementation tasks required within the ns_reni codebase to address each reviewer point.

---

## I. The "Dealbreaker": Comparative Evaluation

> **Priority: CRITICAL** — Both R3 and R4 cite lack of comparison as the primary weakness.

### 1.1 Grid Sampling Strategy for Quantitative Comparison

**Goal**: Sample RENI++ continuous representation into discrete environment maps for PSNR/SSIM comparison.

#### Implementation Tasks:

- [ ] **Create `reni/evaluation/grid_sampler.py`**
  - Function: `sample_to_envmap(reni_field, latent_code, rotation, resolution=(512, 1024)) -> Tensor`
  - Generate uniform spherical sampling grid (equirectangular projection)
  - Query RENI field at each direction to produce environment map
  - Leverage existing `EnvironmentMapField.cart_to_spherical()` and `angles_to_map_coords()` patterns

- [ ] **Create `reni/evaluation/metrics.py`**
  - Implement `compute_psnr(pred_envmap, gt_envmap) -> float`
  - Implement `compute_ssim(pred_envmap, gt_envmap) -> float`
  - Implement `compute_rmse(pred_envmap, gt_envmap) -> float`
  - Consider log-domain metrics for HDR (si-RMSE, si-PSNR)

- [ ] **Create `reni/evaluation/comparison_pipeline.py`**
  - Unified evaluation script that:
    1. Loads GT environment maps from dataset
    2. Samples RENI++ to environment map
    3. Samples baseline methods to environment maps
    4. Computes metrics across all methods
    5. Exports results to CSV/LaTeX table

### 1.2 Baseline Method Implementations

#### Analytical Models:

- [ ] **Implement Hosek-Wilkie Sky Model** `reni/baselines/hosek_wilkie.py`
  - Reference: [1] Hosek & Wilkie, "An Analytic Model for Full Spectral Sky-Dome Radiance"
  - Parameters: sun_direction, turbidity, ground_albedo
  - Output: Environment map at specified resolution
  - Note: Well-defined analytical formula, straightforward to implement

- [ ] **Implement Lalonde-Matthews Model** `reni/baselines/lalonde_matthews.py`
  - Reference: [2] Lalonde et al., "What do the sun and the sky tell us..."
  - Physically-based outdoor lighting model
  - May require fitting to parameters from images

#### Neural Models:

- [ ] **Integrate SkyNet** `reni/baselines/skynet.py`
  - Reference: [3] SkyNet paper
  - Check for available pretrained weights
  - Wrapper to produce environment maps comparable to RENI++
  - If code unavailable: Document in paper with detailed explanation

- [ ] **Integrate SOLD-Net** `reni/baselines/soldnet.py`
  - Reference: [4] SOLD-Net (Reviewer 4 specifically requests this)
  - Sky and outdoor lighting decomposition
  - Prioritize this comparison as explicitly requested

#### Latent/SH Network Models (from Reviewer 3):

- [ ] **Integrate Liang et al. (ECCV 24)** `reni/baselines/liang_eccv24.py`
  - Check for public code/weights
  - If unavailable: best-effort comparison using paper metrics

- [ ] **Integrate Yu & Smith (CVPR 19)** `reni/baselines/yu_smith_cvpr19.py`
  - Check for public code/weights

- [ ] **Integrate Yi et al. (CVPR 23)** `reni/baselines/yi_cvpr23.py`
  - Check for public code/weights

### 1.3 Comparison Infrastructure

- [ ] **Create `scripts/run_comparison_benchmark.py`**
  - Command-line script to run full comparison
  - Arguments: dataset_path, methods, output_dir, resolution
  - Outputs: metrics table, per-sample breakdowns, visualizations

- [ ] **Create `reni/evaluation/visualization.py`**
  - Side-by-side environment map comparisons
  - Difference maps (error visualization)
  - Per-frequency analysis visualizations

### 1.4 Fallback: Detailed Technical Justification

For methods where comparison is mathematically/practically impossible:

- [ ] **Create `publication/comparison_justification.md`**
  - Document why specific comparisons cannot be performed
  - Must include:
    - Architectural differences preventing direct comparison
    - Different input/output domains
    - Unavailable code/weights despite contacting authors
  - Phrase diplomatically: "While X is challenging because of Y, we have performed Z..."

---

## II. Positioning & Claims

> **Priority: HIGH** — R3 contesting "First natural prior" claim

### 2.1 Revise "First" Claim

- [ ] **Update Related Work section**
  - Narrow claim to: "First **rotation-equivariant** natural illumination prior"
  - OR remove "first" entirely
  - Must cite and differentiate from:
    - Liang et al. (ECCV 24)
    - Yu & Smith (CVPR 19)
    - Yi et al. (CVPR 23)

- [ ] **Create differentiation table** in Related Work
  - Columns: Method, Rotation Equivariance, Scale Invariance, Continuous vs Discrete, Learning-based Prior
  - Clearly show RENI++ advantages

### 2.2 High-Resolution Reflection Evidence

- [ ] **Generate zoomed reflection comparisons**
  - Use existing RENI model to render specular spheres
  - Create figure with:
    - Zoomed crops of high-frequency reflection regions
    - Side-by-side: GT, RENI++, SH, SG
  - Add to supplementary materials

- [ ] **Compute specular-region metrics**
  - Segment high-frequency/specular regions in GT
  - Compute per-region PSNR/SSIM
  - Document in paper to support "close to GT" claim

---

## III. Technical & Performance Metrics

> **Priority: HIGH** — R3 specifically requested consumer hardware benchmarks

### 3.1 Runtime Benchmarks on Consumer Hardware

- [ ] **Create `scripts/benchmark_runtime.py`**
  - Benchmark inference speed (images/second)
  - Benchmark memory usage (peak GPU MB)
  - Test on multiple GPU configurations:
    - RTX 3080 (consumer target)
    - RTX 3060 (accessible tier)
    - Current high-end GPU (for comparison)

- [ ] **Benchmark parameters to report**:
  - Latent code optimization time
  - Environment map generation time at various resolutions
  - Memory footprint vs SH/SG representations

- [ ] **Add runtime comparison table to paper**
  - Method | Inference Time | Memory | GPU Tier

---

## IV. Editorial Corrections (Reviewer 4)

> **Priority: MEDIUM** — Quick fixes, but show attention to detail

### 4.1 Typos to Fix

- [ ] Figure 4 caption: Replace incorrect value with correct one
- [ ] Add hyphens: "image-to-sky estimation"
- [ ] LaTeX formatting: "80$\times$" for dimension notation
- [ ] Replace "e.g." with "i.e." where appropriate (logical correction)

### 4.2 Citation Fixes

- [ ] Add missing authors for Reference [32]
- [ ] Update Reference [37] (EverLight) to ICCV 2023 version

### 4.3 Formatting

- [ ] Ensure Figures/Tables appear near first textual reference
- [ ] Double-column format for main PDF
- [ ] Create annotated version with tracked changes

---

## V. Response Letter Quality

> **Priority: HIGH** — R3 was annoyed by previous response quality

### 5.1 Response Letter Structure

- [ ] **Create structured response template**
  - Format: Q1, A1, Q2, A2 numbering
  - Never dismissive phrasing
  - Template phrase: "We thank the reviewer for this suggestion. While X is challenging because of Y, we have performed Z to approximate this comparison."

### 5.2 Proofreading Checklist

- [ ] No unfinished sentences
- [ ] No typos
- [ ] All reviewer points explicitly addressed
- [ ] Page/line references for all changes

---

## VI. Administrative Requirements

- [ ] Author biographies and photos (required for major revision)
- [ ] Clean double-column main PDF
- [ ] Summary of Changes document with tracked edits

---

## Implementation Priority Order

1. **Week 1-2**: Grid sampling infrastructure + metrics (I.1, I.2)
2. **Week 2-3**: Baseline implementations - prioritize SOLD-Net, Hosek-Wilkie (I.2)
3. **Week 3**: Runtime benchmarks (III.1)
4. **Week 3-4**: Related work revision + reflection evidence (II.1, II.2)
5. **Week 4**: Editorial fixes + response letter (IV, V)
6. **Final**: Administrative requirements (VI)

---

## Existing Publication Assets

The following assets already exist in `publication/` and should be **extended** rather than recreated:

### Notebooks
| File | Purpose | Extend For |
|------|---------|------------|
| `figures_and_tables.ipynb` | Main figure/table generation | Add comparison tables, new baseline results |
| `inverse_task.ipynb` | Inverse rendering evaluation | Add runtime benchmarks, specular metrics |
| `non_convexity.ipynb` | Analysis notebook | May contain useful evaluation code |
| `animations.ipynb` | Teaser/animations | N/A |

### Existing Figures (`publication/figures/`)
| File | Content | Action |
|------|---------|--------|
| `comparison.pdf/png` | Existing comparison figure | **Extend** with new baselines |
| `inverse_rendering.pdf/png` | Inverse rendering results | Add zoomed specular crops |
| `mirror.pdf/png` | Mirror/reflection results | Could support R3's reflection claims |
| `old_vs_new.pdf/png` | RENI vs RENI++ comparison | Keep as-is |

### Recommended Approach
1. Add new evaluation code to existing notebooks where possible
2. Create new standalone scripts for repeatable benchmarks (`scripts/`)
3. Keep baselines modular in `reni/baselines/` for reuse

---

## File Structure for New Code

```
reni/
├── evaluation/
│   ├── __init__.py
│   ├── grid_sampler.py        # RENI to envmap conversion
│   ├── metrics.py             # PSNR, SSIM, RMSE
│   ├── comparison_pipeline.py # Unified evaluation
│   └── visualization.py       # Comparison figures
├── baselines/
│   ├── __init__.py
│   ├── hosek_wilkie.py        # Analytical sky model
│   ├── lalonde_matthews.py    # Physical outdoor model
│   ├── skynet.py              # Neural sky model
│   ├── soldnet.py             # SOLD-Net (priority)
│   ├── liang_eccv24.py        # Latent network
│   ├── yu_smith_cvpr19.py     # SH estimation
│   └── yi_cvpr23.py           # Recent SH method
scripts/
├── run_comparison_benchmark.py
└── benchmark_runtime.py
publication/
├── comparison_justification.md
└── response_letter_template.md
```

---

## References to Obtain

1. **Hosek-Wilkie**: Open source, analytical - [hosek-wilkie.org](http://cgg.mff.cuni.cz/projects/SkylightModelling/)
2. **Lalonde-Matthews**: Check CVPR supplementary materials
3. **SkyNet**: Check GitHub/project page
4. **SOLD-Net**: Priority - contact authors if needed
5. **Liang et al. ECCV24**: Recent - check ECCV supplementary
6. **Yu & Smith CVPR19**: Check project page
7. **Yi et al. CVPR23**: Check CVPR supplementary

---

## Key Success Criteria

✅ Quantitative PSNR/SSIM comparison against at least 3 non-SH/SG methods  
✅ Runtime benchmarks on RTX 3080 or equivalent  
✅ Narrowed/removed "first" claim with explicit differentiation  
✅ Visual evidence for high-frequency reflection quality  
✅ Professional, complete response letter  
✅ All editorial corrections made


# TPAMI Baseline Method Implementation

## Objective
Implement baseline comparison methods for RENI++ paper to address reviewer feedback requiring quantitative comparisons against non-SH/SG methods.

## Tasks

### Research & Availability Check
- [x] Hosek-Wilkie Sky Model - ✅ BSD C source available (v1.4a)
- [x] SOLD-Net (Reviewer 4 priority) - ✅ Code at `ChemJeff/SOLD-Net`, pretrained on Baidu
- [x] SkyNet - ❌ No public code (CVPR 2019)
- [x] Lalonde-Matthews Model - ❌ No public implementation
- [x] Liang et al. ECCV 24 - IntrinsicAnything (different task: inverse rendering)
- [x] Yu & Smith CVPR 19 - InverseRenderNet available (SH prediction, not generative prior)
- [x] Yi et al. CVPR 23 - NEnv EGSR23 available (envmap compression, not generative prior)

### FOV Compatibility (RENI ↔ SOLD-Net)
| Property | RENI | SOLD-Net |
|----------|------|----------|
| Resolution | 64×128 | 32×128 |
| Vertical FOV | 180° (full sphere) | 90° (sky only) |
| Horizontal | 360° | 360° |
| Source | Laval Sky HDR | Laval Sky HDR |

**Conversion**: Crop top half of RENI → matches SOLD-Net exactly!

### Implementation Progress
- [x] Clone SOLD-Net to project root (`/home/james/github/ns_reni/SOLD-Net`)
- [x] Test conda compatibility - ✅ Works with `reni++` env
- [x] FOV compatibility analysis - ✅ Trivial conversion (crop top half)
- [x] Download pretrained models (Password: `i6ef`) - ✅ In `checkpoints/SOLD_Net/`
- [x] Implement Hosek-Wilkie in `reni/baselines/hosek_wilkie.py`
- [x] Implement SOLD-Net wrapper in `reni/baselines/soldnet.py`
- [x] Create comparison script (`publication/baseline_comparison.py`)

### Baseline Comparison Results (21 test images)

**Final Results with all fixes applied:**

| Method | PSNR↑ | SSIM↑ | LogMSE↓ | LDR PSNR↑ |
|--------|-------|-------|---------|-----------|
| **RENI++** | **32.24** | **0.66** | **0.022** | **22.55** |
| Hosek-Wilkie | 28.84 | 0.36 | 0.097 | 11.61 |
| SOLD-Net | 28.03 | 0.43 | 0.260 | 14.40 |

Outputs saved to `publication/figures_baseline/`

### Key Implementation Improvements (Dec 2024)

1. **Data Pipeline Alignment**: Fixed `baseline_comparison.py` to use the RENI pipeline datamanager (matching `generate_figures.py`) for consistent GT/prediction handling and proper normalization/unnormalization.

2. **SOLD-Net Sky-Only Decoder**: Added `sky_only=True` mode to bypass the sun decoder blending which caused visible 8×8 white square artifacts. The sun decoder outputs non-zero values outside the sun region when applied to different data distributions.

3. **Hosek-Wilkie Gradient Descent Fitting**: Replaced naive grid search with scipy L-BFGS-B optimization:
   - Stage 1: Coarse grid search (4×4×3 = 48 combinations) for initial parameters
   - Stage 2: Gradient descent to refine all 5 parameters (sun_elev, sun_azim, turbidity, albedo, intensity)
   - Parameters: `maxiter=5000, eps=1e-2, intensity_bounds=(0.01, 1000)`

4. **Hosek-Wilkie Sky-Only Mode**: Fixed equirectangular theta mapping to correctly handle sky hemisphere (θ: 0→π/2) instead of full sphere (θ: 0→π). This eliminated the solid ground color block in the bottom half and improved SSIM from 0.17 → 0.36.

5. **Consistent HDR Space**: All methods compared in HDR linear space with same `field.unnormalise()` and `linear_to_sRGB()` for visualization.

### Documentation
- [ ] Create `publication/comparison_justification.md` for unavailable methods
- [x] Update generate_figures.py with baseline comparisons → separate script created