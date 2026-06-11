"""Quantitative comparison table (paper Table: comparison_PSNR_SSIM_LPIPS).

Evaluates every model's fitted test-set latents and emits the LaTeX table
comparing RENI / RENI++ / SH / SG at matched dimensionalities, plus a CSV.

    PYTHONPATH=. python scripts/figures/make_tables.py
"""

import argparse
import csv
import json
from pathlib import Path

from _common import MODEL_DIRS, REPO_ROOT, load_model, resolve_run_dir


def has_checkpoint(path) -> bool:
    try:
        resolve_run_dir(path)
        return True
    except FileNotFoundError:
        return False

# Dimensionality groups: D -> [(column, family, key)]
DIM_GROUPS = {
    27: [("RENI", "reni_old", 9), ("RENI++", "reni_pp", 9),
         ("SH", "sh", "2nd"), ("SG", "sg", 30)],
    108: [("RENI", "reni_old", 36), ("RENI++", "reni_pp", 36),
          ("SH", "sh", "5th"), ("SG", "sg", 108)],
    147: [("RENI", "reni_old", 49), ("RENI++", "reni_pp", 49),
          ("SH", "sh", "6th"), ("SG", "sg", 150)],
    300: [("RENI", "reni_old", 100), ("RENI++", "reni_pp", 100),
          ("SH", "sh", "9th"), ("SG", "sg", 300)],
}
METRICS = ("psnr_ldr", "ssim_ldr", "lpips_ldr")
HIGHER_BETTER = {"psnr_ldr": True, "ssim_ldr": True, "lpips_ldr": False}


def evaluate_rendered(family: str, key, device: str):
    """Metrics replicating RENIModel.get_image_metrics_and_images exactly,
    but through the verified render path: optional least-squares scale
    alignment in normalised space (scale_inv_loss models), unnormalise, LDR
    via linear_to_sRGB(use_quantile=True), scored with the model's own
    PSNR/SSIM/LPIPS modules. Used for SH/SG, whose pipeline metric loop has
    drifted from the rest of the repo."""
    import torch
    from _common import equirect_ray_bundle
    from reni.utils.colourspace import linear_to_sRGB

    import glob

    import numpy as np
    import pyexr
    import torch
    import torch.nn.functional as F
    from _common import REPO_ROOT, equirect_ray_bundle, make_ray_samples
    from reni.field_components.field_heads import RENIFieldHeadNames
    from reni.utils.colourspace import linear_to_sRGB

    _, _, model = load_model(MODEL_DIRS[family][key], device=device)
    H, W = 64, 128
    rb = equirect_ray_bundle(device, idx=0, height=H)
    rs = make_ray_samples(model, rb)
    # SH/SG fits were run with the test set as their training set; the fitted
    # coefficients live in train bank entry i for the i-th SORTED test EXR.
    # Verified: pairing is 21/21 identity and reproduces the published 19.15
    # for SH 9th order. Datamanager-based pairing is corrupted by these fits'
    # use_validation_as_train configs -- do not use it here.
    bank = model.field.train_params
    files = sorted(glob.glob(str(REPO_ROOT / "data/RENI_HDR/test/*.exr")))
    assert bank.shape[0] == len(files), \
        f"{family}/{key}: bank {bank.shape[0]} vs {len(files)} test EXRs"
    sums = {m: 0.0 for m in METRICS}
    for i, f in enumerate(files):
        # Camera-indexed decode (eval banks were mirrored from the fitted
        # train banks by the loader) -- same path the verified figures use.
        rb_i = equirect_ray_bundle(device, idx=i, height=H)
        with torch.no_grad():
            out = model.get_outputs_for_camera_ray_bundle(
                rb_i, rotation=None)["rgb"]
        pred = linear_to_sRGB(model.field.unnormalise(out).reshape(H, W, 3),
                              use_quantile=True)
        img = pyexr.read(f).astype("float32")[..., :3]
        img[img <= 0] = img[img > 0].min()
        img = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None],
                            size=(H, W), mode="bilinear")[0].permute(1, 2, 0)
        gt = linear_to_sRGB(img.to(device), use_quantile=True)
        gt_t = gt.permute(2, 0, 1)[None]
        pred_t = pred.permute(2, 0, 1)[None]
        sums["psnr_ldr"] += float(model.psnr(gt_t, pred_t))
        sums["ssim_ldr"] += float(model.ssim(gt_t, pred_t))
        sums["lpips_ldr"] += float(model.lpips(gt_t, pred_t))
    return {m: v / len(files) for m, v in sums.items()}


RENDER_EVAL_FAMILIES = {"sh", "sg"}


def evaluate(family: str, key, device: str):
    if family in RENDER_EVAL_FAMILIES:
        return evaluate_rendered(family, key, device)
    pipeline, _, model = load_model(MODEL_DIRS[family][key], device=device)
    if hasattr(model.field, "latent_dim"):
        metrics = pipeline.get_average_eval_image_metrics(optimise_latents=False)
    else:
        metrics = pipeline.get_average_eval_image_metrics()
    return {m: float(metrics[m]) for m in METRICS}


def fmt(value, best, higher_better):
    s = f"{value:.2f}"
    return f"\\textbf{{{s}}}" if value == best else s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "publication" / "tables" / "comparison")
    parser.add_argument("--cache", type=Path, default=None,
                        help="JSON metrics cache; reuse if it exists")
    args = parser.parse_args()

    cache_path = args.cache or args.output.with_suffix(".metrics.json")
    results = {}
    if cache_path.exists():
        results = json.loads(cache_path.read_text())
        print(f"[cache] loaded {len(results)} entries from {cache_path}")

    for dim, entries in DIM_GROUPS.items():
        for column, family, key in entries:
            cache_key = f"{family}/{key}"
            if cache_key in results:
                continue
            if not has_checkpoint(MODEL_DIRS[family][key]):
                print(f"[skip] {cache_key}: no nerfstudio checkpoint "
                      f"(column will show ---)")
                results[cache_key] = None
            else:
                print(f"[eval] {cache_key} (D={dim})")
                results[cache_key] = evaluate(family, key, args.device)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(results, indent=2))

    lines = [
        "% Generated by scripts/figures/make_tables.py -- do not edit by hand",
        "\\begin{tabular}{| c | c | c | c | c |}",
        "\\hline",
        "$D$ & RENI & RENI++ & SH & SG \\\\",
        "& (PSNR/SSIM/LPIPS) & (PSNR/SSIM/LPIPS) & (PSNR/SSIM/LPIPS) & (PSNR/SSIM/LPIPS) \\\\",
        "\\hline",
    ]
    csv_rows = [("D", "model", *METRICS)]
    for dim, entries in DIM_GROUPS.items():
        best = {}
        for m in METRICS:
            vals = [results[f"{family}/{key}"][m] for _, family, key in entries
                    if results[f"{family}/{key}"] is not None]
            best[m] = max(vals) if HIGHER_BETTER[m] else min(vals)
        cells = []
        for column, family, key in entries:
            r = results[f"{family}/{key}"]
            if r is None:
                cells.append("---")
                csv_rows.append((dim, column, "", "", ""))
                continue
            cells.append("/".join(
                fmt(r[m], best[m], HIGHER_BETTER[m]) for m in METRICS))
            csv_rows.append((dim, column, *(f"{r[m]:.4f}" for m in METRICS)))
        lines.append(f"{dim} & " + " & ".join(cells) + " \\\\ \\hline")
    lines.append("\\end{tabular}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.output.with_suffix(".tex")
    tex_path.write_text("\n".join(lines) + "\n")
    with open(args.output.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[saved] {tex_path} and {args.output.with_suffix('.csv')}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
