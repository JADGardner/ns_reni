"""Diagnostics for the peak-angle gap between conditioning variants.

Two questions from the 2026-07-15 invariant-function study:
1. Are the peak-angle differences (Norms 5.5/8.8 deg vs ortho-VNJoint
   6.8/13.6 deg) driven by outlier images or systematic offsets?
   -> per-image peak errors from a full-image refit.
2. Is the ortho-joint model's fit objective flatter under a global azimuth
   rotation of the fitted latent (the "richer conditioning buys rotational
   slack" hypothesis)? -> refit each test latent, rotate it +-2/5/10 deg
   about the gravity axis, measure the fit-loss increase.

Run from the phd repo root:

    docker compose run --rm -w /workspace/phd/code/ns_reni research bash -c \
        "PYTHONPATH=.:scripts/figures python scripts/figures/diag_peak_rotation.py"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from _common import REPO_ROOT, equirect_ray_bundle, seed_all
from eval_latent_reset_compare import (
    _build_test_config,
    _latest_checkpoint,
    _load_decoder_state,
    resolve_run_dir,
)
from eval_outpaint_compare import (
    EPS,
    decode_full_envmap,
    fit_latent_on_visible,
)
from reni.utils.colourspace import linear_to_sRGB  # noqa: F401  (parity import)
from reni.utils.hdr_metrics import compute_hdr_peak_metrics

RUNS = {
    "perchannel_2cyc": "/workspace/phd/outputs/reni/reni_latent_reset_d100_two_bracket_ldrw3_2cyc",
    "norms_2cyc": "outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_norms",
    "vnjoint_ortho_2cyc": "outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho",
}
DATA_ROOT = REPO_ROOT / "data" / "RENI_HDR"
DEVICE = "cuda:0"
DELTAS = (2.0, 5.0, 10.0)
OUT = REPO_ROOT / "outputs" / "evaluations" / "diag_peak_rotation.json"


def rot_y(deg: float, device) -> torch.Tensor:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                        device=device)


def full_image_loss(model, z, ray_bundle, gt_raw) -> float:
    """The fit objective (log-domain mse + cosine) over ALL pixels."""
    channels = gt_raw.shape[-1]
    height, width = gt_raw.shape[0], gt_raw.shape[1]
    pred_raw = decode_full_envmap(model, z, ray_bundle, height, width, channels)
    pred_lin = model._to_linear_hdr(pred_raw).reshape(-1, 3)
    gt_lin = model._to_linear_hdr(gt_raw).reshape(-1, 3)
    pred_log = torch.log(pred_lin.clamp_min(0.0) + EPS)
    gt_log = torch.log(gt_lin.clamp_min(0.0) + EPS)
    mse = torch.nn.functional.mse_loss(pred_log, gt_log)
    cosine = 1 - torch.nn.functional.cosine_similarity(pred_log, gt_log, dim=-1).mean()
    return float(10.0 * mse + cosine)


def main() -> None:
    results = {}
    for label, run in RUNS.items():
        seed_all(42)
        run_dir = resolve_run_dir(Path(run))
        checkpoint = _latest_checkpoint(run_dir)
        model_config = _build_test_config(run_dir, DATA_ROOT, None)
        pipeline = model_config.pipeline.setup(
            device=DEVICE, test_mode="test", world_size=1, local_rank=0,
            grad_scaler=None)
        _load_decoder_state(pipeline, checkpoint, DEVICE)
        model = pipeline.model
        model.to(DEVICE)
        model.eval()
        num_eval = len(pipeline.datamanager.eval_dataset)
        print(f"[{label}] {checkpoint.name}, {num_eval} eval images")

        per_image = []
        for idx in range(num_eval):
            batch = pipeline.datamanager.eval_dataset[idx]
            gt_raw = batch["image"]
            if gt_raw.dim() == 4:
                gt_raw = gt_raw[0]
            gt_raw = gt_raw.to(DEVICE)
            height, width, channels = gt_raw.shape
            mask = torch.ones(height, width)
            ray_bundle = equirect_ray_bundle(DEVICE, idx=0, height=height)

            z = fit_latent_on_visible(model, gt_raw, mask, ray_bundle, DEVICE,
                                      prior_weight=0.0)
            pred_raw = decode_full_envmap(model, z, ray_bundle, height, width,
                                          channels)
            gt_lin = model._to_linear_hdr(gt_raw)
            pred_lin = model._to_linear_hdr(pred_raw)
            peak = compute_hdr_peak_metrics(pred_lin, gt_lin, alignment="none")

            base_loss = full_image_loss(model, z, ray_bundle, gt_raw)
            dloss = {}
            for d in DELTAS:
                vals = []
                for s in (+1.0, -1.0):
                    zr = z @ rot_y(s * d, DEVICE).T
                    vals.append(full_image_loss(model, zr, ray_bundle, gt_raw)
                                - base_loss)
                dloss[f"{d:g}"] = sum(vals) / 2.0
            rec = {
                "idx": idx,
                "peak_angle": float(peak["peak_angle_error_deg"]),
                "peak_argmax": float(peak["peak_argmax_angle_error_deg"]),
                "base_loss": base_loss,
                "dloss": dloss,
            }
            per_image.append(rec)
            print(f"  [{label}] idx={idx:02d} peak {rec['peak_angle']:7.2f} "
                  f"argmax {rec['peak_argmax']:7.2f}  "
                  f"dloss@2/5/10: {dloss['2']:.4f}/{dloss['5']:.4f}/{dloss['10']:.4f}")

        angles = torch.tensor([r["peak_angle"] for r in per_image])
        argmaxs = torch.tensor([r["peak_argmax"] for r in per_image])
        results[label] = {
            "per_image": per_image,
            "peak_angle": {"mean": angles.mean().item(),
                           "median": angles.median().item(),
                           "max": angles.max().item()},
            "peak_argmax": {"mean": argmaxs.mean().item(),
                            "median": argmaxs.median().item(),
                            "max": argmaxs.max().item()},
            "dloss_mean": {d: sum(r["dloss"][d] for r in per_image) / len(per_image)
                           for d in ("2", "5", "10")},
        }
        del pipeline, model
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n[saved] {OUT}\n")
    print(f"{'model':>20} {'peak mean/med/max':>24} {'argmax mean/med/max':>24} "
          f"{'dloss@2':>9} {'dloss@5':>9} {'dloss@10':>9}")
    for label, r in results.items():
        pa, pm = r["peak_angle"], r["peak_argmax"]
        dl = r["dloss_mean"]
        print(f"{label:>20} {pa['mean']:7.2f}/{pa['median']:6.2f}/{pa['max']:7.2f} "
              f"  {pm['mean']:7.2f}/{pm['median']:6.2f}/{pm['max']:7.2f} "
              f"{dl['2']:9.4f} {dl['5']:9.4f} {dl['10']:9.4f}")


if __name__ == "__main__":
    main()
