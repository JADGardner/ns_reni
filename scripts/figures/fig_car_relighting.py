"""Blender car relighting figure.

Renders the Frazer Nash Blender asset under a ground-truth RENI_HDR test
environment map and the matching fitted RENI++ reconstruction, then composes
the two renders into a paper-style figure.

    PYTHONPATH=. python scripts/figures/fig_car_relighting.py
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyexr
import torch

from _common import (MODEL_DIRS, REPO_ROOT, add_common_args,
                     equirect_ray_bundle, load_model, save_figure, seed_all)


DEFAULT_BLEND = (
    REPO_ROOT / "data" / "RENI_HDR" / "3d_models"
    / "frazer_nash_super_sport_1929.blend"
)
BLENDER_HELPER = REPO_ROOT / "scripts" / "figures" / "blender_render_car.py"


def export_hdr_envmaps(args):
    """Export matched GT and RENI++ HDR envmaps for Blender world lighting."""
    _, datamanager, model = load_model(MODEL_DIRS["reni_pp"][100],
                                       device=args.device)
    batch = datamanager.eval_dataset[args.image_idx]
    gt_norm = batch["image"]
    if gt_norm.dim() == 4:
        gt_norm = gt_norm[0]
    height = gt_norm.shape[0]
    ray_bundle = equirect_ray_bundle(args.device, idx=args.image_idx,
                                     height=height)
    gt_norm = gt_norm.to(args.device)

    with torch.no_grad():
        outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle,
                                                          rotation=None)
        pred = model.field.unnormalise(outputs["rgb"]).reshape(
            gt_norm.shape[0], gt_norm.shape[1], 3)
        gt = model.field.unnormalise(gt_norm).reshape(
            gt_norm.shape[0], gt_norm.shape[1], 3)

    env_dir = args.work_dir / "envmaps"
    env_dir.mkdir(parents=True, exist_ok=True)
    gt_path = env_dir / f"car_gt_idx{args.image_idx:03d}.exr"
    pred_path = env_dir / f"car_reni_pp_idx{args.image_idx:03d}.exr"
    pyexr.write(str(gt_path), gt.detach().cpu().numpy().astype("float32"))
    pyexr.write(str(pred_path), pred.detach().cpu().numpy().astype("float32"))
    return gt_path, pred_path


def run_blender(args, gt_env, pred_env):
    render_dir = args.work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "label": "Ground Truth",
            "env_path": str(gt_env),
            "output_path": str(render_dir / "ground_truth.png"),
        },
        {
            "label": "RENI++",
            "env_path": str(pred_env),
            "output_path": str(render_dir / "reni_plus_plus.png"),
        },
    ]
    jobs_path = args.work_dir / "blender_jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2))

    cmd = [
        args.blender,
        "--background",
        str(args.blend),
        "--python",
        str(BLENDER_HELPER),
        "--",
        "--jobs-json",
        str(jobs_path),
        "--camera",
        args.camera,
        "--width",
        str(args.render_width),
        "--height",
        str(args.render_height),
        "--samples",
        str(args.samples),
        "--world-strength",
        str(args.world_strength),
        "--env-rotation-deg",
        str(args.env_rotation_deg),
        "--device",
        args.blender_device,
    ]
    if args.keep_scene_lights:
        cmd.append("--keep-scene-lights")

    print("[blender]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(jobs[0]["output_path"]), Path(jobs[1]["output_path"])


def compose(args, gt_render, pred_render):
    gt = plt.imread(str(gt_render))
    pred = plt.imread(str(pred_render))

    plt.rc("font", family="serif")
    fig, axes = plt.subplots(2, 1, figsize=(10, 11.3), dpi=100)
    for ax, image, title in zip(axes, (gt, pred), ("Ground Truth", "RENI++")):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02,
                        hspace=0.08)
    save_figure(fig, args.output, svg=args.svg, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "car_relighting")
    parser.add_argument("--image_idx", type=int, default=18,
                        help="RENI_HDR test image index to render")
    parser.add_argument("--blend", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--work_dir", type=Path,
                        default=REPO_ROOT / "outputs" / "figures"
                        / "car_relighting")
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--camera", default="Camera Perspective")
    parser.add_argument("--render_width", type=int, default=1000)
    parser.add_argument("--render_height", type=int, default=562)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--world_strength", type=float, default=3.0)
    parser.add_argument("--env_rotation_deg", type=float, default=0.0)
    parser.add_argument("--blender_device", choices=("CPU", "GPU"),
                        default="CPU")
    parser.add_argument("--keep_scene_lights", action="store_true",
                        help="Keep lights authored in the .blend scene")
    parser.add_argument("--skip_blender", action="store_true",
                        help="Reuse existing rendered PNGs in work_dir")
    args = parser.parse_args()
    seed_all(args.seed)

    if not args.blend.exists():
        raise FileNotFoundError(
            f"Missing blend asset: {args.blend}. Expected the downloaded "
            "Frazer Nash asset under data/RENI_HDR/3d_models/."
        )
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_blender:
        gt_render = args.work_dir / "renders" / "ground_truth.png"
        pred_render = args.work_dir / "renders" / "reni_plus_plus.png"
    else:
        gt_env, pred_env = export_hdr_envmaps(args)
        gt_render, pred_render = run_blender(args, gt_env, pred_env)

    compose(args, gt_render, pred_render)


if __name__ == "__main__":
    main()
