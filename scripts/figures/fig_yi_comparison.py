"""Yi et al. (CVPR 2023) vs RENI++ lighting-from-object comparison (thesis Ch.2).

For each RENI_HDR validation environment map: render the Stanford bunny under
the GT illumination (Blinn-Phong), recover lighting with (a) Yi et al.'s 2nd-order
SH inverse-rendering predictor and (b) RENI++ latent-code optimisation, then
compare both recovered environment maps against GT. Emits the comparison grid
(thesis Fig 2.x) and the LaTeX metrics table.

The heavy per-method logic (Yi model loading/inference, RENI++ decoder loading,
the differentiable inverse-render optimisation, rendering utilities and metrics)
is reused verbatim from ``publication/yi_et_al_vs_reni_comparison.py``; only the
driver, grid figure and table are re-done in the shared fig_ style.

    PYTHONPATH=. python scripts/figures/fig_yi_comparison.py
    PYTHONPATH=. python scripts/figures/fig_yi_comparison.py --labels \
        --output publication/figures/yi_et_al_comparison_grid_labeled

NOTE ON THE `model` IMPORT: the reused publication module does
``from model import InverseRenderModel`` against the vendored Yi et al. tree.
That bare ``model`` name is shadowed if a nerfstudio/reni import registers a
``model`` namespace first, so the Yi module is imported *before* ``_common``.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Import the reused Yi driver FIRST (with the vendored tree on the path) so its
# bare `from model import InverseRenderModel` resolves before anything else can
# register a `model` namespace package.
sys.path.insert(0, str(_REPO_ROOT / "thirdparty" / "Yi_et_al_relighting" / "InverseRendering"))
sys.path.insert(0, str(_REPO_ROOT / "publication"))
import yi_et_al_vs_reni_comparison as yi  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# torch>=2.6 defaults weights_only=True; the reused loaders and the paper
# checkpoints (which pickle numpy scalars) predate that. Restore the old default
# for these trusted local checkpoints, matching _common.load_model.
_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):  # noqa: E302
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from _common import (MODEL_DIRS, add_common_args, add_figure_label,  # noqa: E402
                     axis_center, resolve_run_dir, save_figure, seed_all)
from reni.utils.colourspace import linear_to_sRGB  # noqa: E402
from reni.model_components.illumination_samplers import EquirectangularSamplerConfig  # noqa: E402
from reni.model_components.shaders import BlinnPhongShader  # noqa: E402
from reni.illumination_fields.sh_illumination_field import shReconstructSignal  # noqa: E402
from reni.illumination_fields.reni_illumination_field import RENIFieldConfig  # noqa: E402
from reni.field_components.field_heads import RENIFieldHeadNames  # noqa: E402
from reni.utils.tonemap import two_bracket_to_linear  # noqa: E402
from reni.utils.checkpoint_locator import find_checkpoint  # noqa: E402

# Grid columns and the thesis TikZ labels above them.
COL_LABELS = ["Rendered Object", "Yi et al. (SH)", "RENI++", "Ground Truth"]


def _load_two_bracket_decoder(model_dir, device):
    """Load a frozen two-bracket RENI++ decoder (out_features=6, sigmoid).

    Mirrors yi.load_reni_decoder's field config (which matches the two-bracket
    training config) but for the six-channel bracket output.
    """
    run_dir = resolve_run_dir(Path(model_dir))
    ckpt_dir = run_dir / "nerfstudio_models"
    step = sorted(int(p.stem.split("-")[1]) for p in ckpt_dir.glob("step-*.ckpt"))[-1]
    config = RENIFieldConfig(
        conditioning="Attention", invariant_function="VN", equivariance="SO2",
        axis_of_invariance="z", positional_encoding="NeRF",
        encoded_input="Directions", latent_dim=100, hidden_features=128,
        hidden_layers=9, mapping_layers=5, mapping_features=128,
        num_attention_heads=8, num_attention_layers=6, output_activation="sigmoid",
        last_layer_linear=True, fixed_decoder=True, trainable_scale=False,
        out_features=6,
    )
    field = config.setup(num_train_data=None, num_eval_data=None)
    ckpt = torch.load(str(ckpt_dir / f"step-{step:09d}.ckpt"),
                      map_location=device, weights_only=False)
    ignore = ("train_logvar", "eval_logvar", "train_mu", "eval_mu")
    fdict = {k[len("_model.field."):]: v for k, v in ckpt["pipeline"].items()
             if k.startswith("_model.field.") and not any(ig in k for ig in ignore)}
    field.load_state_dict(fdict, strict=False)
    field = field.to(device).eval()
    for p in field.parameters():
        p.requires_grad = False
    print(f"[reni] two-bracket decoder from {ckpt_dir}/step-{step:09d}.ckpt")
    return field


def _decode_two_bracket_envmap(field, latent_codes, scale, ray_samples, num_dirs,
                               height, width, m_ldr, m_log):
    """Decode a latent to a linear-HDR envmap via the two-bracket blend."""
    latents_e = latent_codes.unsqueeze(0).expand(num_dirs, -1, -1)
    scale_e = scale.expand(num_dirs)
    with torch.set_grad_enabled(latent_codes.requires_grad):
        outputs = field.forward(ray_samples=ray_samples, latent_codes=latents_e,
                                scale=scale_e)
    rgb = two_bracket_to_linear(outputs[RENIFieldHeadNames.RGB], m_ldr=m_ldr, m_log=m_log)
    return rgb.reshape(height, width, 3)


def _reni_inverse_render_two_bracket(field, gt_render, normals, mask,
                                     light_directions, view_directions, shader,
                                     sampler, device, m_ldr, m_log, num_steps=500,
                                     lr=0.01, specular_term=0.2, shininess=500.0):
    """Two-bracket variant of yi.reni_inverse_render: optimise latent + exposure
    from object appearance, decoding illumination through the two-bracket blend.
    Same loss, optimiser and trainable exp-scale as the reused paper routine."""
    latent_codes = nn.Parameter(torch.zeros(field.latent_dim, 3, device=device))
    scale = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([latent_codes, scale], lr=lr)
    l2_loss_fn = nn.MSELoss()
    cosine_sim = nn.CosineSimilarity(dim=-1)

    ray_samples = sampler.generate_direction_samples().to(device)
    num_dirs = ray_samples.frustums.directions.shape[0]
    ray_samples.camera_indices = torch.zeros(num_dirs, dtype=torch.long, device=device)

    for _ in range(num_steps + 1):
        optimizer.zero_grad()
        pred_envmap = _decode_two_bracket_envmap(
            field, latent_codes, torch.exp(scale), ray_samples, num_dirs,
            sampler.height, sampler.width, m_ldr, m_log)
        pred_render = yi.render_with_environment(
            normals, mask, pred_envmap, light_directions, view_directions,
            shader, specular_term, shininess)
        q = torch.quantile(gt_render.flatten(), 0.98)
        gt_srgb = linear_to_sRGB(gt_render, q=q, clamp=False)
        pred_srgb = linear_to_sRGB(pred_render, q=q, clamp=False)
        rgb_loss = l2_loss_fn(pred_srgb, gt_srgb)
        cos_loss = 1.0 - cosine_sim(pred_srgb.reshape(-1, 3), gt_srgb.reshape(-1, 3)).mean()
        prior_loss = torch.mean(latent_codes ** 2)
        (100.0 * rgb_loss + 1.0 * cos_loss + 0.001 * prior_loss).backward()
        optimizer.step()

    with torch.no_grad():
        return _decode_two_bracket_envmap(
            field, latent_codes, torch.exp(scale), ray_samples, num_dirs,
            sampler.height, sampler.width, m_ldr, m_log)


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {what}: expected at {path}")
    return path


def _grid(results, add_labels=False, label_fontsize=15.0, max_rows=5):
    n_show = min(len(results), max_rows)
    fig, axes = plt.subplots(
        n_show, 4, figsize=(20, 4 * n_show),
        gridspec_kw={"width_ratios": [1, 2, 2, 2]},
    )
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for row in range(n_show):
        res = results[row]
        yi_d = linear_to_sRGB(res["yi_envmap"], use_quantile=True).numpy()
        reni_d = linear_to_sRGB(res["reni_envmap"], use_quantile=True).numpy()
        gt_d = linear_to_sRGB(res["gt_envmap"], use_quantile=True).numpy()
        cells = [res["gt_render_ldr"].numpy(),
                 np.clip(yi_d, 0, 1), np.clip(reni_d, 0, 1), np.clip(gt_d, 0, 1)]
        for col, cell in enumerate(cells):
            axes[row, col].imshow(cell)
            axes[row, col].axis("off")

    plt.tight_layout()
    if add_labels:
        for col, text in enumerate(COL_LABELS):
            add_figure_label(
                fig,
                axis_center(axes[0, col])[0],
                axes[0, col].get_position().y1 + 0.006,
                text,
                label_fontsize,
            )
    return fig


def _write_table(mean_metrics, out_stem: Path):
    lines = [
        r"\begin{tabular}{l|cc}", r"\hline",
        r"Method & LDR PSNR$\uparrow$ & SSIM$\uparrow$ \\", r"\hline",
    ]
    for method, m in mean_metrics.items():
        lines.append(f"{method} & {m['LDR_PSNR']:.2f} & {m['SSIM']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    table = "\n".join(lines)
    tex_path = Path(f"{out_stem}_metrics.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(table)
    print(f"[saved] {tex_path}")
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "yi_comparison_thesis")
    parser.add_argument("--model", default="two_bracket_w3_2cyc",
                        help="MODEL_DIRS key for the RENI++ decoder (default: "
                             "thesis two-bracket 2-cycle; use reni_pp for the paper model)")
    parser.add_argument("--m_ldr", type=float, default=16.0)
    parser.add_argument("--m_log", type=float, default=10000.0)
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of validation environment maps to evaluate")
    parser.add_argument("--reni_steps", type=int, default=500,
                        help="RENI++ inverse-render optimisation steps per image")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Rendered-object image size (pixels)")
    parser.add_argument("--labels", action="store_true",
                        help="Bake current thesis TikZ column labels into the figure")
    parser.add_argument("--label_fontsize", type=float, default=15.0)
    args = parser.parse_args()
    two_bracket = args.model != "reni_pp"
    seed_all(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() or "cuda" not in args.device
        else "cpu")

    illumination_width, illumination_height = 128, 64

    yi_model_path = _require(
        _REPO_ROOT / "thirdparty" / "Yi_et_al_relighting" / "InverseRendering"
        / "path" / "invrender.pth", "Yi et al. weights")
    yi_net = yi.load_yi_model(yi_model_path, device)

    if two_bracket:
        reni_field = _load_two_bracket_decoder(MODEL_DIRS[args.model][100], device)
    else:
        reni_ckpt_path = find_checkpoint("checkpoints/reni_plus_plus_models/latent_dim_100")
        reni_field = yi.load_reni_decoder(reni_ckpt_path, ckpt_step=50000, device=device)

    normal_map_path = _require(
        _REPO_ROOT / "data" / "RENI_HDR" / "3d_models" / "normal_maps"
        / "bunny_normals.exr", "bunny normal map")
    normals, mask = yi.load_normal_map(normal_map_path, args.image_size)
    normals, mask = normals.to(device), mask.to(device)

    sampler = EquirectangularSamplerConfig(
        width=illumination_width, apply_random_rotation=False,
        remove_lower_hemisphere=False).setup()
    shader = BlinnPhongShader()

    ray_samples = sampler.generate_direction_samples()
    light_directions = ray_samples.frustums.directions.to(device)
    light_directions = light_directions / torch.norm(light_directions, dim=-1, keepdim=True)
    view_directions = yi.create_view_directions(args.image_size, device)

    data_dir = _require(_REPO_ROOT / "data" / "RENI_HDR" / "val",
                        "RENI_HDR validation split")
    exr_files = sorted(data_dir.glob("*.exr"))[:args.num_images]
    if not exr_files:
        raise SystemExit(f"No .exr environment maps found under {data_dir}")
    print(f"[yi] evaluating {len(exr_files)} environment maps")

    all_metrics = {"Yi et al.": [], "RENI++": []}
    results = []

    for exr_path in exr_files:
        gt_envmap = yi.load_environment_map(
            exr_path, illumination_height, illumination_width).to(device)

        with torch.no_grad():
            gt_render = yi.render_with_environment(
                normals, mask, gt_envmap, light_directions, view_directions,
                shader, background_color=1.0)
            gt_render_ldr = torch.clamp(
                linear_to_sRGB(gt_render, use_quantile=True), 0, 1)
            mask_3ch = mask.unsqueeze(-1).expand_as(gt_render_ldr).float()
            gt_render_ldr = gt_render_ldr * mask_3ch + (1.0 - mask_3ch)

        yi_sh = yi.yi_predict_sh(yi_net, gt_render_ldr, mask.float(), device)
        yi_envmap = torch.clamp(
            shReconstructSignal(yi_sh, width=illumination_width, device=device), min=0)

        if two_bracket:
            reni_envmap = _reni_inverse_render_two_bracket(
                reni_field, gt_render, normals, mask, light_directions,
                view_directions, shader, sampler, device, args.m_ldr, args.m_log,
                num_steps=args.reni_steps)
        else:
            reni_envmap = yi.reni_inverse_render(
                reni_field, gt_render, normals, mask, light_directions,
                view_directions, shader, sampler, device, num_steps=args.reni_steps)

        gt_np = gt_envmap.cpu().numpy()
        yi_m = yi.compute_metrics(gt_np, yi_envmap.detach().cpu().numpy())
        reni_m = yi.compute_metrics(gt_np, reni_envmap.detach().cpu().numpy())
        all_metrics["Yi et al."].append(yi_m)
        all_metrics["RENI++"].append(reni_m)
        print(f"  {exr_path.name}: Yi LDR_PSNR={yi_m['LDR_PSNR']:.2f} "
              f"SSIM={yi_m['SSIM']:.4f} | RENI++ LDR_PSNR={reni_m['LDR_PSNR']:.2f} "
              f"SSIM={reni_m['SSIM']:.4f}")

        results.append({
            "gt_envmap": gt_envmap.cpu(),
            "yi_envmap": yi_envmap.detach().cpu(),
            "reni_envmap": reni_envmap.detach().cpu(),
            "gt_render_ldr": gt_render_ldr.cpu(),
        })

    mean_metrics = {
        method: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        for method, ms in all_metrics.items()
    }
    print("\n[yi] aggregate:")
    for method, m in mean_metrics.items():
        print(f"  {method:12s} LDR_PSNR={m['LDR_PSNR']:.2f} SSIM={m['SSIM']:.4f}")

    fig = _grid(results, add_labels=args.labels, label_fontsize=args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg, dpi=150)
    _write_table(mean_metrics, args.output)


if __name__ == "__main__":
    main()
