"""LDR-observation completion comparison grid (thesis Fig: ldr_completion).

For each test envmap: the clipped LDR frustum observation, then completions
by the SH and SG Gaussian priors and the thesis RENI++ (joint GS frame),
against ground truth. Top row per environment is tone-mapped sRGB at the
shared known exposure; bottom row is log-luminance heatmaps with the
colormap range anchored to the ground truth (render_eval_image convention).
Fits follow the completion-benchmark protocol exactly (64x128 visible
pixels, clipped-sRGB loss, chosen prior weights); panels decode at a higher
display resolution from the fitted parameters.

    PYTHONPATH=.:scripts/figures python scripts/figures/fig_ldr_completion.py
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _common import (MODEL_DIRS, REPO_ROOT, add_figure_label, axis_center,
                     equirect_ray_bundle, load_model, save_figure, seed_all)
from eval_outpaint_compare import EPS, _frustum_mask
from eval_outpaint_ldrfit import (
    FIT_LR,
    FIT_STEPS,
    SG_FIT_STEPS,
    SG_PRIOR_PATH,
    SH_PRIOR_PATH,
    fit_learnt,
    ldr_target,
)
from eval_outpaint_sg_prior import BatchedSG
from eval_outpaint_sh_prior import design_matrix
from nerfstudio.utils import colormaps
from reni.field_components.field_heads import RENIFieldHeadNames
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import fixed_gauge_scale, luminance
from reni.utils.tonemap import two_bracket_to_linear

FIT_H, FIT_W = 64, 128


def load_pair(path: str, device: str, disp_h: int):
    """EXR -> (fit-res gauge linear [64,128,3], display-res gauge linear).

    The gauge scale and exposure quantile come from the fit resolution so
    the figure matches the benchmark protocol exactly."""
    import numpy as np
    import pyexr

    img = pyexr.read(path).astype("float32")[..., :3]
    finite = np.isfinite(img)
    if not finite.all():
        img[~finite] = img[finite].max()
    img[img <= 0] = img[img > 0].min()
    t = torch.tensor(img).permute(2, 0, 1)[None]
    fit = F.interpolate(t, size=(FIT_H, FIT_W), mode="bilinear")[0].permute(1, 2, 0).to(device)
    disp = F.interpolate(t, size=(disp_h, disp_h * 2), mode="bilinear")[0].permute(1, 2, 0).to(device)
    scale = fixed_gauge_scale(fit)
    return fit * scale, disp * scale


def heat(lin: torch.Tensor, near, far) -> torch.Tensor:
    gray = torch.log1p(luminance(lin).clamp_min(0.0)).unsqueeze(-1)
    return colormaps.apply_depth_colormap(gray, near_plane=near, far_plane=far)


def fit_sh_prior(gauge_fit, target_vis, q, basis_vis, mu, sigma_inv, lam, device):
    from eval_outpaint_ldrfit import ldr_loss

    c = mu.reshape(-1, 3).clone().requires_grad_(True)
    optimiser = torch.optim.Adam([c], lr=FIT_LR)
    for _ in range(FIT_STEPS):
        optimiser.zero_grad()
        pred_lin_vis = torch.exp(basis_vis @ c)
        loss = ldr_loss(pred_lin_vis, target_vis, q)
        dev = c.reshape(-1) - mu
        loss = loss + lam * (dev @ sigma_inv @ dev) / dev.shape[0]
        loss.backward()
        optimiser.step()
    return c.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_indices", type=int, nargs="+", default=[4, 8, 18])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disp-height", type=int, default=128)
    parser.add_argument("--sh-lambda", type=float, default=1.0)
    parser.add_argument("--sg-lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_fontsize", type=float, default=13.0)
    parser.add_argument("--output", default=str(
        REPO_ROOT / "publication" / "figures" / "ldr_completion_thesis"))
    parser.add_argument("--svg", action="store_true")
    args = parser.parse_args()
    seed_all(args.seed)
    device = args.device
    disp_h = args.disp_height

    import glob
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "RENI_HDR" / "test" / "*.exr")))

    fit_bundle = equirect_ray_bundle(device, idx=0, height=FIT_H)
    disp_bundle = equirect_ray_bundle(device, idx=0, height=disp_h)
    mask_fit = _frustum_mask(FIT_H, FIT_W, 90.0, 60.0)
    mask_disp = _frustum_mask(disp_h, disp_h * 2, 90.0, 60.0).to(device)
    visible = torch.nonzero(mask_fit.reshape(-1).bool(), as_tuple=False).squeeze(1).to(device)

    # SH prior pieces.
    sh_prior = torch.load(SH_PRIOR_PATH)
    sh_mu = sh_prior["mu"].to(device).float()
    sh_sigma_inv = torch.linalg.inv(sh_prior["sigma"]).to(device).float()
    sh_basis_vis = design_matrix(fit_bundle, 9, device)[visible]
    sh_basis_disp = design_matrix(disp_bundle, 9, device)

    # SG prior pieces.
    sg_prior = torch.load(SG_PRIOR_PATH)
    sg_mu = sg_prior["mu"].to(device)
    sg_sigma_inv = torch.linalg.inv(sg_prior["sigma"]).to(device)
    _, _, sg_model = load_model(MODEL_DIRS["sg"][300], device=device)
    sg_fit = BatchedSG(sg_model.field, fit_bundle.directions)
    sg_disp = BatchedSG(sg_model.field, disp_bundle.directions)
    del sg_model
    torch.cuda.empty_cache()

    _, _, reni = load_model(MODEL_DIRS["vnjoint_ortho_2cyc"][100], device=device)
    reni.eval()

    panels = []
    for idx in args.image_indices:
        gauge_fit, gauge_disp = load_pair(files[idx], device, disp_h)
        target, q = ldr_target(gauge_fit)
        target_vis = target.reshape(-1, 3)[visible]

        # Observation panel: clipped sRGB inside the frustum, grey outside.
        obs_disp = linear_to_sRGB(gauge_disp, q=q, clamp=True)
        m = mask_disp.unsqueeze(-1)
        obs_disp = obs_disp * m + 0.55 * (1 - m)

        # SH + Gaussian prior.
        c = fit_sh_prior(gauge_fit, target_vis, q, sh_basis_vis.double().float(),
                         sh_mu, sh_sigma_inv, args.sh_lambda, device)
        sh_lin = torch.exp(sh_basis_disp @ c).reshape(disp_h, disp_h * 2, 3)

        # SG + Gaussian prior.
        from eval_outpaint_ldrfit import ldr_loss
        u = sg_fit.init_u(1, device).requires_grad_(True)
        optimiser = torch.optim.Adam([u], lr=FIT_LR)
        for _ in range(SG_FIT_STEPS):
            optimiser.zero_grad()
            out = sg_fit.render_log(u)[:, visible]
            loss = ldr_loss(torch.exp(out[0]), target_vis, q)
            dev = u.reshape(1, -1).double() - sg_mu[None]
            maha = torch.einsum("bi,ij,bj->b", dev, sg_sigma_inv, dev)
            loss = loss + args.sg_lambda * (maha.mean() / dev.shape[1]).float()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            sg_lin = torch.exp(sg_disp.render_log(u.detach())).reshape(
                disp_h, disp_h * 2, 3)

        # Thesis RENI++ (two-bracket, fixed gauge; no exposure scalar).
        z, log_s = fit_learnt(reni, gauge_fit, target, q, visible, fit_bundle,
                              device, fit_exposure=False)
        with torch.no_grad():
            chunks = []
            for start in range(0, disp_bundle.origins.shape[0], 65536):
                end = start + 65536
                samples = reni.create_ray_samples(
                    disp_bundle.origins[start:end],
                    disp_bundle.directions[start:end],
                    disp_bundle.camera_indices[start:end])
                out = reni.field.forward(
                    samples, rotation=None,
                    latent_codes=z.repeat(samples.shape[0], 1, 1))[RENIFieldHeadNames.RGB]
                chunks.append(two_bracket_to_linear(
                    out, m_ldr=reni.tonemap_m_ldr, m_log=reni.tonemap_m_log))
            reni_lin = torch.cat(chunks, dim=0).reshape(disp_h, disp_h * 2, 3)

        gray_gt = torch.log1p(luminance(gauge_disp).clamp_min(0.0))
        near, far = gray_gt.min(), gray_gt.max()
        row = {
            "obs": obs_disp,
            "sh": linear_to_sRGB(sh_lin, q=q, clamp=True),
            "sg": linear_to_sRGB(sg_lin, q=q, clamp=True),
            "reni": linear_to_sRGB(reni_lin, q=q, clamp=True),
            "gt": linear_to_sRGB(gauge_disp, q=q, clamp=True),
            "sh_heat": heat(sh_lin, near, far),
            "sg_heat": heat(sg_lin, near, far),
            "reni_heat": heat(reni_lin, near, far),
            "gt_heat": heat(gauge_disp, near, far),
        }
        panels.append({k: v.cpu() for k, v in row.items()})
        print(f"[fig] idx={idx} done")

    cols = ("obs", "sh", "sg", "reni", "gt")
    heats = (None, "sh_heat", "sg_heat", "reni_heat", "gt_heat")
    titles = ("Input (LDR $90^{\\circ}\\!\\times\\!60^{\\circ}$)",
              "SH + Gaussian prior", "SG + Gaussian prior",
              "RENI++ (ours)", "Ground truth")
    n_env = len(panels)
    fig, axs = plt.subplots(2 * n_env, 5, figsize=(16, 3.3 * n_env))
    if axs.ndim == 1:
        axs = axs[None]
    for r, row in enumerate(panels):
        for cidx, key in enumerate(cols):
            ax = axs[2 * r, cidx]
            ax.imshow(row[key].numpy())
            ax.axis("off")
        for cidx, key in enumerate(heats):
            ax = axs[2 * r + 1, cidx]
            if key is None:
                ax.axis("off")
                continue
            ax.imshow(row[key].numpy())
            ax.axis("off")
    plt.tight_layout()
    for cidx, title in enumerate(titles):
        ax = axs[0, cidx]
        add_figure_label(fig, axis_center(ax)[0],
                         ax.get_position().y1 + 0.018, title,
                         args.label_fontsize)
    save_figure(fig, args.output, svg=args.svg)


if __name__ == "__main__":
    main()
