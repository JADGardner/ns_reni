"""Runtime measurements for the joint-GS core (tab:runtime refresh).

Measures, on the current GPU, for the thesis core model:
  - single decoder forward, 8192 rays (ms, median of 100 after warmup)
  - per-step latent-optimisation time and total for 2500 steps x 21 images
  - peak GPU memory for both

Training time is taken from the migration queue logs, not measured here.

    docker compose run --rm -w /workspace/phd/code/ns_reni research bash -c \
      "PYTHONPATH=.:scripts/figures python scripts/figures/diag_runtime_core.py"
"""

from __future__ import annotations

import time

import torch

from _common import MODEL_DIRS, equirect_ray_bundle, load_model
from reni.field_components.field_heads import RENIFieldHeadNames

DEVICE = "cuda:0"
RAYS = 8192
FIT_STEPS = 2500
N_IMAGES = 21


def main() -> None:
    _, _, model = load_model(MODEL_DIRS["vnjoint_ortho_2cyc_testfit"][100],
                             device=DEVICE)
    model.eval()
    rb = equirect_ray_bundle(DEVICE, idx=0, height=64)
    o, d, c = rb.origins[:RAYS], rb.directions[:RAYS], rb.camera_indices[:RAYS]
    z = torch.randn(1, model.field.latent_dim, 3, device=DEVICE)

    def forward():
        samples = model.create_ray_samples(o, d, c)
        latents = z.repeat(samples.shape[0], 1, 1)
        return model.field.forward(samples, rotation=None,
                                   latent_codes=latents)[RENIFieldHeadNames.RGB]

    # --- inference timing ---------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(20):
            forward()
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    fwd_ms = times[len(times) // 2] * 1e3
    fwd_mem = torch.cuda.max_memory_allocated() / 2**20
    print(f"forward {RAYS} rays: {fwd_ms:.2f} ms median, peak {fwd_mem:.0f} MiB")

    # --- fit timing ---------------------------------------------------------
    target = torch.rand(RAYS, model.field.out_features, device=DEVICE)
    zf = (1e-2 * torch.randn(1, model.field.latent_dim, 3,
                             device=DEVICE)).requires_grad_(True)
    optim = torch.optim.Adam([zf], lr=1e-1)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(20):
        optim.zero_grad()
        samples = model.create_ray_samples(o, d, c)
        latents = zf.repeat(samples.shape[0], 1, 1)
        out = model.field.forward(samples, rotation=None,
                                  latent_codes=latents)[RENIFieldHeadNames.RGB]
        torch.nn.functional.mse_loss(out, target).backward()
        optim.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 200
    for _ in range(steps):
        optim.zero_grad()
        samples = model.create_ray_samples(o, d, c)
        latents = zf.repeat(samples.shape[0], 1, 1)
        out = model.field.forward(samples, rotation=None,
                                  latent_codes=latents)[RENIFieldHeadNames.RGB]
        torch.nn.functional.mse_loss(out, target).backward()
        optim.step()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / steps * 1e3
    fit_mem = torch.cuda.max_memory_allocated() / 2**20
    total_s = step_ms / 1e3 * FIT_STEPS * N_IMAGES
    print(f"fit step: {step_ms:.2f} ms, peak {fit_mem:.0f} MiB, "
          f"{FIT_STEPS} steps x {N_IMAGES} imgs = {total_s:.1f} s")


if __name__ == "__main__":
    main()
