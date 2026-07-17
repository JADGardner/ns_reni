"""Gate-flip vs trajectory-divergence diagnostic for the joint-GS model.

The rotation-sweep residual at generic angles (~5e-2, precision-independent)
against machine-precision recovery at 90 degrees suggests the live VNReLU
gates make the fit dynamics piecewise: conjugate trajectories commute
exactly until an epsilon-scale difference crosses a gate boundary, and each
crossing injects a small kick. This script runs a reference fit and a
rotated fit (co-rotated init, rotated directions, same pixels) side by
side, recording at every GD step:

  - rel divergence ||Z_rot - Z R|| / ||Z R||
  - the number of VNReLU gate-sign disagreements between the two passes

at a signed-permutation angle (90, expected: zero flips, flat divergence)
and a generic angle (45, expected: flips correlated with growth).

Run from the phd repo root:

    docker compose run --rm -w /workspace/phd/code/ns_reni research bash -c \
      "PYTHONPATH=.:scripts/figures python scripts/figures/diag_gate_flip_divergence.py"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from _common import REPO_ROOT, equirect_ray_bundle, init_fit_latent, seed_all
from eval_latent_reset_compare import (
    _build_test_config,
    _latest_checkpoint,
    _load_decoder_state,
    resolve_run_dir,
)
from reni.field_components.field_heads import RENIFieldHeadNames

RUN = "outputs/reni_latent_reset_d100_two_bracket_ldrw3_2cyc_vnjoint_ortho"
DATA_ROOT = REPO_ROOT / "data" / "RENI_HDR"
DEVICE = "cuda:0"
IMAGE_IDX = 0
STEPS = 400
GD_LR = 0.3
GD_MOMENTUM = 0.9
ANGLES = (90.0, 45.0)
OUT = REPO_ROOT / "outputs" / "evaluations" / "diag_gate_flip_divergence.json"


def rot_about_axis(deg: float, axis: int, device, dtype) -> torch.Tensor:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 1:  # y
        m = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    else:  # z (the rotation-sweep convention for these models)
        m = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return torch.tensor(m, device=device, dtype=dtype)


class GateRecorder:
    """Captures VNReLU gate signs (sign of q.k) via forward hooks."""

    def __init__(self, model):
        self.signs = []
        self.handles = []
        from reni.field_components.vn_layers import VNReLU

        def hook(mod, inp, out):
            x = inp[0]
            q = torch.einsum("... i c, o i -> ... o c", x, mod.W)
            k = torch.einsum("... i c, o i -> ... o c", x, mod.U)
            qk = (q * k).sum(-1)
            self.signs.append(qk >= 0)

        for m in model.modules():
            if isinstance(m, VNReLU):
                self.handles.append(m.register_forward_hook(hook))

    def take(self):
        s = self.signs
        self.signs = []
        return s

    def close(self):
        for h in self.handles:
            h.remove()


def main() -> None:
    seed_all(42)
    run_dir = resolve_run_dir(Path(RUN))
    checkpoint = _latest_checkpoint(run_dir)
    model_config = _build_test_config(run_dir, DATA_ROOT, None)
    pipeline = model_config.pipeline.setup(
        device=DEVICE, test_mode="test", world_size=1, local_rank=0,
        grad_scaler=None)
    _load_decoder_state(pipeline, checkpoint, DEVICE)
    model = pipeline.model
    model.to(DEVICE)
    model.eval()

    batch = pipeline.datamanager.eval_dataset[IMAGE_IDX]
    gt = batch["image"]
    if gt.dim() == 4:
        gt = gt[0]
    gt = gt.to(DEVICE)
    H, W, C = gt.shape
    target = gt.reshape(-1, C)
    ray_bundle = equirect_ray_bundle(DEVICE, idx=0, height=H)
    directions = ray_bundle.directions.to(DEVICE)

    # Axis of invariance for the field (index): rotate about it.
    axis = model.field.axis_of_invariance

    z_base = init_fit_latent(model, DEVICE, requires_grad=False)

    recorder = GateRecorder(model)
    results = {}

    def fit(directions_in, z0, record_states):
        """Momentum-GD fit; returns list of (z_t, gate_signs_t)."""
        torch.manual_seed(0)
        z = z0.detach().clone().requires_grad_(True)
        buf = torch.zeros_like(z)
        states = []
        for t in range(STEPS):
            samples = model.create_ray_samples(
                ray_bundle.origins.to(DEVICE), directions_in,
                ray_bundle.camera_indices.to(DEVICE))
            latents = z.repeat(samples.shape[0], 1, 1)
            recorder.take()
            out = model.field.forward(samples, rotation=None,
                                      latent_codes=latents)[RENIFieldHeadNames.RGB]
            gates = recorder.take()
            mse = torch.nn.functional.mse_loss(out, target)
            cos = 1 - torch.nn.functional.cosine_similarity(out, target, dim=-1).mean()
            loss = 10.0 * mse + cos
            grad = torch.autograd.grad(loss, z)[0]
            with torch.no_grad():
                buf = GD_MOMENTUM * buf + grad
                z = (z - GD_LR * buf).detach().requires_grad_(True)
            if record_states:
                states.append((z.detach().clone(),
                               [g.detach().clone() for g in gates]))
        return states

    for angle in ANGLES:
        R = rot_about_axis(angle, axis, DEVICE, directions.dtype)
        ref = fit(directions, z_base, True)
        rot = fit(directions @ R, z_base @ R, True)
        div, flips = [], []
        for (z_r, g_r), (z_t, g_t) in zip(ref, rot):
            zRr = z_r @ R
            div.append(float(torch.norm(z_t - zRr) / torch.norm(zRr)))
            n = sum(int((a != b).sum()) for a, b in zip(g_r, g_t))
            flips.append(n)
        results[str(angle)] = {"divergence": div, "gate_flips": flips}
        print(f"[{angle:g} deg] final div {div[-1]:.3e}  "
              f"total flips {sum(flips)}  "
              f"first nonzero flip step: "
              f"{next((i for i, f in enumerate(flips) if f), None)}")

    recorder.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
