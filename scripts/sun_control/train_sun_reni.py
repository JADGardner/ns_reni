"""Train RENI++ with one latent channel supervised to be the sun direction.

Thin wrapper over scripts/train_reni.py: it patches RENIModel.get_loss_dict
at class level to add a cosine alignment loss between the designated latent
channel's direction and the per-image ground-truth sun direction (from the
synthetic generator's sun_labels.json), then delegates to the standard
training entry point with the remaining CLI arguments.

Label ordering matches the training bank layout verified by the emergence
probe: originals in sorted-filename order first, mirrored copies second,
with the mirror negating the azimuthal x component of the sun direction.

    PYTHONPATH=. python scripts/sun_control/train_sun_reni.py \
        --sun-labels data/RENI_SUN_SYNTH/train/sun_labels.json \
        --sun-channel 9 --sun-weight 2.0 \
        --data data/RENI_SUN_SYNTH --latent-dim 100 --variant two_bracket \
        --ldr-bracket-weight 3 --invariant-function VNJoint \
        --canonical-frame-ortho --training-paradigm latent_reset \
        --latent-reset-cycles 1 --max-num-iterations 50001 \
        --experiment-name reni_sun_synth_d100
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F


def pop_arg(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def pop_flag(argv: list[str], name: str) -> bool:
    if name not in argv:
        return False
    argv.remove(name)
    return True


def main() -> None:
    argv = sys.argv
    labels_path = pop_arg(argv, "--sun-labels")
    channel = int(pop_arg(argv, "--sun-channel", "9"))
    weight = float(pop_arg(argv, "--sun-weight", "2.0"))
    pair_weight = float(pop_arg(argv, "--pair-weight", "0.0"))
    cmd_weight = float(pop_arg(argv, "--cmd-weight", "0.0"))
    init_decoder = pop_arg(argv, "--init-decoder-from")
    structural_groups = pop_flag(argv, "--structural-groups")
    if labels_path is None:
        raise SystemExit("--sun-labels is required")
    if structural_groups and (pair_weight > 0 or cmd_weight > 0):
        raise SystemExit(
            "--structural-groups replaces pair and command penalties; use "
            "--pair-weight 0 --cmd-weight 0")

    labels = json.loads(Path(labels_path).read_text())
    names = sorted(labels)
    dirs = torch.tensor([labels[n]["sun_direction"] for n in names],
                        dtype=torch.float32)
    mirrored = dirs * torch.tensor([-1.0, 1.0, 1.0])
    full = torch.cat([dirs, mirrored], dim=0)          # [2N, 3] bank order
    full = F.normalize(full, dim=-1)
    wts = torch.tensor([float(labels[n].get("weight", 1.0)) for n in names])
    wts = torch.cat([wts, wts], dim=0)                 # mirrored copies share
    print(f"[sun] {len(names)} labelled envs -> {full.shape[0]} bank rows, "
          f"channel {channel}, weight {weight}")

    # Counterfactual pairs: bank indices (a, b) for both the original and the
    # mirrored halves; non-sun channels of pair members are tied.
    pair_a, pair_b = [], []
    if pair_weight > 0 and not structural_groups:
        by_pair: dict[int, dict[str, int]] = {}
        for idx, n in enumerate(names):
            info = labels[n]
            if "pair_id" in info:
                by_pair.setdefault(info["pair_id"], {})[info["member"]] = idx
        n_imgs = len(names)
        for members in by_pair.values():
            for ma, mb in combinations(sorted(members), 2):
                for off in (0, n_imgs):
                    pair_a.append(members[ma] + off)
                    pair_b.append(members[mb] + off)
        print(f"[sun] {len(pair_a)} tied member pairs (incl. mirrored), "
              f"pair weight {pair_weight}")
    pair_a_t = torch.tensor(pair_a, dtype=torch.long)
    pair_b_t = torch.tensor(pair_b, dtype=torch.long)

    from reni.models.reni_model import RENIModel

    if structural_groups:
        argv.extend([
            "--structural-sun-labels", labels_path,
            "--structural-sun-channel", str(channel),
        ])
        print(
            "[sun] structural groups enabled: non-sun channels are shared "
            "by construction and the sun channel is hard-set; cosine, norm, "
            "pair and command penalties are disabled")

    if init_decoder is not None:
        ckpt_path = Path(init_decoder)
        orig_pop = RENIModel.populate_modules

        def pop_patched(self):
            orig_pop(self)
            sd = torch.load(ckpt_path, map_location="cpu",
                            weights_only=False)["pipeline"]
            field_sd = {k.split("_model.field.", 1)[1]: v
                        for k, v in sd.items() if "_model.field." in k}
            own = self.field.state_dict()
            filtered = {k: v for k, v in field_sd.items()
                        if k in own and own[k].shape == v.shape
                        and "train_mu" not in k and "eval_mu" not in k
                        and "train_log_var" not in k and "eval_log_var" not in k}
            missing = self.field.load_state_dict(filtered, strict=False)
            print(f"[sun] decoder init from {ckpt_path.name}: "
                  f"{len(filtered)} tensors loaded, latents fresh")

        RENIModel.populate_modules = pop_patched

    # Command-consistency: decode random bank latents with ch9 overwritten
    # by a random commanded direction on a small ERP grid, and require the
    # decoded luminance soft-peak to sit at the command. Reconstruction
    # alone lets the decoder source the sun from content channels for real
    # images (ch9 decouples); this loss forces the causal route for ALL
    # content. Gradients flow to the decoder only (latents detached).
    CMD_H, CMD_W = 24, 48
    cmd_state: dict = {}

    def cmd_loss(self):
        import math as _m
        if "dirs" not in cmd_state:
            v = (torch.arange(CMD_H) + 0.5) / CMD_H
            u = (torch.arange(CMD_W) + 0.5) / CMD_W
            pol, az = torch.meshgrid(v * _m.pi, u * 2 * _m.pi - _m.pi,
                                     indexing="ij")
            d = torch.stack([torch.sin(pol) * torch.sin(az), torch.cos(pol),
                             torch.sin(pol) * torch.cos(az)], -1).reshape(-1, 3)
            dev = self.field.train_mu.device
            cmd_state["dirs"] = d.to(dev)
            cmd_state["origins"] = torch.zeros_like(cmd_state["dirs"])
            cmd_state["cam"] = torch.zeros(d.shape[0], 1, dtype=torch.long,
                                           device=dev)
        dirs = cmd_state["dirs"]
        mu = self.field.train_mu.detach()
        # Fresh fine-tune latents start near zero; an unfloored norm writes
        # the command with no magnitude and the loss can only suppress
        # brightness globally (sunless collapse).
        norm = mu[:, channel].norm(dim=-1).median().clamp(min=1.0)
        total = 0.0
        from reni.utils.tonemap import two_bracket_to_linear, luminance
        for _ in range(2):
            i = int(torch.randint(0, mu.shape[0], (1,)))
            el = torch.deg2rad(5.0 + 75.0 * torch.rand(1, device=mu.device))
            az = torch.rand(1, device=mu.device) * 2 * torch.pi - torch.pi
            d = torch.cat([torch.cos(el) * torch.sin(az), torch.sin(el),
                           torch.cos(el) * torch.cos(az)])
            z = mu[i : i + 1].clone()
            z[0, channel] = norm * d
            sm = self.create_ray_samples(cmd_state["origins"], dirs,
                                         cmd_state["cam"])
            from reni.field_components.field_heads import RENIFieldHeadNames
            out = self.field.forward(
                sm, rotation=None,
                latent_codes=z.repeat(sm.shape[0], 1, 1))[RENIFieldHeadNames.RGB]
            lin = two_bracket_to_linear(out, m_ldr=self.tonemap_m_ldr,
                                        m_log=self.tonemap_m_log)
            lum = luminance(lin).clamp(min=1e-8)
            # Add-only objective: a bright blob (>=5x the detached median
            # luminance) must exist within ~15 deg of the command. Zero when
            # satisfied, and the only gradient direction is BRIGHTEN the
            # commanded spot. Both dominance-style variants (soft-argmax,
            # max-margin) had a "dim other content" direction and collapsed
            # reconstruction through it; this one cannot.
            near = (dirs @ d) > 0.966                   # within ~15 deg
            m_near = lum[near].max()
            ref = lum.median().detach()
            total = total + torch.relu(1.0 - m_near / (5.0 * ref + 1e-8))
        return total / 2.0

    original = RENIModel.get_loss_dict
    state = {"warned": False}

    def patched(self, outputs, batch, metrics_dict=None):
        loss_dict = original(self, outputs, batch, metrics_dict)
        if structural_groups:
            return loss_dict
        mu = self.field.train_mu                        # [B, D, 3]
        target = full.to(mu.device)
        if mu.shape[0] != target.shape[0]:
            if not state["warned"]:
                print(f"[sun][warn] bank {mu.shape[0]} != labels "
                      f"{target.shape[0]}; supervising the overlap")
                state["warned"] = True
            n = min(mu.shape[0], target.shape[0])
            mu, target = mu[:n], target[:n]
        z = F.normalize(mu[:, channel, :], dim=-1)
        w = wts.to(mu.device)[:mu.shape[0]]
        cos_term = (1.0 - (z * target).sum(-1)) * w
        loss_dict["sun_channel_loss"] = weight * cos_term.sum() / w.sum().clamp(min=1e-6)
        # Anchor the channel's MAGNITUDE too: the cosine term is
        # norm-invariant, so under latent-reset cycles the KLD prior shrank
        # ch9 to ~0 (cycle-2 decoder already rendered suns from content
        # channels) and the sun channel died. A unit-norm target resists
        # that collapse.
        nrm = mu[:, channel, :].norm(dim=-1)
        loss_dict["sun_norm_loss"] = weight * 0.1 * (
            ((nrm - 1.0) ** 2 * w).sum() / w.sum().clamp(min=1e-6))
        if pair_weight > 0 and len(pair_a):
            a = mu[pair_a_t.to(mu.device)]
            b = mu[pair_b_t.to(mu.device)]
            keep = [c for c in range(mu.shape[1]) if c != channel]
            diff = a[:, keep, :] - b[:, keep, :]
            loss_dict["sun_pair_loss"] = pair_weight * diff.pow(2).mean()
        if cmd_weight > 0:
            state["steps"] = state.get("steps", 0) + 1
            if state["steps"] > 3000:
                loss_dict["sun_cmd_loss"] = cmd_weight * cmd_loss(self)
        return loss_dict

    RENIModel.get_loss_dict = patched

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train_reni

    train_reni.main()


if __name__ == "__main__":
    main()
