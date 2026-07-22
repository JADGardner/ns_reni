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


def main() -> None:
    argv = sys.argv
    labels_path = pop_arg(argv, "--sun-labels")
    channel = int(pop_arg(argv, "--sun-channel", "9"))
    weight = float(pop_arg(argv, "--sun-weight", "2.0"))
    pair_weight = float(pop_arg(argv, "--pair-weight", "0.0"))
    if labels_path is None:
        raise SystemExit("--sun-labels is required")

    labels = json.loads(Path(labels_path).read_text())
    names = sorted(labels)
    dirs = torch.tensor([labels[n]["sun_direction"] for n in names],
                        dtype=torch.float32)
    mirrored = dirs * torch.tensor([-1.0, 1.0, 1.0])
    full = torch.cat([dirs, mirrored], dim=0)          # [2N, 3] bank order
    full = F.normalize(full, dim=-1)
    print(f"[sun] {len(names)} labelled envs -> {full.shape[0]} bank rows, "
          f"channel {channel}, weight {weight}")

    # Counterfactual pairs: bank indices (a, b) for both the original and the
    # mirrored halves; non-sun channels of pair members are tied.
    pair_a, pair_b = [], []
    if pair_weight > 0:
        by_pair: dict[int, dict[str, int]] = {}
        for idx, n in enumerate(names):
            info = labels[n]
            if "pair_id" in info:
                by_pair.setdefault(info["pair_id"], {})[info["member"]] = idx
        n_imgs = len(names)
        for members in by_pair.values():
            if "a" in members and "b" in members:
                for off in (0, n_imgs):
                    pair_a.append(members["a"] + off)
                    pair_b.append(members["b"] + off)
        print(f"[sun] {len(pair_a)} tied pairs (incl. mirrored), "
              f"pair weight {pair_weight}")
    pair_a_t = torch.tensor(pair_a, dtype=torch.long)
    pair_b_t = torch.tensor(pair_b, dtype=torch.long)

    from reni.models.reni_model import RENIModel

    original = RENIModel.get_loss_dict
    state = {"warned": False}

    def patched(self, outputs, batch, metrics_dict=None):
        loss_dict = original(self, outputs, batch, metrics_dict)
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
        loss_dict["sun_channel_loss"] = weight * (1.0 - (z * target).sum(-1)).mean()
        if pair_weight > 0 and len(pair_a):
            a = mu[pair_a_t.to(mu.device)]
            b = mu[pair_b_t.to(mu.device)]
            keep = [c for c in range(mu.shape[1]) if c != channel]
            diff = a[:, keep, :] - b[:, keep, :]
            loss_dict["sun_pair_loss"] = pair_weight * diff.pow(2).mean()
        return loss_dict

    RENIModel.get_loss_dict = patched

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train_reni

    train_reni.main()


if __name__ == "__main__":
    main()
