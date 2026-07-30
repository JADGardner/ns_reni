"""Fit and evaluate causal sun-control bases on held-out synthetic scenes.

The decoder is frozen. For each held-out counterfactual scene, channel 9 is
pinned to the known sun direction while a single set of the other 99 latent
channels is fitted jointly to several renders of that scene under different
suns. Sharing those 99 channels prevents them from memorising one original
sun. The resulting base is then tested on unseen directions.

Controls:

* ``shared``: one content latent fitted across six counterfactual renders;
* ``single``: the same fit using only one render, so sun leakage is possible;
* ``mean``: the checkpoint's mean training latent;
* ``raw``: the checkpoint's ordinary reconstruction-fitted eval latents.

The script writes metrics, rendered sweeps, and a ``ui_bases.pt`` file that
``sun_ui.py --bases-file ... --bases-only`` can serve directly.

Example:

    PYTHONPATH=.:scripts/figures python scripts/sun_control/make_ui_bases.py \
        outputs/reni_sun_synth_v8_d100/reni/v8_1cyc \
        --label v8_1cyc --output-dir outputs/sun_control/counterfactual_control
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/figures")
from _common import equirect_ray_bundle, load_model  # noqa: E402
from reni.field_components.field_heads import RENIFieldHeadNames  # noqa: E402
from reni.utils.tonemap import (  # noqa: E402
    encode_two_bracket,
    luminance,
    two_bracket_to_linear,
)

from synthetic_sky import (  # noqa: E402
    apply_gauge,
    erp_directions,
    render_sky,
    sample_params,
)


FIT_COMMANDS = (
    (12.0, -145.0),
    (25.0, -75.0),
    (40.0, -5.0),
    (55.0, 65.0),
    (70.0, 135.0),
    (82.0, -175.0),
)
TEST_ELEVATIONS = (20.0, 35.0, 50.0, 65.0, 80.0)
TEST_AZIMUTHS = (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0)
TWILIGHT_ELEVATIONS = (-3.0, -7.0, -11.0)
TWILIGHT_AZIMUTHS = (-120.0, 0.0, 120.0)
DISPLAY_COMMANDS = (
    (30.0, -135.0),
    (30.0, -45.0),
    (30.0, 45.0),
    (30.0, 135.0),
    (70.0, -135.0),
    (70.0, -45.0),
    (70.0, 45.0),
    (70.0, 135.0),
)


@dataclass
class HeldoutScene:
    name: str
    fit_targets: torch.Tensor
    fit_directions: torch.Tensor
    eval_sky_mask: torch.Tensor
    nuisance_seed: int
    parameters: dict


def direction(elevation_deg: float, azimuth_deg: float, device=None) -> torch.Tensor:
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    return torch.tensor(
        [math.cos(el) * math.sin(az), math.sin(el),
         math.cos(el) * math.cos(az)],
        dtype=torch.float32,
        device=device,
    )


def analytic_grid(height: int, device=None) -> torch.Tensor:
    width = 2 * height
    pol = (torch.arange(height, device=device) + 0.5) / height * math.pi
    az = (torch.arange(width, device=device) + 0.5) / width * 2.0 * math.pi - math.pi
    pol, az = torch.meshgrid(pol, az, indexing="ij")
    return torch.stack(
        [torch.sin(pol) * torch.sin(az), torch.cos(pol),
         torch.sin(pol) * torch.cos(az)],
        dim=-1,
    )


def circular_error_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else float("nan")


class FrozenDecoder:
    def __init__(self, run_dir: Path, device: str, channel: int):
        _, _, model = load_model(run_dir, device=device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if not getattr(model, "two_bracket", False):
            raise ValueError("This experiment expects a two-bracket RENI++ model")

        self.model = model
        self.device = device
        self.channel = channel
        self.train_bank = model.field.train_mu.detach()
        self.eval_bank = getattr(model.field, "eval_mu", None)
        self.latent_dim = int(self.train_bank.shape[1])
        self.native_norm = float(
            self.train_bank[:, channel].norm(dim=-1).median())
        # A collapsed checkpoint is tested both at its native norm and at a
        # usable unit command magnitude. Healthy checkpoints retain their
        # native scale.
        self.command_norm = max(self.native_norm, 1.0)
        self._bundles: dict[int, object] = {}

    def bundle(self, height: int):
        if height not in self._bundles:
            self._bundles[height] = equirect_ray_bundle(
                self.device, idx=0, height=height)
        return self._bundles[height]

    def brackets(self, latents: torch.Tensor, height: int) -> torch.Tensor:
        """Decode B latent codes to BxHx2Hx6 bracket tensors."""
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        batch = int(latents.shape[0])
        rb = self.bundle(height)
        pixels = int(rb.origins.shape[0])
        origins = rb.origins.repeat(batch, 1)
        directions = rb.directions.repeat(batch, 1)
        camera_indices = rb.camera_indices.repeat(batch, 1)
        samples = self.model.create_ray_samples(
            origins, directions, camera_indices)
        per_ray_latents = (
            latents[:, None]
            .expand(batch, pixels, self.latent_dim, 3)
            .reshape(batch * pixels, self.latent_dim, 3)
        )
        output = self.model.field.forward(
            samples,
            rotation=None,
            latent_codes=per_ray_latents,
        )[RENIFieldHeadNames.RGB]
        return output.reshape(batch, height, 2 * height, 6)

    def linear(self, latents: torch.Tensor, height: int,
               chunk: int = 6) -> torch.Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(latents), chunk):
                brackets = self.brackets(latents[start:start + chunk], height)
                outputs.append(two_bracket_to_linear(
                    brackets,
                    m_ldr=self.model.tonemap_m_ldr,
                    m_log=self.model.tonemap_m_log,
                ))
        return torch.cat(outputs)

    def latents(self, content: torch.Tensor, commands: torch.Tensor,
                command_norm: float) -> torch.Tensor:
        """Insert B command vectors into a shared [N-1,3] content latent."""
        content = content.unsqueeze(0).expand(len(commands), -1, -1)
        command = command_norm * F.normalize(commands, dim=-1)
        return torch.cat(
            [content[:, :self.channel],
             command[:, None],
             content[:, self.channel:]],
            dim=1,
        )

    def fit_content(
        self,
        targets: torch.Tensor,
        commands: torch.Tensor,
        *,
        height: int,
        steps: int,
        learning_rate: float,
        command_norm: float,
        tag: str,
    ) -> tuple[torch.Tensor, dict]:
        targets = targets.to(self.device)
        commands = commands.to(self.device)
        target_brackets = encode_two_bracket(
            targets,
            m_ldr=self.model.tonemap_m_ldr,
            m_log=self.model.tonemap_m_log,
        )
        mean = self.train_bank.mean(0)
        initial = torch.cat(
            [mean[:self.channel], mean[self.channel + 1:]], dim=0)
        content = initial.clone().requires_grad_(True)
        optimiser = torch.optim.Adam([content], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=steps, eta_min=learning_rate * 0.1)

        first_loss = None
        for step in range(steps):
            optimiser.zero_grad()
            latents = self.latents(content, commands, command_norm)
            prediction = self.brackets(latents, height)
            ldr = F.mse_loss(prediction[..., :3], target_brackets[..., :3])
            log = F.mse_loss(prediction[..., 3:], target_brackets[..., 3:])
            prior = 1e-5 * content.square().mean()
            loss = 3.0 * ldr + log + prior
            loss.backward()
            optimiser.step()
            scheduler.step()
            if first_loss is None:
                first_loss = float(loss.detach())
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                print(
                    f"[fit {tag}] {step + 1:04d}/{steps}: "
                    f"loss={float(loss.detach()):.6f} "
                    f"ldr={float(ldr.detach()):.6f} "
                    f"log={float(log.detach()):.6f}",
                    flush=True,
                )

        with torch.no_grad():
            final_latents = self.latents(content, commands, command_norm)
            prediction = self.brackets(final_latents, height)
            ldr = F.mse_loss(prediction[..., :3], target_brackets[..., :3])
            log = F.mse_loss(prediction[..., 3:], target_brackets[..., 3:])
            pred_linear = two_bracket_to_linear(
                prediction,
                m_ldr=self.model.tonemap_m_ldr,
                m_log=self.model.tonemap_m_log,
            )
            log_linear = F.mse_loss(
                torch.log1p(pred_linear.clamp(min=0)),
                torch.log1p(targets),
            )
        stats = {
            "initial_loss": first_loss,
            "ldr_mse": float(ldr),
            "log_bracket_mse": float(log),
            "linear_log1p_mse": float(log_linear),
        }
        return content.detach(), stats


def build_scenes(
    count: int,
    fit_height: int,
    eval_height: int,
    seed: int,
) -> list[HeldoutScene]:
    scenes = []
    for index in range(count):
        parameter_rng = np.random.default_rng(seed + 101 * index)
        params = sample_params(parameter_rng)
        # The counterfactual group shares atmospheric/photometric parameters.
        # Only sun direction changes.
        nuisance_seed = seed + 10000 + 1009 * index
        fit_images = []
        fit_dirs = []
        for elevation, azimuth in FIT_COMMANDS:
            render_params = dict(params)
            render_params["sun_elevation_deg"] = elevation
            render_params["sun_azimuth_deg"] = azimuth
            image, sun_direction = render_sky(
                fit_height,
                2 * fit_height,
                dirs=erp_directions(fit_height, 2 * fit_height),
                nuisance_rng=np.random.default_rng(nuisance_seed),
                **render_params,
            )
            fit_images.append(apply_gauge(image))
            fit_dirs.append(sun_direction)

        mask_params = dict(params)
        mask_params["sun_elevation_deg"], mask_params["sun_azimuth_deg"] = \
            FIT_COMMANDS[0]
        _, _, geometry = render_sky(
            eval_height,
            2 * eval_height,
            dirs=erp_directions(eval_height, 2 * eval_height),
            nuisance_rng=np.random.default_rng(nuisance_seed),
            return_scene=True,
            **mask_params,
        )
        serialisable = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in params.items()
        }
        scenes.append(HeldoutScene(
            name=f"heldout{index}",
            fit_targets=torch.stack(fit_images),
            fit_directions=torch.stack(fit_dirs),
            eval_sky_mask=geometry["sky_mask"].bool(),
            nuisance_seed=nuisance_seed,
            parameters=serialisable,
        ))
    return scenes


def content_to_base(
    decoder: FrozenDecoder,
    content: torch.Tensor,
    command_norm: float,
) -> torch.Tensor:
    command = direction(*FIT_COMMANDS[0], device=content.device).unsqueeze(0)
    return decoder.latents(content, command, command_norm)[0].detach().cpu()


def detection_metrics(
    image: torch.Tensor,
    command: torch.Tensor,
    grid: torch.Tensor,
    sky_mask: torch.Tensor,
) -> dict:
    height, width = image.shape[:2]
    lum = luminance(image).reshape(1, 1, height, width)
    smooth = F.avg_pool2d(lum, 3, stride=1, padding=1).reshape(-1)
    flat_grid = grid.reshape(-1, 3)
    mask = sky_mask.reshape(-1).clone()
    # Ignore the extreme zenith row, where ERP sampling collapses many
    # azimuths into almost the same direction.
    mask &= flat_grid[:, 1] < math.sin(math.radians(88.0))
    masked = smooth.clone()
    masked[~mask] = -torch.inf
    peak_index = int(torch.argmax(masked))
    predicted = flat_grid[peak_index]
    command = F.normalize(command, dim=0)
    angular = float(torch.rad2deg(torch.acos(
        torch.clamp(predicted @ command, -1.0, 1.0))))
    pred_el = math.degrees(math.asin(float(predicted[1])))
    pred_az = math.degrees(math.atan2(
        float(predicted[0]), float(predicted[2])))
    cmd_el = math.degrees(math.asin(float(command[1])))
    cmd_az = math.degrees(math.atan2(float(command[0]), float(command[2])))

    dot = flat_grid @ command
    near = mask & (dot >= math.cos(math.radians(8.0)))
    far = mask & (dot <= math.cos(math.radians(20.0)))
    near_peak = float(smooth[near].max()) if bool(near.any()) else float("nan")
    far_peak = float(smooth[far].max()) if bool(far.any()) else float("nan")
    median = float(smooth[mask].median())
    return {
        "angle_error_deg": angular,
        "azimuth_error_deg": circular_error_deg(pred_az, cmd_az),
        "elevation_error_deg": abs(pred_el - cmd_el),
        "predicted_elevation_deg": pred_el,
        "predicted_azimuth_deg": pred_az,
        "command_peak": near_peak,
        "off_command_peak": far_peak,
        "ghost_ratio": far_peak / max(near_peak, 1e-8),
        "command_prominence": near_peak / max(median, 1e-8),
    }


def twilight_metrics(
    image: torch.Tensor,
    command_azimuth: float,
    grid: torch.Tensor,
    sky_mask: torch.Tensor,
) -> dict:
    height, width = image.shape[:2]
    lum = luminance(image).reshape(1, 1, height, width)
    smooth = F.avg_pool2d(lum, 5, stride=1, padding=2).reshape(-1)
    flat = grid.reshape(-1, 3)
    elevation = torch.asin(torch.clamp(flat[:, 1], -1.0, 1.0))
    band = sky_mask.reshape(-1) & (elevation > math.radians(-2.0)) \
        & (elevation < math.radians(15.0))
    values = smooth.clone()
    values[~band] = -torch.inf
    predicted = flat[int(torch.argmax(values))]
    predicted_azimuth = math.degrees(math.atan2(
        float(predicted[0]), float(predicted[2])))
    return {
        "azimuth_error_deg": circular_error_deg(
            predicted_azimuth, command_azimuth),
        "predicted_azimuth_deg": predicted_azimuth,
    }


def evaluate_base(
    decoder: FrozenDecoder,
    base: torch.Tensor,
    command_norm: float,
    eval_height: int,
    sky_mask: torch.Tensor | None,
) -> tuple[list[dict], list[dict], dict]:
    device = decoder.device
    grid = analytic_grid(eval_height, device=device)
    if sky_mask is None:
        mask = grid[..., 1] > math.sin(math.radians(1.0))
    else:
        mask = sky_mask.to(device)

    command_pairs = [
        (elevation, azimuth)
        for elevation in TEST_ELEVATIONS
        for azimuth in TEST_AZIMUTHS
    ]
    command_directions = torch.stack([
        direction(elevation, azimuth, device=device)
        for elevation, azimuth in command_pairs
    ])
    base = base.to(device)
    latents = base.unsqueeze(0).repeat(len(command_pairs), 1, 1)
    latents[:, decoder.channel] = command_norm * command_directions
    images = decoder.linear(latents, eval_height)
    daylight = []
    for (elevation, azimuth), command, image in zip(
            command_pairs, command_directions, images):
        record = {
            "command_elevation_deg": elevation,
            "command_azimuth_deg": azimuth,
        }
        record.update(detection_metrics(image, command, grid, mask))
        daylight.append(record)

    twilight_pairs = [
        (elevation, azimuth)
        for elevation in TWILIGHT_ELEVATIONS
        for azimuth in TWILIGHT_AZIMUTHS
    ]
    twilight_directions = torch.stack([
        direction(elevation, azimuth, device=device)
        for elevation, azimuth in twilight_pairs
    ])
    twilight_latents = base.unsqueeze(0).repeat(len(twilight_pairs), 1, 1)
    twilight_latents[:, decoder.channel] = command_norm * twilight_directions
    twilight_images = decoder.linear(twilight_latents, eval_height)
    twilight = []
    for (elevation, azimuth), image in zip(twilight_pairs, twilight_images):
        record = {
            "command_elevation_deg": elevation,
            "command_azimuth_deg": azimuth,
        }
        record.update(twilight_metrics(image, azimuth, grid, mask))
        twilight.append(record)

    path_azimuths = np.linspace(-170.0, 170.0, 35)
    path_directions = torch.stack([
        direction(40.0, float(azimuth), device=device)
        for azimuth in path_azimuths
    ])
    path_latents = base.unsqueeze(0).repeat(len(path_directions), 1, 1)
    path_latents[:, decoder.channel] = command_norm * path_directions
    path_images = decoder.linear(path_latents, eval_height)
    deltas = []
    for previous, current in zip(path_images[:-1], path_images[1:]):
        deltas.append(float(
            (current - previous).square().mean().sqrt()
            / previous.square().mean().sqrt().clamp(min=1e-8)))
    smoothness = {
        "relative_rms_delta_mean": float(np.mean(deltas)),
        "relative_rms_delta_max": float(np.max(deltas)),
        "relative_rms_delta_spike": float(
            np.max(deltas) / max(np.median(deltas), 1e-8)),
    }
    return daylight, twilight, smoothness


def summarise(records: list[dict], twilight: list[dict],
              smoothness: list[dict]) -> dict:
    summary = {
        "commands": len(records),
        "angle_error_median_deg": percentile(
            (r["angle_error_deg"] for r in records), 50),
        "angle_error_p90_deg": percentile(
            (r["angle_error_deg"] for r in records), 90),
        "azimuth_error_median_deg": percentile(
            (r["azimuth_error_deg"] for r in records), 50),
        "elevation_error_median_deg": percentile(
            (r["elevation_error_deg"] for r in records), 50),
        "ghost_ratio_median": percentile(
            (r["ghost_ratio"] for r in records), 50),
        "ghost_ratio_p90": percentile(
            (r["ghost_ratio"] for r in records), 90),
        "command_prominence_median": percentile(
            (r["command_prominence"] for r in records), 50),
        "twilight_azimuth_error_median_deg": percentile(
            (r["azimuth_error_deg"] for r in twilight), 50),
    }
    if smoothness:
        summary.update({
            "smooth_delta_max": max(
                item["relative_rms_delta_max"] for item in smoothness),
            "smooth_spike_max": max(
                item["relative_rms_delta_spike"] for item in smoothness),
        })
    return summary


def render_grid(
    decoder: FrozenDecoder,
    rows: list[tuple[str, torch.Tensor, float]],
    height: int,
    output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = decoder.device
    commands = torch.stack([
        direction(elevation, azimuth, device=device)
        for elevation, azimuth in DISPLAY_COMMANDS
    ])
    figure, axes = plt.subplots(
        len(rows), len(commands),
        figsize=(2.1 * len(commands), 1.45 * len(rows)),
        squeeze=False,
    )
    for row, (name, base, norm) in enumerate(rows):
        latents = base.to(device).unsqueeze(0).repeat(len(commands), 1, 1)
        latents[:, decoder.channel] = norm * commands
        images = decoder.linear(latents, height).cpu()
        exposure = torch.quantile(
            images.reshape(-1, 3).max(-1).values, 0.97).clamp(min=1e-6)
        display = (images / exposure).clamp(0, 1) ** (1.0 / 2.2)
        for column, ((elevation, azimuth), image) in enumerate(
                zip(DISPLAY_COMMANDS, display)):
            axis = axes[row, column]
            axis.imshow(image.numpy())
            x = (azimuth + 180.0) / 360.0 * (2 * height)
            y = (90.0 - elevation) / 180.0 * height
            axis.scatter([x], [y], s=38, facecolors="none",
                         edgecolors="white", linewidths=1.2)
            if row == 0:
                axis.set_title(f"el {elevation:.0f}, az {azimuth:+.0f}",
                               fontsize=8)
            if column == 0:
                axis.set_ylabel(name, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout(pad=0.35)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"[saved] {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/sun_control/counterfactual_control"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channel", type=int, default=9)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--fit-height", type=int, default=32)
    parser.add_argument("--eval-height", type=int, default=64)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=240723)
    parser.add_argument("--training-labels", type=Path, default=None,
                        help="Optional training sun_labels.json. When given, "
                             "also test bases formed by averaging the tied "
                             "non-sun channels within counterfactual groups.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = args.output_dir / args.label
    output.mkdir(parents=True, exist_ok=True)

    decoder = FrozenDecoder(args.run_dir, args.device, args.channel)
    print(
        f"[model] {args.label}: native ch{args.channel} norm "
        f"{decoder.native_norm:.6f}; experiment norm "
        f"{decoder.command_norm:.6f}",
        flush=True,
    )
    scenes = build_scenes(
        args.scenes, args.fit_height, args.eval_height, args.seed)

    shared_bases: dict[str, torch.Tensor] = {}
    single_bases: dict[str, torch.Tensor] = {}
    fit_stats = {"shared": {}, "single": {}}
    for scene in scenes:
        shared_content, stats = decoder.fit_content(
            scene.fit_targets,
            scene.fit_directions,
            height=args.fit_height,
            steps=args.fit_steps,
            learning_rate=args.learning_rate,
            command_norm=decoder.command_norm,
            tag=f"{scene.name}/shared",
        )
        shared_bases[scene.name] = content_to_base(
            decoder, shared_content, decoder.command_norm)
        fit_stats["shared"][scene.name] = stats

        single_content, stats = decoder.fit_content(
            scene.fit_targets[:1],
            scene.fit_directions[:1],
            height=args.fit_height,
            steps=args.fit_steps,
            learning_rate=args.learning_rate,
            command_norm=decoder.command_norm,
            tag=f"{scene.name}/single",
        )
        single_bases[scene.name] = content_to_base(
            decoder, single_content, decoder.command_norm)
        fit_stats["single"][scene.name] = stats

    base_payload = {
        "bases": shared_bases,
        "norm": decoder.command_norm,
        "label": args.label,
        "channel": args.channel,
    }
    bases_path = output / "ui_bases.pt"
    torch.save(base_payload, bases_path)
    print(f"[saved] {bases_path}", flush=True)

    base_groups: dict[str, list[tuple[str, torch.Tensor, torch.Tensor | None]]] = {
        "shared": [
            (scene.name, shared_bases[scene.name], scene.eval_sky_mask)
            for scene in scenes
        ],
        "single": [
            (scene.name, single_bases[scene.name], scene.eval_sky_mask)
            for scene in scenes
        ],
        "mean": [("mean", decoder.train_bank.mean(0).cpu(), None)],
    }
    if decoder.eval_bank is not None:
        base_groups["raw"] = [
            (f"raw{index}", decoder.eval_bank[index].detach().cpu(), None)
            for index in range(min(args.scenes, len(decoder.eval_bank)))
        ]
    training_group_stats = {}
    if args.training_labels:
        labels = json.loads(args.training_labels.read_text())
        names = sorted(labels)
        grouped: dict[str, dict[object, list[int]]] = {
            "lattice": {},
            "pair": {},
        }
        for index, name in enumerate(names):
            info = labels[name]
            if "group_id" in info:
                grouped["lattice"].setdefault(
                    info["group_id"], []).append(index)
            elif "pair_id" in info:
                grouped["pair"].setdefault(
                    info["pair_id"], []).append(index)
        for family, family_groups in grouped.items():
            if not family_groups:
                continue
            train_group_bases = []
            ordered_groups = sorted(
                family_groups.items(), key=lambda item: str(item[0]))
            for group_id, indices in ordered_groups[:args.scenes]:
                member_latents = decoder.train_bank[indices].detach()
                content = torch.cat(
                    [member_latents[:, :args.channel],
                     member_latents[:, args.channel + 1:]],
                    dim=1,
                )
                mean_content = content.mean(0)
                base = content_to_base(
                    decoder, mean_content, decoder.command_norm)
                train_group_bases.append(
                    (f"train_{family}_{group_id}", base, None))
                deviations = (content - mean_content).square().mean(
                    dim=(1, 2)).sqrt()
                training_group_stats[f"{family}:{group_id}"] = {
                    "members": len(indices),
                    "content_rms_to_group_mean": [
                        float(value) for value in deviations.cpu()
                    ],
                }
            base_groups[f"train_{family}_group"] = train_group_bases
    if abs(decoder.native_norm - decoder.command_norm) > 1e-3:
        base_groups["mean_native"] = base_groups["mean"]
        if "raw" in base_groups:
            base_groups["raw_native"] = base_groups["raw"]

    all_results = {
        "label": args.label,
        "run_dir": str(args.run_dir),
        "native_norm": decoder.native_norm,
        "command_norm": decoder.command_norm,
        "fit_stats": fit_stats,
        "training_group_stats": training_group_stats,
        "scenes": [
            {
                "name": scene.name,
                "nuisance_seed": scene.nuisance_seed,
                "parameters": scene.parameters,
            }
            for scene in scenes
        ],
        "groups": {},
    }
    for group, entries in base_groups.items():
        norm = decoder.native_norm if group.endswith("_native") \
            else decoder.command_norm
        daylight_records = []
        twilight_records = []
        smoothness_records = []
        per_base = {}
        for name, base, mask in entries:
            daylight, twilight, smoothness = evaluate_base(
                decoder, base, norm, args.eval_height, mask)
            for record in daylight:
                record["base"] = name
            for record in twilight:
                record["base"] = name
            daylight_records.extend(daylight)
            twilight_records.extend(twilight)
            smoothness_records.append(smoothness)
            per_base[name] = {
                "daylight": daylight,
                "twilight": twilight,
                "smoothness": smoothness,
            }
        summary = summarise(
            daylight_records, twilight_records, smoothness_records)
        all_results["groups"][group] = {
            "norm": norm,
            "summary": summary,
            "bases": per_base,
        }
        print(
            f"[result {group}] angle {summary['angle_error_median_deg']:.1f} "
            f"deg median / {summary['angle_error_p90_deg']:.1f} p90; "
            f"az {summary['azimuth_error_median_deg']:.1f}; "
            f"el {summary['elevation_error_median_deg']:.1f}; "
            f"ghost {summary['ghost_ratio_median']:.2f}; "
            f"twilight az {summary['twilight_azimuth_error_median_deg']:.1f}",
            flush=True,
        )

    display_rows = [
        ("shared", shared_bases[scenes[0].name], decoder.command_norm),
        ("single", single_bases[scenes[0].name], decoder.command_norm),
        ("mean", decoder.train_bank.mean(0).cpu(), decoder.command_norm),
    ]
    if decoder.eval_bank is not None:
        display_rows.append(
            ("raw eval", decoder.eval_bank[0].detach().cpu(),
             decoder.command_norm))
    render_grid(
        decoder, display_rows, args.eval_height, output / "sweep_grid.png")

    results_path = output / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"[saved] {results_path}", flush=True)


if __name__ == "__main__":
    main()
