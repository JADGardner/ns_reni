"""RENI-specific training loops."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import writer
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter


@dataclass
class RENITrainerConfig(TrainerConfig):
    """Configuration for RENI training."""

    _target: Type = field(default_factory=lambda: RENITrainer)
    """target class to instantiate"""
    training_paradigm: Literal["standard", "latent_reset"] = "standard"
    """Training paradigm. latent_reset repeats full training passes with train latents reset to zero between passes."""
    latent_reset_cycles: int = 1
    """Number of full training passes to run when training_paradigm is latent_reset."""


class RENITrainer(Trainer):
    """Trainer with RENI-specific latent-reset training support."""

    config: RENITrainerConfig

    def train(self) -> None:
        """Train the model."""
        if self.config.training_paradigm == "standard":
            super().train()
            return

        self._train_with_latent_reset()

    def _train_with_latent_reset(self) -> None:
        """Run repeated full training passes, resetting train latents between passes."""
        if self.config.latent_reset_cycles < 1:
            raise ValueError("latent_reset_cycles must be at least 1.")

        iterations_per_cycle = self.config.max_num_iterations
        total_iterations = iterations_per_cycle * self.config.latent_reset_cycles

        assert self.pipeline.datamanager.train_dataset is not None, "Missing dataset inputs"
        if hasattr(self.pipeline.datamanager, "train_dataparser_outputs"):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(  # type: ignore
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        self.stop_training = False

        CONSOLE.print(
            "[bold cyan]Latent-reset training enabled:[/bold cyan] "
            f"{self.config.latent_reset_cycles} cycles x {iterations_per_cycle} iterations "
            f"({total_iterations} total iterations)."
        )

        if 0 < self._start_step < total_iterations and self._start_step % iterations_per_cycle == 0:
            self._reset_for_latent_cycle(next_cycle=self._start_step // iterations_per_cycle, step=self._start_step)

        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            for step in range(self._start_step, total_iterations):
                self.step = step
                if self.stop_training:
                    break
                while self.training_state == "paused":
                    if self.stop_training:
                        self._after_train()
                        return
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                if self.pipeline.datamanager.eval_dataset:
                    with self.train_lock:
                        self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

                next_step = step + 1
                if next_step < total_iterations and next_step % iterations_per_cycle == 0:
                    self._reset_for_latent_cycle(
                        next_cycle=next_step // iterations_per_cycle,
                        step=next_step,
                    )

        self._after_train()

    def _reset_for_latent_cycle(self, next_cycle: int, step: int) -> None:
        """Reset train latents and restart optimizer state for the next latent-reset cycle."""
        CONSOLE.print(
            "[bold cyan]Latent-reset cycle "
            f"{next_cycle + 1}/{self.config.latent_reset_cycles}:[/bold cyan] "
            "zeroing train latents and restarting optimizers."
        )

        self.pipeline.reset_train_latents_to_zero()
        self.optimizers = self.setup_optimizers()
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self,
            )
        )
        writer.put_scalar(name="Latent Reset Cycle", scalar=next_cycle + 1, step=step)
