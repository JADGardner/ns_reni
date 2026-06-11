"""
Benchmark RENI++ runtime: inference speed, latent convergence time, and GPU memory.
"""
import os
import time
import torch
import functools
from pathlib import Path

from nerfstudio.engine.optimizers import Optimizers, AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from reni.configs.reni_config import RENIField
from reni.utils.utils import find_nerfstudio_project_root
from reni.utils.colourspace import linear_to_sRGB

project_root = find_nerfstudio_project_root(Path(__file__))
os.chdir(project_root)

device = "cuda:0"
torch.cuda.reset_peak_memory_stats(device)

# --- Setup ---
reni_config = RENIField
from reni.utils.checkpoint_locator import find_checkpoint
reni_config.config.load_dir = find_checkpoint("reni_plus_plus_models/latent_dim_100") / "nerfstudio_models"
reni_config.config.load_step = 50000
reni_config.config.pipeline.test_mode = "test"
reni_config.config.pipeline.model_load_strict = False
reni_config.config.vis = "tensorboard"

trainer = reni_config.config.setup(local_rank=0, world_size=1)
trainer.setup(test_mode="test")
pipeline = trainer.pipeline
datamanager = pipeline.datamanager
model = pipeline.model
model.eval()
model.fitting_eval_latents = True

mem_after_setup = torch.cuda.max_memory_allocated(device) / 1024**2
print(f"\n=== GPU Memory after model setup: {mem_after_setup:.0f} MiB ===")

# --- Benchmark single forward pass using next_eval (ray samples) ---
torch.cuda.reset_peak_memory_stats(device)
ray_bundle, batch = datamanager.next_eval(0)
num_rays = ray_bundle.origins.shape[0]
latent_code_sample = torch.zeros((num_rays, model.field.latent_dim, 3), device=device)
scale_sample = torch.ones((num_rays,), device=device)

# Warmup
with torch.no_grad():
    for _ in range(5):
        model.get_outputs(ray_bundle, rotation=None, latent_codes=latent_code_sample, scale=scale_sample)
torch.cuda.synchronize()

# Time forward passes
N_FORWARD = 50
start = time.time()
with torch.no_grad():
    for _ in range(N_FORWARD):
        model.get_outputs(ray_bundle, rotation=None, latent_codes=latent_code_sample, scale=scale_sample)
torch.cuda.synchronize()
elapsed_forward = time.time() - start

mem_inference = torch.cuda.max_memory_allocated(device) / 1024**2
print(f"\n=== Single Forward Pass (inference, {num_rays} rays) ===")
print(f"  Avg time per forward pass: {elapsed_forward / N_FORWARD * 1000:.1f} ms")
print(f"  Peak GPU memory (inference): {mem_inference:.0f} MiB")

# --- Benchmark latent code optimisation (convergence) ---
torch.cuda.reset_peak_memory_stats(device)

num_eval = len(datamanager.eval_dataset)
latent_codes = torch.nn.Parameter(torch.zeros((num_eval, model.field.latent_dim, 3), requires_grad=True, device=device))
scale = torch.nn.Parameter(torch.ones((num_eval,), requires_grad=True, device=device))

optimiser_config = {
    "latents": {
        "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=2500),
    },
}
param_group = {"latents": [latent_codes, scale]}
optimizer = Optimizers(optimiser_config, param_group)
steps = optimizer.config["latents"]["scheduler"].max_steps

print(f"\n=== Latent Code Optimisation ({steps} steps, {num_eval} eval images) ===")
start = time.time()

for step in range(steps):
    ray_bundle, batch = datamanager.next_eval(step)

    latent_code_sample = latent_codes[ray_bundle.camera_indices.squeeze()]
    scale_sample = scale[ray_bundle.camera_indices.squeeze()]

    model_outputs = model.get_outputs(ray_bundle, rotation=None, latent_codes=latent_code_sample, scale=scale_sample)
    if model.metadata.get("fit_val_in_ldr", False):
        model_outputs["rgb"] = linear_to_sRGB(model.field.unnormalise(model_outputs["rgb"]))
    loss_dict = model.get_loss_dict(model_outputs, batch, ray_bundle)
    loss = functools.reduce(torch.add, loss_dict.values())

    optimizer.zero_grad_all()
    loss.backward()
    optimizer.optimizer_step("latents")
    optimizer.scheduler_step("latents")

    if step % 500 == 0 or step == steps - 1:
        print(f"  Step {step}/{steps}  loss={loss.item():.4f}")

torch.cuda.synchronize()
elapsed_optim = time.time() - start
mem_optim = torch.cuda.max_memory_allocated(device) / 1024**2

print(f"\n=== Results ===")
print(f"  Latent optimisation total time: {elapsed_optim:.1f}s ({elapsed_optim/60:.1f} min)")
print(f"  Per-step time: {elapsed_optim/steps*1000:.1f} ms")
print(f"  Peak GPU memory (optimisation): {mem_optim:.0f} MiB")
print(f"  Number of eval images: {num_eval}")

print(f"\n=== Summary ===")
print(f"  Model setup memory:        {mem_after_setup:.0f} MiB")
print(f"  Inference memory (peak):   {mem_inference:.0f} MiB")
print(f"  Optimisation memory (peak):{mem_optim:.0f} MiB")
print(f"  Forward pass time:         {elapsed_forward / N_FORWARD * 1000:.1f} ms")
print(f"  Latent optim total:        {elapsed_optim:.1f}s for {steps} steps x {num_eval} images")
