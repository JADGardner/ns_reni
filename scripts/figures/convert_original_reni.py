"""Convert original-repo RENI checkpoints to nerfstudio format.

The committed `checkpoints/reni_original/ndims_*/files/` dirs hold wandb dumps
from the original RENI (NeurIPS 2022) codebase:

    RENI.pt        decoder (SIREN, `net.{0..6}.linear.*`) + TRAIN latents
    RENI_Latent.pt the same decoder + latents fitted on the 21-image TEST set

The modern `reni` package reproduces the original architecture with
`old_implementation=True` (Concat conditioning, GramMatrix invariance, SO(2)
about y, no positional encoding), so conversion is a key rename plus a
minimal config.yml. Output is written to
`ndims_<D>/converted/{config.yml,nerfstudio_models/step-000000000.ckpt}`,
where scripts/figures/_common.resolve_run_dir picks it up automatically.

    PYTHONPATH=. python scripts/figures/convert_original_reni.py

Normalisation: the original prior normalised log-radiance to [-1, 1] with the
dataset min/max (-18.0536, 11.4533) (see reni/configs/reni_config.py).
"""

import argparse
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIG_ROOT = REPO_ROOT / "checkpoints" / "reni_original"
MIN_MAX = (-18.0536, 11.4533)

CONFIG_TEMPLATE = {
    "pipeline": {
        "test_mode": "test",
        "datamanager": {
            "dataparser": {
                "convert_to_ldr": False,
                "convert_to_log_domain": True,
                "eval_mask_path": None,
                "min_max_normalize": list(MIN_MAX),
                "augment_with_mirror": False,
            }
        },
        "model": {
            "loss_inclusions": {
                "log_mse_loss": True,
                "hdr_mse_loss": False,
                "ldr_mse_loss": False,
                "cosine_similarity_loss": True,
                "kld_loss": True,
                "scale_inv_loss": False,
                "scale_inv_grad_loss": False,
            },
            "field": {
                "conditioning": "Concat",
                "invariant_function": "GramMatrix",
                "equivariance": "SO2",
                "axis_of_invariance": "y",
                "positional_encoding": "None",
                "encoded_input": "None",
                "latent_dim": None,  # filled per checkpoint
                "hidden_features": 128,
                "hidden_layers": 5,
                "mapping_layers": 0,
                "mapping_features": 0,
                "num_attention_heads": 0,
                "num_attention_layers": 0,
                "output_activation": "None",
                "last_layer_linear": True,
                "trainable_scale": False,
                "old_implementation": True,
            },
        },
    }
}


def convert_one(ndims_dir: Path) -> Path:
    files = ndims_dir / "files"
    decoder_train = torch.load(files / "RENI.pt", map_location="cpu")
    test_fit = torch.load(files / "RENI_Latent.pt", map_location="cpu")

    latent_dim = decoder_train["mu"].shape[1]

    pipeline = {
        "_model.field.train_mu": decoder_train["mu"],
        "_model.field.train_logvar": decoder_train["log_var"],
        "_model.field.eval_mu": test_fit["mu"],
        "_model.field.eval_logvar": test_fit["log_var"],
        "_model.field.min_max": torch.tensor(MIN_MAX, dtype=torch.float32),
        "_model.field.log_domain": torch.tensor(True),
    }

    # Decoder: net.{0..5}.linear.* map 1:1; the final plain-Linear layer is
    # net.6.linear.* in the original and network.net.6.* in the new package.
    for k, v in decoder_train.items():
        if not k.startswith("net."):
            continue
        layer = int(k.split(".")[1])
        suffix = k.split(".")[-1]  # weight | bias
        if layer == 6:
            new_key = f"_model.field.network.net.6.{suffix}"
        else:
            new_key = f"_model.field.network.{k}"
        pipeline[new_key] = v
        # Sanity: the test-fit dump must carry the SAME decoder
        assert torch.equal(v, test_fit[k]), \
            f"{ndims_dir.name}: decoder mismatch between RENI.pt and " \
            f"RENI_Latent.pt at {k}"

    out_dir = ndims_dir / "converted"
    (out_dir / "nerfstudio_models").mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "pipeline": pipeline},
               out_dir / "nerfstudio_models" / "step-000000000.ckpt")

    config = CONFIG_TEMPLATE.copy()
    config["pipeline"]["model"]["field"]["latent_dim"] = latent_dim
    with open(out_dir / "config.yml", "w") as f:
        yaml.safe_dump(config, f)

    print(f"[converted] {ndims_dir.name}: latent_dim={latent_dim}, "
          f"train={decoder_train['mu'].shape[0]}, "
          f"eval={test_fit['mu'].shape[0]} -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[9, 20, 36, 49, 100])
    args = parser.parse_args()
    for d in args.dims:
        ndims_dir = ORIG_ROOT / f"ndims_{d}"
        if not (ndims_dir / "files" / "RENI.pt").exists():
            print(f"[skip] {ndims_dir}: no RENI.pt")
            continue
        convert_one(ndims_dir)


if __name__ == "__main__":
    main()
