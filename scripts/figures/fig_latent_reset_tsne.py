"""t-SNE comparison of RENI++ and latent-reset train latents.

The plot uses field.train_mu from each checkpoint, flattened to 300-D vectors
for D=100 RENI++. Points are colored by a low-frequency descriptor of the
corresponding training HDR image, so smoother color regions indicate that
nearby latents correspond to visually similar images.

Run from the ns_reni repo root:

    PYTHONPATH=. python scripts/figures/fig_latent_reset_tsne.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyexr
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from _common import MODEL_DIRS, REPO_ROOT, save_figure


DEFAULT_ORIGINAL = MODEL_DIRS["reni_pp"][100] / "nerfstudio_models" / "step-000050000.ckpt"
DEFAULT_RESTART = (
    REPO_ROOT
    / "outputs"
    / "reni_latent_reset_4_rerun"
    / "reni"
    / "2026-07-01_4cycles_rerun"
    / "nerfstudio_models"
    / "step-000200003.ckpt"
)
DEFAULT_OUTPUT = REPO_ROOT / "publication" / "figures" / "latent_reset_train_latents_tsne"


def _load_train_mu(checkpoint: Path) -> np.ndarray:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    for key in ("_model.field.train_mu", "_model.module.field.train_mu"):
        if key in state:
            latents = state[key]
            break
    else:
        raise KeyError(f"No field.train_mu in {checkpoint}")
    return latents.detach().cpu().float().numpy().reshape(latents.shape[0], -1)


def _clean_exr(path: Path) -> np.ndarray:
    image = pyexr.read(str(path)).astype("float32")
    if image.ndim == 2:
        image = image[:, :, None].repeat(3, axis=2)
    image = image[:, :, :3]
    finite = np.isfinite(image)
    finite_max = float(np.max(image[finite])) if np.any(finite) else 0.0
    image = np.nan_to_num(image, nan=0.0, posinf=finite_max, neginf=0.0)
    positive = image[image > 0]
    floor = float(np.min(positive)) if positive.size else 1e-8
    image[image <= 0] = floor
    return image


def _appearance_descriptors(data_root: Path, target_count: int, size: Tuple[int, int]) -> np.ndarray:
    train_files = sorted((data_root / "train").glob("*.exr"))
    if not train_files:
        raise FileNotFoundError(f"No train EXRs under {data_root / 'train'}")

    augmented = target_count == 2 * len(train_files)
    if target_count not in (len(train_files), 2 * len(train_files)):
        raise ValueError(
            f"Cannot map {target_count} train latents onto {len(train_files)} train EXRs "
            f"(expected N or 2N for mirror augmentation)."
        )

    descriptors = []
    for path in train_files:
        image = _clean_exr(path)
        image = np.log(image + 1e-8)
        low = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        descriptors.append(low.reshape(-1))

    descriptors = np.asarray(descriptors, dtype=np.float32)
    if augmented:
        mirrored = []
        height, width = size[1], size[0]
        for descriptor in descriptors:
            low = descriptor.reshape(height, width, 3)
            mirrored.append(low[:, ::-1, :].reshape(-1))
        descriptors = np.concatenate([descriptors, np.asarray(mirrored, dtype=np.float32)], axis=0)
    return descriptors


def _preprocess_latents(latents: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(latents)
    if pca_dim > 0 and pca_dim < scaled.shape[1]:
        scaled = PCA(n_components=pca_dim, random_state=seed).fit_transform(scaled)
    return scaled


def _tsne(latents: np.ndarray, perplexity: float, iterations: int, seed: int) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=iterations,
        random_state=seed,
        metric="euclidean",
        verbose=1,
    ).fit_transform(latents)


def _normalise_2d(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    points -= points.mean(axis=0, keepdims=True)
    scale = np.percentile(np.abs(points), 99)
    if scale > 0:
        points /= scale
    return points


def _appearance_colors(descriptors: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    desc = StandardScaler().fit_transform(descriptors)
    appearance_pca = PCA(n_components=3, random_state=seed).fit_transform(desc)
    lo = np.percentile(appearance_pca, 1, axis=0)
    hi = np.percentile(appearance_pca, 99, axis=0)
    colors = np.clip((appearance_pca - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    return colors, appearance_pca


def _mirror_pair_stats(latents: np.ndarray) -> Dict[str, float]:
    if latents.shape[0] % 2 != 0:
        return {}
    half = latents.shape[0] // 2
    distances = pairwise_distances(latents[:half], latents, metric="euclidean")
    mirror_distances = distances[np.arange(half), np.arange(half, latents.shape[0])]
    ranks = (distances < mirror_distances[:, None]).sum(axis=1)
    percentiles = ranks / (latents.shape[0] - 1)
    return {
        "mirror_pair_distance_mean": float(np.mean(mirror_distances)),
        "mirror_pair_distance_median": float(np.median(mirror_distances)),
        "mirror_pair_neighbor_rank_median": float(np.median(ranks)),
        "mirror_pair_neighbor_percentile_median": float(np.median(percentiles)),
    }


def _neighbor_metrics(latents: np.ndarray, appearance: np.ndarray, embedding: np.ndarray, k: int) -> Dict[str, float]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(latents)
    indices = nn.kneighbors(return_distance=False)[:, 1:]

    appearance_scaled = StandardScaler().fit_transform(appearance)
    appearance_norm = appearance_scaled / np.maximum(
        np.linalg.norm(appearance_scaled, axis=1, keepdims=True),
        1e-8,
    )
    local_cosine = []
    local_dist = []
    for row, neighbors in enumerate(indices):
        local_cosine.append(float(np.mean(appearance_norm[row] @ appearance_norm[neighbors].T)))
        local_dist.append(float(np.mean(np.linalg.norm(appearance_scaled[row] - appearance_scaled[neighbors], axis=1))))

    rng = np.random.default_rng(1234)
    random_indices = rng.integers(0, latents.shape[0], size=indices.shape)
    random_cosine = []
    random_dist = []
    for row, neighbors in enumerate(random_indices):
        random_cosine.append(float(np.mean(appearance_norm[row] @ appearance_norm[neighbors].T)))
        random_dist.append(float(np.mean(np.linalg.norm(appearance_scaled[row] - appearance_scaled[neighbors], axis=1))))

    metrics = {
        f"latent_{k}nn_appearance_cosine_mean": float(np.mean(local_cosine)),
        f"latent_{k}nn_appearance_l2_mean": float(np.mean(local_dist)),
        f"random_{k}nn_appearance_cosine_mean": float(np.mean(random_cosine)),
        f"random_{k}nn_appearance_l2_mean": float(np.mean(random_dist)),
        f"tsne_trustworthiness_k{k}": float(trustworthiness(latents, embedding, n_neighbors=k)),
    }
    metrics.update(_mirror_pair_stats(latents))
    return metrics


def _plot(
    original_embedding: np.ndarray,
    restart_embedding: np.ndarray,
    colors: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    output: Path,
    svg: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), constrained_layout=True)
    panels = [
        ("Original RENI++ Training Latents", original_embedding, "original"),
        ("Latent-Reset Training Latents", restart_embedding, "latent_reset"),
    ]
    for ax, (title, embedding, key) in zip(axes, panels):
        embedding = _normalise_2d(embedding)
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            s=7,
            linewidths=0,
            alpha=0.78,
            rasterized=True,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        m = metrics[key]
        ax.text(
            0.02,
            0.02,
            "\n".join(
                [
                    f"Image-kNN Cosine: {m['latent_10nn_appearance_cosine_mean']:.3f}",
                    f"Mirror-Pair Percentile: {m['mirror_pair_neighbor_percentile_median']:.3f}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 3},
        )

    save_figure(fig, output, svg=svg, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--restart", type=Path, default=DEFAULT_RESTART)
    parser.add_argument("--data", type=Path, default=Path("/home/james/data/RENI_HDR"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--perplexity", type=float, default=45.0)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--image-width", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svg", action="store_true", help="Also write SVG")
    args = parser.parse_args()

    print(f"[load] original: {args.original}")
    original = _load_train_mu(args.original)
    print(f"[load] latent reset: {args.restart}")
    restart = _load_train_mu(args.restart)
    if original.shape != restart.shape:
        raise ValueError(f"Latent banks differ in shape: {original.shape} vs {restart.shape}")
    print(f"[latents] shape={original.shape}")

    print(f"[appearance] reading {args.data / 'train'}")
    descriptors = _appearance_descriptors(
        args.data,
        target_count=original.shape[0],
        size=(args.image_width, args.image_height),
    )
    colors, appearance_pca = _appearance_colors(descriptors, args.seed)

    print("[preprocess] standardize + PCA")
    original_pre = _preprocess_latents(original, args.pca_dim, args.seed)
    restart_pre = _preprocess_latents(restart, args.pca_dim, args.seed)

    print("[tsne] original")
    original_embedding = _tsne(original_pre, args.perplexity, args.iterations, args.seed)
    print("[tsne] latent reset")
    restart_embedding = _tsne(restart_pre, args.perplexity, args.iterations, args.seed)

    print("[metrics]")
    metrics = {
        "original": _neighbor_metrics(original_pre, appearance_pca, original_embedding, k=10),
        "latent_reset": _neighbor_metrics(restart_pre, appearance_pca, restart_embedding, k=10),
    }

    output = args.output
    _plot(original_embedding, restart_embedding, colors, metrics, output, args.svg)
    payload = {
        "original": str(args.original),
        "restart": str(args.restart),
        "data": str(args.data),
        "latent_shape": list(original.shape),
        "appearance_descriptor": {
            "source": "log HDR train EXRs",
            "low_frequency_size": [args.image_height, args.image_width],
            "coloring": "RGB from first 3 PCA components of descriptors",
        },
        "tsne": {
            "perplexity": args.perplexity,
            "iterations": args.iterations,
            "pca_dim": args.pca_dim,
            "seed": args.seed,
        },
        "metrics": metrics,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[saved] {output.with_suffix('.json')}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
