"""Teaser figure (paper Fig: teaser).

Overlays inference outputs on publication/figures/teaser_base.png (the static
diagram backdrop): a fitted train latent decoded at 0 deg and 90 deg rotation
(with 3D latent-vector plots), the zero-latent mean environment, and five
interpolated samples.

    PYTHONPATH=. python scripts/figures/fig_teaser.py
"""

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from _common import (MODEL_DIRS, REPO_ROOT, add_common_args, decode_latents,
                     equirect_ray_bundle, load_model, make_ray_samples,
                     rotation_fn, save_figure, seed_all)

# Overlay positions on teaser_base.png (pixel coords), from the paper notebook
IMAGE_PROPERTIES = {
    "image1": {"x": 2875, "y": 260, "zoom": 3.5},
    "image2": {"x": 2875, "y": 1095, "zoom": 3.5},
    "image3": {"x": 630, "y": 660, "zoom": 1.7},
    "image4": {"x": 850, "y": 250, "zoom": 1.7},
    "image5": {"x": 180, "y": 380, "zoom": 1.7},
    "image6": {"x": 1150, "y": 700, "zoom": 1.7},
    "image7": {"x": 230, "y": 950, "zoom": 1.7},
    "image8": {"x": 900, "y": 1100, "zoom": 1.7},
    "latent_code1": {"x": 1450, "y": 260, "zoom": 0.8},
    "latent_code2": {"x": 1450, "y": 1100, "zoom": 0.8},
}


def plot_latent_vectors(latent_code, num_vectors=30):
    """3D quiver plot of latent vectors, returned as a cropped image array."""
    arr = latent_code.cpu().detach().numpy().squeeze()[:num_vectors]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    origin = np.zeros((len(arr), 3))
    colors = get_cmap("viridis")(np.linspace(0, 1, len(arr)))
    ax.quiver(origin[:, 0], origin[:, 1], origin[:, 2],
              arr[:, 0], arr[:, 1], arr[:, 2],
              color=colors, arrow_length_ratio=0.1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim([-2.0, 2.0])
    buf = io.BytesIO()
    fig.savefig(buf, format="raw")
    buf.seek(0)
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img[80:410, 157:500, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "teaser")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--latent_idx", type=int, default=200,
                        help="Train latent shown in the vector plots")
    parser.add_argument("--base_image", type=Path,
                        default=REPO_ROOT / "publication" / "figures" / "teaser_base.png")
    args = parser.parse_args()
    seed_all(args.seed)

    _, _, model = load_model(MODEL_DIRS["reni_pp"][args.latent_dim],
                             device=args.device)
    ray_bundle = equirect_ray_bundle(args.device, idx=0, height=args.height)
    ray_samples = make_ray_samples(model, ray_bundle)
    get_rotation = rotation_fn(model)

    images = {}
    for i in range(8):
        if i in (0, 1):
            # Fitted latent at 0/90 deg rotation, with its vector plot
            rot_angle = 90.0 if i == 1 else 0.0
            rotation = get_rotation(
                torch.tensor(np.deg2rad(rot_angle)).float()).to(args.device)
            z = model.field.train_mu[args.latent_idx]
            img = decode_latents(model, ray_samples, z.unsqueeze(0),
                                 rotation=rotation, height=args.height)
            z_rot = torch.matmul(rotation, z.unsqueeze(-1)).squeeze(-1)
            images[i] = {"pred_img": img,
                         "latent_code_image": plot_latent_vectors(z_rot)}
        elif i == 2:
            # Zero latent: the mean environment
            z = torch.zeros(1, model.field.latent_dim, 3, device=args.device)
            images[i] = {"pred_img": decode_latents(model, ray_samples, z,
                                                    height=args.height)}
        else:
            # Interpolated train latents as plausible random samples
            z = torch.lerp(model.field.train_mu[i].unsqueeze(0),
                           model.field.train_mu[i + 1].unsqueeze(0), 0.5)
            images[i] = {"pred_img": decode_latents(model, ray_samples,
                                                    z.to(args.device),
                                                    height=args.height)}

    base_image = plt.imread(str(args.base_image))
    h, w = base_image.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(base_image)
    ax.set_axis_off()

    for i in images:
        props = IMAGE_PROPERTIES.get(f"image{i+1}", {"x": 0, "y": 0, "zoom": 1.0})
        ab = AnnotationBbox(OffsetImage(images[i]["pred_img"], zoom=props["zoom"]),
                            (props["x"], props["y"]), frameon=False, pad=0)
        ax.add_artist(ab)
        if "latent_code_image" in images[i]:
            props = IMAGE_PROPERTIES[f"latent_code{i+1}"]
            ab = AnnotationBbox(
                OffsetImage(images[i]["latent_code_image"], zoom=props["zoom"]),
                (props["x"], props["y"]), frameon=False, pad=0)
            ax.add_artist(ab)

    save_figure(fig, args.output, svg=args.svg, dpi=dpi)


if __name__ == "__main__":
    main()
