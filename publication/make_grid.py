"""Rebuild comparison grid from precomputed individual images."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def make_white_background(img_arr: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Replace near-black background with white."""
    mask = np.all(img_arr <= threshold, axis=-1)
    out = img_arr.copy()
    out[mask] = 255
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="publication/figures_yi_et_al")
    parser.add_argument("--num_images", type=int, default=5)
    args = parser.parse_args()

    base = Path(args.input_dir)
    image_dirs = sorted(base.glob("image_*"))[:args.num_images]
    n = len(image_dirs)
    if n == 0:
        print("No image directories found")
        return

    # Fix bunny backgrounds and re-save
    for d in image_dirs:
        render_path = d / "input_render.png"
        img = np.array(Image.open(render_path))
        img = make_white_background(img)
        Image.fromarray(img).save(render_path)
        print(f"Fixed background: {render_path}")

    # Build grid: columns = Input (narrow), Yi et al., RENI++, GT
    # Bunny is 128x128 (1:1), envmaps are 64x128 (1:2).
    # Give the bunny column half the width so rows have consistent height.
    fig, axes = plt.subplots(
        n, 4, figsize=(16, 2.5 * n),
        gridspec_kw={"width_ratios": [0.5, 1, 1, 1]},
    )
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, d in enumerate(image_dirs):
        render = np.array(Image.open(d / "input_render.png"))
        yi = np.array(Image.open(d / "yi_envmap.png"))
        reni = np.array(Image.open(d / "reni_envmap.png"))
        gt = np.array(Image.open(d / "gt_envmap.png"))

        axes[row, 0].imshow(render, aspect="equal")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(yi, aspect="equal")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(reni, aspect="equal")
        axes[row, 2].axis("off")
        axes[row, 3].imshow(gt, aspect="equal")
        axes[row, 3].axis("off")

        if row == 0:
            axes[row, 0].set_title("Rendered Object", fontsize=11)
            axes[row, 1].set_title("Yi et al. (SH)", fontsize=11)
            axes[row, 2].set_title("RENI++", fontsize=11)
            axes[row, 3].set_title("Ground Truth", fontsize=11)

    plt.tight_layout()
    out_path = base / "comparison_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
