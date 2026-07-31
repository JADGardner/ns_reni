"""Animated RENI++ latent rotation using the thesis model and artwork.

The default model is the joint-frame, two-bracket, two-cycle thesis model. A
fitted latent for ``00030.exr`` is rotated once around the gravity axis. Each
frame shows the rotated latent vectors, the blue decoder artwork cropped from
``publication/figures/teaser_base.svg``, and the decoded illumination in
display-mapped and log-HDR-luminance forms.

    PYTHONPATH=.:scripts/figures python scripts/figures/fig_teaser_animation.py
"""

import argparse
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from _common import (MODEL_DIRS, REPO_ROOT, decode_latents,
                     equirect_ray_bundle, load_model, make_ray_samples,
                     rotation_fn, seed_all)
from fig_teaser import plot_latent_vectors
from nerfstudio.utils import colormaps
from reni.utils.colourspace import linear_to_sRGB
from reni.utils.tonemap import luminance, two_bracket_to_linear


DEFAULT_SVG = REPO_ROOT / "publication" / "figures" / "teaser_base.svg"
DEFAULT_OUTPUT = (
    REPO_ROOT / "publication" / "figures" / "reni_thesis_rotation.gif"
)

# Crop containing the upper decoder instance in teaser_base.svg. Coordinates
# are fractions of the rendered SVG so the crop remains stable under scaling.
ARCHITECTURE_CROP = (0.495, 0.015, 0.755, 0.385)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def _render_svg_crop(svg_path: Path, width: int = 1800) -> Image.Image:
    """Rasterise the thesis SVG and return its upper decoder artwork."""
    try:
        import cairosvg
        png = cairosvg.svg2png(url=str(svg_path), output_width=width)
    except ImportError:
        executable = shutil.which("cairosvg")
        if executable is None:
            raise RuntimeError(
                "CairoSVG is required to rasterise the thesis architecture "
                "SVG. Install the project dependencies or `pip install "
                "cairosvg`."
            )
        png = subprocess.run(
            [
                executable,
                str(svg_path),
                "--format",
                "png",
                "--output-width",
                str(width),
                "--output",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
    with Image.open(io.BytesIO(png)) as image:
        rgba = image.convert("RGBA")
    left, top, right, bottom = ARCHITECTURE_CROP
    crop = rgba.crop((
        round(left * rgba.width),
        round(top * rgba.height),
        round(right * rgba.width),
        round(bottom * rgba.height),
    ))
    white = Image.new("RGBA", crop.size, "white")
    white.alpha_composite(crop)
    return white.convert("RGB")


def _to_linear_hdr(model, image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 6:
        return two_bracket_to_linear(
            image,
            m_ldr=model.tonemap_m_ldr,
            m_log=model.tonemap_m_log,
        )
    return model.field.unnormalise(image)


def _display_image(model, image: torch.Tensor) -> torch.Tensor:
    hdr = _to_linear_hdr(model, image).detach().float().cpu()
    return linear_to_sRGB(hdr, use_quantile=True).clamp(0.0, 1.0)


def _detail_score(model, image: torch.Tensor) -> float:
    """Score a training target for visible structure after display mapping."""
    ldr = _display_image(model, image)
    luminance = (
        0.2126 * ldr[..., 0] + 0.7152 * ldr[..., 1] + 0.0722 * ldr[..., 2]
    )
    dx = torch.abs(luminance[:, 1:] - luminance[:, :-1]).mean()
    dy = torch.abs(luminance[1:, :] - luminance[:-1, :]).mean()
    contrast = torch.quantile(luminance, 0.98) - torch.quantile(luminance, 0.02)
    colour = ldr.std(dim=-1).mean()
    return float((dx + dy) * (0.5 + contrast) + 0.15 * colour)


def _rolling_hills_score(model, image: torch.Tensor) -> float:
    """Prefer blue sky over structured green terrain, while retaining detail."""
    ldr = _display_image(model, image)
    height = ldr.shape[0]
    sky = ldr[: max(1, round(0.48 * height))]
    land = ldr[round(0.42 * height): max(1, round(0.82 * height))]

    blue = (
        (sky[..., 2] > sky[..., 0] + 0.05)
        & (sky[..., 2] > sky[..., 1] + 0.01)
        & (sky[..., 2] > 0.30)
    ).float().mean()
    green = (
        (land[..., 1] > land[..., 0] + 0.02)
        & (land[..., 1] > land[..., 2] + 0.01)
        & (land[..., 1] > 0.15)
    ).float().mean()
    scenic_balance = torch.minimum(blue, green)
    return (
        _detail_score(model, image)
        + 2.0 * float(blue)
        + 1.5 * float(green)
        + 3.0 * float(scenic_balance)
    )


def select_detailed_latent(
    model,
    datamanager,
    *,
    max_candidates: int = 0,
    rank: int = 0,
    selection: str = "rolling-hills",
) -> tuple[int, list[tuple[float, int]]]:
    """Select a training image and its paired fitted latent."""
    count = min(len(datamanager.train_dataset), model.field.train_mu.shape[0])
    if max_candidates > 0 and max_candidates < count:
        indices = torch.linspace(
            0, count - 1, max_candidates, dtype=torch.long
        ).unique().tolist()
    else:
        indices = range(count)

    score_fn = (
        _rolling_hills_score if selection == "rolling-hills" else _detail_score
    )
    scores = [
        (score_fn(model, datamanager.train_dataset[index]["image"]), index)
        for index in indices
    ]
    scores.sort(reverse=True)
    if not scores:
        raise RuntimeError("The loaded checkpoint has no paired training latents")
    if rank < 0 or rank >= len(scores):
        raise ValueError(
            f"--detail-rank must be in [0, {len(scores) - 1}], got {rank}"
        )
    return scores[rank][1], scores[:10]


def latent_index_for_filename(datamanager, filename: str) -> int:
    """Resolve the first, non-mirrored training occurrence of an EXR."""
    paths = datamanager.train_dataset._dataparser_outputs.image_filenames
    matches = [index for index, path in enumerate(paths) if path.name == filename]
    if not matches:
        raise ValueError(f"No training image named {filename!r}")
    return matches[0]


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize an image to fit within a white panel without changing aspect."""
    panel = Image.new("RGB", size, "white")
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted = image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def compose_frame(
    latent_plot: np.ndarray,
    architecture: Image.Image,
    envmap: np.ndarray,
    log_hdr: np.ndarray,
    *,
    width: int,
    angle_deg: float,
) -> Image.Image:
    """Compose one polished README animation frame."""
    height = round(width * 0.285)
    margin = round(width * 0.018)
    title_h = round(height * 0.16)
    content_h = height - title_h - margin
    latent_w = round(width * 0.21)
    arch_w = round(width * 0.20)
    env_w = (width - 5 * margin - latent_w - arch_w) // 2

    frame = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(frame)
    title_font = _font(max(13, round(height * 0.055)), bold=True)
    label_font = _font(max(13, round(height * 0.046)))

    latent = Image.fromarray(latent_plot[..., :3].astype(np.uint8), "RGB")
    output = Image.fromarray(
        np.clip(envmap * 255.0, 0, 255).astype(np.uint8), "RGB"
    )
    hdr_output = Image.fromarray(
        np.clip(log_hdr * 255.0, 0, 255).astype(np.uint8), "RGB"
    )
    latent_panel = _fit_panel(latent, (latent_w, content_h))
    arch_panel = _fit_panel(architecture, (arch_w, content_h))
    output_panel = _fit_panel(output, (env_w, content_h))
    hdr_panel = _fit_panel(hdr_output, (env_w, content_h))

    x_latent = margin
    x_arch = x_latent + latent_w + margin
    x_output = x_arch + arch_w + margin
    x_hdr = x_output + env_w + margin
    y = title_h
    frame.paste(latent_panel, (x_latent, y))
    frame.paste(arch_panel, (x_arch, y))
    frame.paste(output_panel, (x_output, y))
    frame.paste(hdr_panel, (x_hdr, y))

    draw.text(
        (x_latent + latent_w // 2, margin),
        "Rotated Latent Code",
        fill="black",
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (x_latent + latent_w // 2, margin + round(title_h * 0.48)),
        f"{angle_deg:03.0f} deg around gravity",
        fill="black",
        font=label_font,
        anchor="ma",
    )
    draw.text(
        (x_arch + arch_w // 2, margin),
        "Spherical Neural Field",
        fill="black",
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (x_output + env_w // 2, margin),
        "Environment Map",
        fill="black",
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (x_hdr + env_w // 2, margin),
        "Log-HDR Luminance",
        fill="black",
        font=title_font,
        anchor="ma",
    )
    d_box = draw.textbbox((0, 0), "D", font=label_font)
    d_width = d_box[2] - d_box[0]
    d_height = d_box[3] - d_box[1]
    draw.text(
        (
            x_arch + round(arch_w * 0.095) + 2 * d_width - 3,
            y + round(content_h * 0.41) - 3 * d_height - 19,
        ),
        "D",
        fill="black",
        font=label_font,
        anchor="mm",
    )
    draw.text(
        (x_arch + round(arch_w * 0.07) - 2, y + round(content_h * 0.53) + 2),
        "Z",
        fill="black",
        font=label_font,
        anchor="mm",
    )
    return frame


def save_gif(frames: list[Image.Image], output: Path, fps: int) -> str:
    """Encode frames with a shared high-quality palette when FFmpeg exists."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        duration_ms = round(1000 / fps)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        return "Pillow per-frame palette"

    with tempfile.TemporaryDirectory(prefix="reni-animation-") as tmp:
        frame_dir = Path(tmp)
        for index, frame in enumerate(frames):
            frame.save(frame_dir / f"frame_{index:04d}.png", compress_level=1)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%04d.png"),
                "-filter_complex",
                (
                    "[0:v]split[frames][palette_input];"
                    "[palette_input]palettegen=max_colors=256:"
                    "reserve_transparent=0:stats_mode=full[palette];"
                    "[frames][palette]paletteuse=dither=sierra2_4a:"
                    "diff_mode=rectangle"
                ),
                "-loop",
                "0",
                str(output),
            ],
            check=True,
        )
    return "FFmpeg shared palette with Sierra dithering"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vnjoint_ortho_2cyc",
                        choices=sorted(MODEL_DIRS),
                        help="Model key; default is the thesis headline model")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--latent-index", type=int, default=None,
                        help="Fixed train-latent index; overrides --latent-file")
    parser.add_argument(
        "--latent-file",
        default="00030.exr",
        help="Training EXR paired with the fitted latent; use 'auto' to score",
    )
    parser.add_argument("--detail-rank", type=int, default=0,
                        help="Rank in the deterministic detail score (0 is highest)")
    parser.add_argument(
        "--selection",
        choices=("rolling-hills", "detail"),
        default="rolling-hills",
        help="Training-latent selection objective",
    )
    parser.add_argument("--detail-candidates", type=int, default=0,
                        help="Evenly sampled candidates; 0 scores the full training set")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--height", type=int, default=192,
                        help="Decoded environment-map height")
    parser.add_argument("--decode-chunk", type=int, default=32768)
    parser.add_argument("--width", type=int, default=1600,
                        help="GIF frame width")
    parser.add_argument("--num-vectors", type=int, default=40)
    parser.add_argument("--quiver-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture-svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    seed_all(args.seed)

    _, datamanager, model = load_model(
        MODEL_DIRS[args.model][args.latent_dim], device=args.device
    )
    model.eval()
    if args.latent_index is not None:
        latent_index = args.latent_index
    elif args.latent_file != "auto":
        latent_index = latent_index_for_filename(datamanager, args.latent_file)
        print(f"[latent-file] {args.latent_file} -> train index {latent_index}")
    else:
        latent_index, top = select_detailed_latent(
            model,
            datamanager,
            max_candidates=args.detail_candidates,
            rank=args.detail_rank,
            selection=args.selection,
        )
        print(f"[{args.selection}] top train latents: " + ", ".join(
            f"{index}={score:.5f}" for score, index in top
        ))
    if not 0 <= latent_index < model.field.train_mu.shape[0]:
        raise ValueError(
            f"Train latent {latent_index} is outside [0, "
            f"{model.field.train_mu.shape[0] - 1}]"
        )
    print(f"[latent] using train index {latent_index}")

    architecture = _render_svg_crop(args.architecture_svg)
    ray_bundle = equirect_ray_bundle(args.device, idx=0, height=args.height)
    ray_samples = make_ray_samples(model, ray_bundle)
    get_rotation = rotation_fn(model)
    latent = model.field.train_mu[latent_index].detach()

    frames = []
    log_range = None
    for frame_index in range(args.frames):
        angle_deg = 360.0 * frame_index / args.frames
        angle = torch.tensor(
            np.deg2rad(angle_deg), dtype=torch.float32, device=args.device
        )
        rotation = get_rotation(angle).to(args.device)
        linear = decode_latents(
            model,
            ray_samples,
            latent.unsqueeze(0),
            rotation=rotation,
            height=args.height,
            chunk_size=args.decode_chunk,
            return_linear=True,
        )
        image = linear_to_sRGB(linear, use_quantile=True).numpy()
        log_luminance = torch.log1p(luminance(linear).clamp_min(0.0)).unsqueeze(-1)
        if log_range is None:
            log_range = (
                float(torch.min(log_luminance)),
                float(torch.max(log_luminance)),
            )
        log_hdr = colormaps.apply_depth_colormap(
            log_luminance,
            near_plane=log_range[0],
            far_plane=log_range[1],
        ).numpy()
        rotated_latent = torch.matmul(
            rotation, latent.unsqueeze(-1)
        ).squeeze(-1)
        latent_plot = plot_latent_vectors(
            rotated_latent,
            num_vectors=args.num_vectors,
            scale=args.quiver_scale,
            linewidth=2.2,
        )
        frames.append(compose_frame(
            latent_plot,
            architecture,
            image,
            log_hdr,
            width=args.width,
            angle_deg=angle_deg,
        ))
        print(f"[frame] {frame_index + 1}/{args.frames}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = save_gif(frames, args.output, args.fps)
    print(
        f"[saved] {args.output} ({args.frames} frames, "
        f"{args.fps} fps, train latent {latent_index}; {encoder})"
    )


if __name__ == "__main__":
    main()
