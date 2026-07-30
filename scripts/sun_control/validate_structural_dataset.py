"""Validate a generated structural sun-control training split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--sample-images", type=int, default=16)
    args = parser.parse_args()

    labels_path = args.split / "sun_labels.json"
    labels = json.loads(labels_path.read_text())
    files = {path.name for path in args.split.glob("*.exr")}
    labelled = set(labels)
    if files != labelled:
        missing_labels = sorted(files - labelled)[:5]
        missing_images = sorted(labelled - files)[:5]
        raise ValueError(
            f"EXR/label mismatch: {len(files)} files vs {len(labelled)} "
            f"labels; unlabelled={missing_labels}, missing={missing_images}")

    directions = np.asarray(
        [labels[name]["sun_direction"] for name in sorted(labels)],
        dtype=np.float64,
    )
    norm_error = float(np.max(np.abs(
        np.linalg.norm(directions, axis=-1) - 1.0)))
    if norm_error > 1e-5:
        raise ValueError(f"sun direction norm error is {norm_error}")

    groups = Counter()
    for name, info in labels.items():
        if "group_id" in info:
            key = ("group", info["group_id"])
        elif "pair_id" in info:
            key = ("pair", info["pair_id"])
        else:
            raise ValueError(f"{name} has no structural group identifier")
        groups[key] += 1

    group_sizes = Counter(groups.values())
    if any(kind == "pair" and groups[(kind, identifier)] != 4
           for kind, identifier in groups):
        raise ValueError("legacy V8 counterfactual groups must contain 4 images")
    if any(kind == "group" and groups[(kind, identifier)] != 60
           for kind, identifier in groups):
        raise ValueError("dense V8 lattice groups must contain 60 images")

    if args.sample_images:
        import pyexr

        ordered = sorted(files)
        indices = np.linspace(
            0, len(ordered) - 1,
            min(args.sample_images, len(ordered)),
            dtype=int,
        )
        shapes = set()
        for index in indices:
            image = pyexr.read(str(args.split / ordered[index]))
            shapes.add(tuple(image.shape))
            if not np.isfinite(image).all() or np.any(image < 0):
                raise ValueError(f"invalid pixels in {ordered[index]}")
        if len(shapes) != 1:
            raise ValueError(f"inconsistent sampled image shapes: {shapes}")
    else:
        shapes = set()

    print(json.dumps({
        "images": len(files),
        "groups": len(groups),
        "group_size_histogram": dict(sorted(group_sizes.items())),
        "max_direction_norm_error": norm_error,
        "sampled_shapes": sorted(shapes),
    }, indent=2))


if __name__ == "__main__":
    main()
