"""Build the hi-res figure test set from the highres mapping.

Creates data/RENI_HDR_hires_figs with test/ symlinks into RENI_HDR_512x1024
for every mapping entry with status matched/matched_exact/candidate
(candidates for the default figure indices were verified visually,
2026-07-06), low-res fallbacks for uncertain entries, and train/val
symlinked to the low-res splits (only needed for datamanager setup).

    python scripts/figures/build_hires_fig_dataset.py \
        --mapping ../../artifacts/reni_hdr_test_highres_mapping.json
"""

import argparse
import json
from pathlib import Path

ACCEPT = {"matched", "matched_exact", "candidate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path,
                        default=Path(__file__).resolve().parents[3]
                        / "artifacts" / "reni_hdr_test_highres_mapping.json")
    parser.add_argument("--lowres", type=Path,
                        default=Path("/home/james/data/RENI_HDR"))
    parser.add_argument("--hires", type=Path,
                        default=Path("/home/james/data/RENI_HDR_512x1024"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/james/data/RENI_HDR_hires_figs"))
    args = parser.parse_args()

    entries = json.loads(args.mapping.read_text())
    (args.out / "test").mkdir(parents=True, exist_ok=True)
    n_hi = n_lo = 0
    for e in entries:
        dst = args.out / "test" / e["target_file"]
        dst.unlink(missing_ok=True)
        if e["status"] in ACCEPT:
            dst.symlink_to(args.hires / e["best"]["relpath"])
            n_hi += 1
        else:
            dst.symlink_to(args.lowres / "test" / e["target_file"])
            n_lo += 1
            print(f"[fallback] {e['target_file']} ({e['status']}) -> low-res")
    for split in ("train", "val"):
        link = args.out / split
        if not link.exists():
            link.symlink_to(args.lowres / split)
    print(f"[built] {args.out}: {n_hi} hi-res, {n_lo} low-res fallbacks")


if __name__ == "__main__":
    main()
