"""Download pretrained RENI++ checkpoints.

Default (no flags): the COMPLETE paper model archive — every checkpoint used
for the RENI++ paper figures and tables (RENI++ at all latent dims + masked,
nerfstudio retrains of original RENI, SH/SG fits, ablations, inverse task):

    python scripts/download_models.py

extracts to checkpoints/paper_models/ (override with a positional path).
This is the archive the figure scripts (scripts/figures/) resolve via
_common.PAPER_MODELS.

--reni-plus-plus-only downloads just the main RENI++ models (legacy zip).
"""

import argparse
import os
import sys
import zipfile

import requests

ALL_MODELS_URL = ("https://www.dropbox.com/scl/fi/oudtihiraighitknm5kbn/"
                  "all_reni_plus_plus_models_from_paper.zip"
                  "?rlkey=nlud9ihk1rempeym1ydj8rm5d&st=zhoorhfl&dl=1")
RENI_PP_ONLY_URL = ("https://www.dropbox.com/scl/fi/tw6ukedy9oc8anx0s02kk/"
                    "reni_plus_plus_models.zip"
                    "?rlkey=smr8ecsw919cokzmot7dksxgx&dl=1")


def download_file(url, dest_folder, filename):
    os.makedirs(dest_folder, exist_ok=True)
    path = os.path.join(dest_folder, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r{done / 1e6:.0f}/{total / 1e6:.0f} MB")
                    sys.stdout.flush()
    print()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="checkpoints/paper_models",
                        help="Extraction directory (default: checkpoints/paper_models)")
    parser.add_argument("--reni-plus-plus-only", action="store_true",
                        help="Download only the main RENI++ models (legacy zip)")
    args = parser.parse_args()

    url = RENI_PP_ONLY_URL if args.reni_plus_plus_only else ALL_MODELS_URL
    print(f"Downloading to {args.output} ...")
    zip_path = download_file(url, args.output, "models.zip")
    print("Unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.output)
    os.remove(zip_path)

    # The all-models zip wraps everything in an ns_reni/ folder; flatten it.
    nested = os.path.join(args.output, "ns_reni")
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            os.rename(os.path.join(nested, entry),
                      os.path.join(args.output, entry))
        os.rmdir(nested)
    print("Done. Contents:", sorted(os.listdir(args.output)))


if __name__ == "__main__":
    main()
