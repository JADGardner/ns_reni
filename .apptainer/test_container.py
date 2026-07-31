#!/usr/bin/env python3
"""Verify that the ns_reni container has all required dependencies.

Run after building the SIF or Docker image:
    # Apptainer:
    .apptainer/apptainer.sh exec -- python .apptainer/test_container.py
    # Docker:
    docker compose run --rm research python .apptainer/test_container.py
"""

import subprocess
import sys

errors = []
NERFSTUDIO_COMMIT = "50e0e3c70c775e89333256213363badbf074f29d"


def check(description, fn):
    """Run a check and record pass/fail."""
    try:
        fn()
        print(f"  [PASS] {description}")
    except Exception as e:
        print(f"  [FAIL] {description}: {e}")
        errors.append(description)


def check_nerfstudio_revision():
    revision = subprocess.check_output(
        [
            "git",
            "-c",
            "safe.directory=/opt/nerfstudio",
            "-C",
            "/opt/nerfstudio",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if revision != NERFSTUDIO_COMMIT:
        raise RuntimeError(f"found {revision}, expected {NERFSTUDIO_COMMIT}")


print("=== ns_reni Container Verification ===\n")

# -- CUDA --
print("[1/4] CUDA")
check(
    "torch.cuda.is_available()",
    lambda: (_ for _ in ()).throw(RuntimeError("CUDA not available"))
    if not __import__("torch").cuda.is_available()
    else None,
)
check(
    "GPU detected",
    lambda: print(f"         GPU: {__import__('torch').cuda.get_device_name(0)}"),
)

# -- Core dependencies --
print("\n[2/4] Core Dependencies")
check("torch", lambda: __import__("torch"))
check("torchvision", lambda: __import__("torchvision"))
check("numpy", lambda: __import__("numpy"))
check("scipy", lambda: __import__("scipy"))

# -- CUDA extensions --
print("\n[3/4] CUDA Extensions")
check("tinycudann", lambda: __import__("tinycudann"))

# -- nerfstudio + ns_reni --
print("\n[4/4] nerfstudio + ns_reni")
check("nerfstudio", lambda: __import__("nerfstudio"))
check("nerfstudio revision", check_nerfstudio_revision)
check("reni", lambda: __import__("reni"))
check("einops", lambda: __import__("einops"))
check("pyexr", lambda: __import__("pyexr"))
check("roma", lambda: __import__("roma"))

# -- Summary --
print()
if errors:
    print(f"FAILED: {len(errors)} check(s) failed: {', '.join(errors)}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
