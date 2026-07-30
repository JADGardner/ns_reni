# RENI++

### Official Nerfstudio Implementation of RENI++

Paper: RENI++: A Rotation-Equivariant, Scale-Invariant, Natural Illumination Prior

![NeuSky Teaser](publication/figures/reni_plus_plus_teaser.gif)

## Installation

RENI++ is a nerfstudio extension. It requires CUDA 12.8, Python 3.12, PyTorch 2.x, tiny-cuda-nn, and nerfstudio. The recommended way to run it is via Docker or Apptainer (for HPC clusters).

### Option 1: Docker (recommended for local machines)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
git clone https://github.com/JADGardner/ns_reni.git
cd ns_reni
```

**Set up data and model directories.** Either create symlinks in the project root:

```bash
ln -s /path/to/datasets data
ln -s /path/to/pretrained-models model-storage
mkdir -p outputs
```

Or set environment variables (in your shell or a `.env` file in the project root):

```bash
# .env
DATA_PATH=/path/to/datasets
MODEL_STORAGE_PATH=/path/to/pretrained-models
OUTPUTS_PATH=/path/to/outputs
```

**Build and run:**

```bash
# Build the image (compiles CUDA extensions — takes 20-40 min first time)
docker compose build research

# Start an interactive shell
docker compose run research bash

# Or train directly
docker compose run research ns-train reni --data /workspace/data/RENI_HDR
```

Inside the container, the project is mounted at `/workspace` with:
- `/workspace/data` -- datasets
- `/workspace/outputs` -- training outputs
- `/workspace/model-storage` -- pretrained checkpoints

### Option 2: Apptainer (recommended for HPC clusters)

See the `.apptainer/` directory for HPC/SLURM setup.

```bash
git clone https://github.com/JADGardner/ns_reni.git
cd ns_reni

# Configure host paths
cp .apptainer/.env.example .apptainer/.env
# Edit .apptainer/.env with your cluster's data/model/output paths

# Build the SIF image + overlay (submit as a SLURM job on HPC)
.apptainer/apptainer.sh build

# Register ns_reni code in the overlay
.apptainer/apptainer.sh install

# Interactive shell
.apptainer/apptainer.sh shell

# Run a command
.apptainer/apptainer.sh exec -- ns-train reni --help

# Verify the container
.apptainer/apptainer.sh exec -- python .apptainer/test_container.py
```

### Option 3: Manual install (conda)

```bash
git clone https://github.com/JADGardner/ns_reni.git
cd ns_reni
conda create --name reni -y python=3.12
conda activate reni
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge colmap -y
sudo apt install libopenexr-dev  # or: conda install -c conda-forge openexr
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git clone --depth 1 https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio && pip install -e . && cd ..
pip install -e .
```

#### Troubleshooting

- `-lcuda not found`
  - Solution: `ln -s {cuda directory}/lib/stubs/libcuda.so {cuda directory}/lib/libcuda.so`

## Download Data and Pretrained Models

Download the public
[RENI HDR v1.0 dataset](https://huggingface.co/datasets/jadgardner/reni-hdr):

```bash
python3 scripts/download_data.py ./data/
```

The downloader retrieves the tagged Hugging Face release, verifies its
SHA256, and extracts `./data/RENI_HDR`. GNU `tar` and `zstd` are required.
Release provenance, per-file checksums, split counts, and direct
`curl`/Hugging Face CLI instructions are recorded on the dataset page.

Download the pretrained models:

```bash
python3 scripts/download_models.py ./output/model/
```

The data and model paths can be changed in the config.
