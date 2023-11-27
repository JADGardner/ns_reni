# RENI++

### Official Nerfstudio Implementation of RENI++

Paper: RENI++: A Rotation-Equivariant, Scale-Invariant, Natural Illumination Prior

## Installation

We build on top of Nerfstudio. However, since Nerfstudio is still in very activate development with fairly large codebase changes still occurring compatibility might be an issue. Pull requests and issues are very welcome.

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create Environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

#### Install nerfstudio

```bash
conda create --name nerfstudio -y python=3.8

conda activate nerfstudio

pip install --upgrade pip

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

git clone https://github.com/nerfstudio-project/nerfstudio.git

cd nerfstudio

pip install --upgrade pip setuptools

pip install -e .
```

#### Install NeuSky

a. Clone repo and install RENI++

```bash
git clone https://github.com/JADGardner/ns_reni.git

sudo apt install libopenexr-dev

conda install -c conda-forge openexr

cd ns_reni

pip install -e .
```

b. Setup Nerfstudio CLI

```bash
ns-install-cli
```

c. Close and reopen your terminal and source conda environment again:

```bash
conda activate nerfstudio
```

## Download Data and Pretrained Models

```bash
python3 python3 scripts/download_data.py output/data/path/

python3 python3 scripts/download_models.py output/model/path/
```
