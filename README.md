# RENI++

### Official Nerfstudio Implementation of RENI++

Paper: RENI++: A Rotation-Equivariant, Scale-Invariant, Natural Illumination Prior

![NeuSky Teaser](publication/figures/reni_plus_plus_teaser.gif)

## Installation

We build on top of Nerfstudio. However, since Nerfstudio is still in very active development with fairly large codebase changes still occurring compatibility might be an issue. Pull requests and issues are very welcome.

#### Install

```bash
conda create --name reni++ -y python=3.11
conda activate reni++
pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c -y "nvidia/label/cuda-11.8.0" cuda-toolkit
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
sudo apt install libopenexr-dev
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
cd ..
pip install -e .
pip install numpy==1.26.4
```

## Download Data and Pretrained Models

```bash
python3 scripts/download_data.py output/data/path/

python3 scripts/download_models.py output/model/path/
```
