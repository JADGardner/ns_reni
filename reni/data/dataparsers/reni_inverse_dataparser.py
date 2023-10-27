# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Union, Literal, Tuple, Optional

import imageio
import torch
import wget
import zipfile

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)


@dataclass
class RENIInverseDataParserConfig(DataParserConfig):
    """RENI dataset parser config"""

    _target: Type = field(default_factory=lambda: RENIInverseDataParser)
    """target class to instantiate"""
    data: Path = Path("data/RENI_HDR")
    """Directory specifying location of data."""
    download_data: bool = False
    """Whether to download data."""
    val_subset_size: Union[int, None] = None
    """Size of validation subset."""
    use_validation_as_train: bool = False
    """Whether to use validation set as training set."""
    specular_terms = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """Specular terms to use for training."""


@dataclass
class RENIInverseDataParser(DataParser):
    """RENI Dataparser"""

    config: RENIInverseDataParserConfig

    def __init__(self, config: RENIInverseDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    def _generate_dataparser_outputs(self, split="train"):
        if self.config.use_validation_as_train:
            split = "val"

        path = self.data / split

        # if it doesn't exist, download the data
        url = "https://www.dropbox.com/s/15gn7zlzgua7s8n/RENI_HDR.zip?dl=1"
        if not path.exists() and self.config.download_data:
            wget.download(url, out=str(self.data) + ".zip")
            with zipfile.ZipFile(str(self.data) + ".zip", "r") as zip_ref:
                zip_ref.extractall(str(self.data))
            Path(str(self.data) + ".zip").unlink()

        # get paths for all images in the directory
        image_filenames = sorted(path.glob("*.exr"))

        object_path = self.data / "3d_models"
        model_filenames = sorted(object_path.glob("*.obj"))

        if self.config.val_subset_size and split == "val":
            image_filenames = image_filenames[: self.config.val_subset_size]

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        num_images = len(image_filenames)

        cx = torch.tensor(image_width // 2, dtype=torch.float32).repeat(num_images)
        cy = torch.tensor(image_height // 2, dtype=torch.float32).repeat(num_images)
        fx = torch.tensor(image_height, dtype=torch.float32).repeat(num_images)
        fy = torch.tensor(image_height, dtype=torch.float32).repeat(num_images)

        c2w = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]], dtype=torch.float32).repeat(num_images, 1, 1)

        cameras = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.EQUIRECTANGULAR)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            metadata={
                "image_height": image_height,
                "image_width": image_width,
                "model_filenames": model_filenames,
            },
        )

        return dataparser_outputs
