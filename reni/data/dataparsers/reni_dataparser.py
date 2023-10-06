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
class RENIDataParserConfig(DataParserConfig):
    """RENI dataset parser config"""

    _target: Type = field(default_factory=lambda: RENIDataParser)
    """target class to instantiate"""
    data: Path = Path("data/RENI_HDR")
    """Directory specifying location of data."""
    download_data: bool = False
    """Whether to download data."""
    convert_to_ldr: bool = False
    """Whether to convert images to LDR."""
    convert_to_log_domain: bool = False
    """Whether to convert images to log domain."""
    augment_with_mirror: bool = False
    """Whether to augment training dataset with mirror images."""
    train_subset_size: Union[int, None] = None
    """Size of training subset."""
    val_subset_size: Union[int, None] = None
    """Size of validation subset."""
    use_validation_as_train: bool = False
    """Whether to use validation set as training set."""
    min_max_normalize: Union[Literal["min_max", "quantile"], Tuple[float, float], None] = "min_max"
    """Whether to min-max normalize the images."""
    eval_mask_path: Optional[Path] = None
    """Path to the evaluation mask or a directory of masks."""
    fit_val_in_ldr: bool = False
    """Whether to fit validation images in LDR, still includes metrics against HDR versions."""


@dataclass
class RENIDataParser(DataParser):
    """RENI Dataparser"""

    config: RENIDataParserConfig

    def __init__(self, config: RENIDataParserConfig):
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

        if self.config.train_subset_size and split == "train":
            image_filenames = image_filenames[: self.config.train_subset_size]

        if self.config.val_subset_size and split == "val":
            image_filenames = image_filenames[: self.config.val_subset_size]

        if self.config.augment_with_mirror and split == "train":
            # just double the number of images and dataset will handle the mirroring
            image_filenames = image_filenames + image_filenames

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        num_images = len(image_filenames)

        mask_filenames = None
        if split in ["val", "test"] and self.config.eval_mask_path is not None:
            # if path is directory, assume it contains masks for each image
            if self.config.eval_mask_path.is_dir():
                mask_filenames = sorted(self.config.eval_mask_path.glob("*.png"))
                assert len(mask_filenames) == num_images
            else:
                mask_filenames = [self.config.eval_mask_path] * num_images

        cx = torch.tensor(image_width // 2, dtype=torch.float32).repeat(num_images)
        cy = torch.tensor(image_height // 2, dtype=torch.float32).repeat(num_images)
        fx = torch.tensor(image_height, dtype=torch.float32).repeat(num_images)
        fy = torch.tensor(image_height, dtype=torch.float32).repeat(num_images)

        c2w = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]], dtype=torch.float32).repeat(num_images, 1, 1)

        cameras = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.EQUIRECTANGULAR)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames,
            cameras=cameras,
            metadata={
                "convert_to_ldr": self.config.convert_to_ldr,
                "convert_to_log_domain": self.config.convert_to_log_domain,
                "augment_with_mirror": self.config.augment_with_mirror,
                "min_max_normalize": self.config.min_max_normalize,
                "image_height": image_height,
                "image_width": image_width,
                "fit_val_in_ldr": self.config.fit_val_in_ldr,
            },
        )

        return dataparser_outputs
