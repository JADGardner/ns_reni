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

"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import pyexr

import reni.utils.exposure_helper as exphelp

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.data.datasets.base_dataset import InputDataset


class NeRDDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        fname = self._dataparser_outputs.image_filenames[image_idx]
        exr_file = pyexr.open(str(fname))
        image = exr_file.get("Image")[..., 0:3]
        mask = exr_file.get("Mask")
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)  # [1, H, W, 3]
        mask = np.expand_dims(mask, 0)  # [1, H, W, 1]
        mask = np.clip(np.array(mask).astype(np.float32), 0, 1)  # [1, H, W, 1]
        image, ev100 = exphelp.compute_auto_exp(image)  # [1, H, W, 3] [1]
        image = exphelp.linearTosRGB(image)

        if self.metadata["split"] == "train":
            white_balance = np.average(
                pyexr.open(str(fname).replace("results_train", "wb_train")).get("Image")[  # pylint: disable=W1514
                    ..., :3
                ],
                axis=(0, 1),
            )  # [3]
            white_balance = np.expand_dims(white_balance, 0)  # [1, 3]
            white_balance = white_balance * exphelp.convert_ev100_to_exp(ev100)[:, np.newaxis]  # [1, 3]
        else:
            # just create numpy array of shape 0, 3
            white_balance = np.zeros((0, 3))

        # Convert to torch
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask)
        white_balance = torch.from_numpy(white_balance)
        ev100 = torch.from_numpy(ev100)

        if self.metadata["mask_out_background"]:
            # set to background to black or white based on self.background_color
            image = image * mask + (1 - mask) * torch.tensor([1.0, 1.0, 1.0])

        # Use standard white balance if there are not enough white balance values
        difference_white_balances = image.shape[0] - white_balance.shape[0]
        base_white_balance = [0.8, 0.8, 0.8]
        additional_white_balances = torch.tensor([base_white_balance for _ in range(difference_white_balances)])
        white_balance = torch.cat([white_balance, additional_white_balances], 0)

        image = image.squeeze(0)  # [H, W, 3]
        return image

    def get_mask(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        fname = self._dataparser_outputs.image_filenames[image_idx]
        exr_file = pyexr.open(str(fname))
        mask = exr_file.get("Mask")
        mask = np.clip(np.array(mask).astype(np.float32), 0, 1)  # [H, W, 1]
        mask = torch.from_numpy(mask)

        return mask

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        mask = self.get_mask(image_idx)
        data = {
            "image_idx": image_idx,
            "image": image,
            "mask": mask,
            # "white_balance": self.white_balances[image_idx],
            # "ev100": self.ev100s[image_idx]
        }
        return data

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data
