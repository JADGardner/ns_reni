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

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
import imageio
from PIL import Image
from torch import Tensor
import scipy

from typing import Type, Union, Tuple, Dict

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from reni.utils.colourspace import linear_to_sRGB

class RENIDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, min_max: Union[Tuple[float, float], None] = None):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)
        self.min_max_normalize = self._dataparser_outputs.metadata["min_max_normalize"]

        if isinstance(self.min_max_normalize, tuple):
            min_max = self.min_max_normalize
            self.min_max_normalize = True
        elif self.min_max_normalize == "min_max":
            self.min_max_normalize = True
            print("Computing min and max values of the dataset...")
            # min_max = self.get_dataset_min_max()
            min_max = self.get_dataset_min_max()
            print(f"Min and max values of the dataset are {min_max}.")
        elif self.min_max_normalize == "quantile":
            self.min_max_normalize = True
            print("Computing min and max quantiles of the dataset...")
            # min_max = self.get_dataset_min_max()
            min_max = self.get_dataset_percentiles(0.5, 0.999)
            print(f"Min and max values of the dataset are {min_max}.")

        self.metadata["min_max"] = min_max
        self.metadata["image_height"] = self._dataparser_outputs.metadata["image_height"]
        self.metadata["image_width"] = self._dataparser_outputs.metadata["image_width"]

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        image = imageio.imread(image_filename).astype("float32")
        if self.scale_factor != 1.0:
            width, height = image.shape[1], image.shape[0]
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            # reszie
            image = image.resize(newsize, Image.ANTIALIAS)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)

        # make any inf values equal to max non inf
        image[image == np.inf] = np.nanmax(image[image != np.inf])
        # make any values less than zero equal to min non negative
        image[image <= 0] = np.nanmin(image[image > 0])
        assert np.all(np.isfinite(image)), "Image contains non finite values."
        assert np.all(image >= 0), "Image contains negative values."
        assert len(image.shape) == 3
        assert image.dtype == np.float32
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32"))
        image = image[:, :, :3] # remove alpha channel if present
        if self._dataparser_outputs.metadata["convert_to_ldr"]:
            image = linear_to_sRGB(image)
        if self._dataparser_outputs.metadata["convert_to_log_domain"]:
            image = torch.log(image + 1e-8)
        if self._dataparser_outputs.metadata["min_max_normalize"]:
            min_val, max_val = self.metadata["min_max"]
            # convert to between -1 and 1
            image = (image - min_val) / (max_val - min_val) * 2 - 1
        return image
    
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        # use top 1% of values as estimate of sun mask
        sun_mask = torch.mean(data["image"], dim=-1)
        sun_mask = sun_mask > torch.quantile(sun_mask, 0.99)
        data["sun_mask"] = sun_mask
        return data
    
    def get_dataset_min_max(self) -> Tuple[float, float]:
        """Returns the min and max values of the dataset."""

        # Initialize the min and max with the values from the first image.
        first_image = torch.from_numpy(self.get_numpy_image(0).astype("float32"))
        first_image = first_image[:, :, :3] # remove alpha channel if present

        if self._dataparser_outputs.metadata["convert_to_log_domain"]:
            first_image = torch.log(first_image + 1e-8)
        
        min_val = torch.min(first_image)
        max_val = torch.max(first_image)

        # Iterate over the rest of the images in the dataset.
        for image_idx in range(1, len(self)):
            image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32"))
            image = image[:, :, :3] # remove alpha channel if present

            if self._dataparser_outputs.metadata["convert_to_log_domain"]:
                image = torch.log(image + 1e-8)

            # Update the min and max values.
            min_val = min(min_val, torch.min(image))
            max_val = max(max_val, torch.max(image))

        return min_val.item(), max_val.item()

    def get_dataset_percentiles(self, lower_percentile=0.01, upper_percentile=0.99) -> Tuple[float, float]:
        """Returns the lower and upper percentiles of the dataset."""

        # Initialize the percentiles with the values from the first image.
        first_image = torch.from_numpy(self.get_numpy_image(0).astype("float32"))
        first_image = first_image[:, :, :3] # remove alpha channel if present

        if self._dataparser_outputs.metadata["convert_to_log_domain"]:
            first_image = torch.log(first_image + 1e-8)
        
        lower_perc = torch.quantile(first_image, lower_percentile)
        upper_perc = torch.quantile(first_image, upper_percentile)

        # Iterate over the rest of the images in the dataset.
        for image_idx in range(1, len(self)):
            image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32"))
            image = image[:, :, :3] # remove alpha channel if present

            if self._dataparser_outputs.metadata["convert_to_log_domain"]:
                image = torch.log(image + 1e-8)

            # Update the percentiles.
            lower_perc = min(lower_perc, torch.quantile(image, lower_percentile))
            upper_perc = max(upper_perc, torch.quantile(image, upper_percentile))

        return lower_perc.item(), upper_perc.item()