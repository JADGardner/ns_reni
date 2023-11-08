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
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.utils.dataloaders import CacheDataloader


class RENIInverseCacheDataloader(CacheDataloader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        # call super
        self.current_start_idx = 0
        self.current_end_idx = self.current_start_idx + num_images_to_sample_from if num_images_to_sample_from != -1 else len(dataset)
        super().__init__(
            dataset=dataset,
            num_images_to_sample_from=num_images_to_sample_from,
            num_times_to_repeat_images=num_times_to_repeat_images,
            device=device,
            collate_fn=collate_fn,
            exclude_batch_keys_from_device=exclude_batch_keys_from_device,
            **kwargs,
        )

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        # indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
        if self.current_start_idx >= len(self.dataset):
            self.current_start_idx = 0
            self.current_end_idx = self.current_start_idx + self.num_images_to_sample_from if self.num_images_to_sample_from != -1 else len(self.dataset)
        indices = list(range(self.current_start_idx, self.current_end_idx))
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list
    
    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                if not self.first_time:
                    self.current_start_idx += self.num_images_to_sample_from
                    self.current_end_idx += self.num_images_to_sample_from
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch