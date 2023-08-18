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
Data parser for pre-prepared datasets for all cameras, with no additional processing needed
Optional fields - semantics, mask_filenames, cameras.distortion_params, cameras.times
"""

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Literal, Tuple

import numpy as np
import torch
import pyexr

import reni.utils.exposure_helper as exphelp

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox


def _get_camera_params(
    base_dir: Path, split: Literal["train", "val", "test"]
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Reads the camera parameters from the json file."""
    with open(base_dir / f"transforms_{split}.json", "r", encoding='utf-8') as f:
        metadata = json.load(f)

    camera_to_worlds = []

    exr_file = pyexr.open(str(base_dir / metadata["frames"][0]["file_path"])) # pylint: disable=W1514
    image = exr_file.get("Image")[..., 0:3]
    H, W = image.shape[:2]
    camera_angle_x = float(metadata["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    cx, cy, fx, fy = H / 2, W / 2, focal, focal

    intrinisics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()

    for frame in metadata["frames"]:
        # Read the poses
        camera_to_worlds.append(np.array(frame["transform_matrix"])) # 4x4
    
    # convert pose to torch
    camera_to_worlds = torch.tensor(np.array(camera_to_worlds)).float()

    return intrinisics, camera_to_worlds, camera_to_worlds.shape[0]


@dataclass
class NeRDDataParserConfig(DataParserConfig):
    """Minimal dataset config"""

    _target: Type = field(default_factory=lambda: NeRDDataParser)
    """target class to instantiate"""
    data: Path = Path("data/NeRD")
    """Directory specifying location of data."""
    scene: Literal["Car", "Chair", "Globe", "EthiopianHead", "Gnome", "GoldCape", "MotherChild", "StatueOfLiberty"] = "Car"
    """Scene name"""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "vertical"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use for centering."""
    scene_scale: float = 1.0
    """The scale of the scene."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scale_factor: float = 1.0
    """Scale factor to apply to the poses."""


@dataclass
class NeRDDataParser(DataParser):
    """NeRDDataParser"""

    config: NeRDDataParserConfig

    def __post_init__(self):
        self.scene_source = 'synthetic' if self.config.scene in ['Car', 'Chair', 'Globe'] else 'real_world'

    def _generate_dataparser_outputs(self, split="train"):
        base_dir = self.config.data / self.scene_source / self.config.scene
        
        intrinisics_train, camera_to_worlds_train, n_train = _get_camera_params(base_dir, "train")
        intrinisics_val, camera_to_worlds_val, n_val = _get_camera_params(base_dir, "val")
        intrinisics_test, camera_to_worlds_test, _ = _get_camera_params(base_dir, "test")

        camera_to_worlds = torch.cat([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], dim=0)

        # intrinisics.shape = [3, 3]
        # camera_to_worlds.shape = [N, 4, 4]

        camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))

        camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        if split == "train":
            camera_to_worlds = camera_to_worlds[:n_train]
            intrinsics = intrinisics_train
        elif split == "val":
            camera_to_worlds = camera_to_worlds[n_train : n_train + n_val]
            intrinsics = intrinisics_val
        elif split == "test":
            camera_to_worlds = camera_to_worlds[n_train + n_val :]
            intrinsics = intrinisics_test
        
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            camera_to_worlds=camera_to_worlds[:, :3, :4],
        )

        images = []
        white_balances = []
        ev100 = []
        masks = []
        image_filenames = []

        with open(base_dir / f"transforms_{split}.json", "r", encoding='utf-8') as f:
            metadata = json.load(f)

        for frame in metadata["frames"]:
            # Read the image
            fname = base_dir / frame["file_path"]
            exr_file = pyexr.open(str(fname)) # pylint: disable=W1514
            image_filenames.append(fname)
            images.append(exr_file.get("Image")[..., 0:3])
            masks.append(exr_file.get("Mask"))
            
            if split == "train":
                white_balance = np.average(
                    pyexr.open(str(fname).replace("results_train", "wb_train")).get("Image")[ # pylint: disable=W1514
                        ..., :3
                    ],
                    axis=(0, 1),
                )
                white_balances.append(white_balance)
        
        # Do the full image reconstruction pipeline
        # The EXRs are in linear color space and 0-1
        images = np.array(images).astype(np.float32) # [N, H, W, 3]
        # Start with auto exposure
        # Compute auto-exposed images and their respective EV100 values.
        images, ev100 = exphelp.compute_auto_exp(images) # [N, H, W, 3], [N]

        if split == "train":
            white_balances = (
                np.stack(white_balances, 0)
                * exphelp.convert_ev100_to_exp(ev100)[:, np.newaxis]
            ) # [N, 3]
        else:
            # just create numpy array of shape 0, 3
            white_balances = np.zeros((0, 3))

        # The images are now linear from 0 to 1
        # Convert them to sRGB
        images = exphelp.linearTosRGB(images)

        # Continue with the masks.
        # They only require values to be between 0 and 1
        # Clip to be sure
        masks = np.clip(np.array(masks).astype(np.float32), 0, 1) # [N, H, W, 1]

        # Convert to torch
        images = torch.from_numpy(images).float()
        masks = torch.from_numpy(masks)
        white_balances = torch.from_numpy(white_balances)
        ev100 = torch.from_numpy(ev100)
        
        # Use standard white balance if there are not enough white balance values
        difference_white_balances = images.shape[0] - white_balances.shape[0]
        base_white_balance = [0.8, 0.8, 0.8]
        additional_white_balances = torch.tensor(
            [base_white_balance for _ in range(difference_white_balances)]
        )
        white_balances = torch.cat([white_balances, additional_white_balances], 0)

        metadata = {
            "images": images, # [N, H, W, 3]
            "masks": masks, # [N, H, W, 1]
            "white_balances": white_balances, # [N, 3]
            "ev100s": ev100, # [N]
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
        )

        return dataparser_outputs