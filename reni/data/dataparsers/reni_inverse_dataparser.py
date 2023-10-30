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
import numpy as np
from typing import Type, Union, Literal, Tuple, Optional

import imageio
import torch
import wget
import zipfile

from nerfstudio.utils.io import load_from_json
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
    specular_terms = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """Specular terms to use for training."""
    normal_map_resolution: int = 128
    """Resize res of normal map."""
    shininess: float = 500.0
    """Shininess of the object."""


@dataclass
class RENIInverseDataParser(DataParser):
    """RENI Dataparser"""

    config: RENIInverseDataParserConfig

    def __init__(self, config: RENIInverseDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    def _generate_dataparser_outputs(self, split="train"):
        if split == "train":
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
        environment_maps_filenames = sorted(path.glob("*.exr"))

        object_path = self.data / "3d_models"
        model_filenames = sorted(object_path.glob("*.obj"))

        normals_path = self.data / "3d_models" / "normal_maps"

        normal_cam_transforms = load_from_json(Path(normals_path / "normal_cam_transforms.json"))

        render_paths = self.data / "3d_models" / "image"
        render_filenames = sorted(render_paths.glob("*.png"))

        if self.config.val_subset_size and split == "val":
            environment_maps_filenames = environment_maps_filenames[: self.config.val_subset_size]

        img_0 = imageio.v2.imread(environment_maps_filenames[0])
        env_height, env_width = img_0.shape[:2]

        # create render metadata for each image
        render_metadata = []
        normal_filenames = []
        poses = []
        for frame in normal_cam_transforms["frames"]:
            normal_map_path = normals_path / Path(frame["file_path"])
            for env_idx, environment_map_path in enumerate(environment_maps_filenames):
                for specular_term in self.config.specular_terms:
                    poses.append(np.array(frame["transform_matrix"])) # each same normal map has same pose
                    normal_filenames.append(normal_map_path)
                    render_metadata.append({
                        "normal_map_path": normal_map_path,
                        "environment_map_path": environment_map_path,
                        "specular_term": specular_term,
                        "environment_map_idx": env_idx
                    })

        poses = np.array(poses).astype(np.float32)
        image_dim = self.config.normal_map_resolution
        camera_angle_x = float(normal_cam_transforms["camera_angle_x"])
        focal_length = 0.5 * image_dim / np.tan(0.5 * camera_angle_x)
        cx = image_dim / 2.0
        cy = image_dim / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=normal_filenames,
            cameras=cameras,
            metadata={
                "env_height": env_height,
                "env_width": env_width,
                "model_filenames": model_filenames,
                "environment_maps_filenames": environment_maps_filenames,
                "normal_cam_transforms": normal_cam_transforms,
                "normal_map_resolution": self.config.normal_map_resolution,
                "specular_terms": self.config.specular_terms,
                "shininess": self.config.shininess,
                "render_metadata": render_metadata,
                "render_filenames": render_filenames,
            },
        )

        return dataparser_outputs
