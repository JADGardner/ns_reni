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
import torch.nn.functional as F

from typing import Type, Union, Tuple, Dict, List

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

from reni.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni.model_components.shaders import LambertianShader, BlinnPhongShader
from reni.utils.colourspace import linear_to_sRGB

from reni.utils.colourspace import linear_to_sRGB
from reni.model_components.shaders import BlinnPhongShader


class RENIInverseDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        split: str = "train",
    ):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)
        self.split = split

        self.metadata["image_height"] = self._dataparser_outputs.metadata["image_height"]
        self.metadata["image_width"] = self._dataparser_outputs.metadata["image_width"]
        self.metadata["model_filenames"] = self._dataparser_outputs.metadata["model_filenames"]
        self.metadata["normal_filenames"] = self._dataparser_outputs.metadata["normal_filenames"]
        self.metadata["normal_cam_transforms"] = self._dataparser_outputs.metadata["normal_cam_transforms"]
        self.metadata["normal_map_resolution"] = self._dataparser_outputs.metadata["normal_map_resolution"]
        self.metadata["specular_terms"] = self._dataparser_outputs.metadata["specular_terms"]
        self.metadata["shininess"] = self._dataparser_outputs.metadata["shininess"]

        # load all the environment maps
        self.metadata["environment_maps"] = []
        for environment_map_path in self._dataparser_outputs.image_filenames:
            environment_map = imageio.v2.imread(environment_map_path)
            environment_map = torch.tensor(environment_map).float()

        # create render metadata for each image
        self.metadata["render_metadata"] = []
        for normal_map_path in self.metadata["normal_filenames"]:
            for env_idx, environment_map_path in enumerate(self._dataparser_outputs.image_filenames):
                for specular_term in self.metadata["specular_terms"]:
                    self.metadata["render_metadata"].append({
                        "normal_map_path": normal_map_path,
                        "environment_map_path": environment_map_path,
                        "specular_term": specular_term,
                        "environment_map_idx": env_idx
                    })

        poses = []
        self.image_dim = self.metadata["normal_map_resolution"]
        for frame in self.metadata["normal_cam_transforms"]["frames"]:
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        camera_angle_x = float(self.metadata["normal_cam_transforms"]["camera_angle_x"])
        focal_length = 0.5 * self.image_dim / np.tan(0.5 * camera_angle_x)
        cx = self.image_dim / 2.0
        cy = self.image_dim / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )
        camera_rays = cameras.generate_rays(0) # currently same for all images
        self.view_directions = camera_rays.directions.reshape(-1, 3) # N x 3

        ray_sampler_config = EquirectangularSamplerConfig(width=128, apply_random_rotation=False, remove_lower_hemisphere=False)
        ray_sampler = ray_sampler_config.setup()
        illumination_samples = ray_sampler.generate_direction_samples()
        light_directions = illumination_samples.frustums.directions.unsqueeze(0).repeat(self.image_dim**2, 1, 1) # N x M x 3
        # ensure light directions are normalized
        self.light_directions = light_directions / torch.norm(light_directions, dim=-1, keepdim=True)

        self.blinn_phong_shader = BlinnPhongShader()

    def __len__(self):
        return len(self.metadata["render_metadata"])

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        data = {}

        normals = imageio.v2.imread(self.metadata["render_metadata"][image_idx]["normal_map_path"])
        environment_map = imageio.v2.imread(self.metadata["render_metadata"][image_idx]["environment_map_path"])
        specular_term = self.metadata["render_metadata"][image_idx]["specular_term"]

        normals = torch.tensor(normals).float()
        normals = F.interpolate(normals.unsqueeze(0).permute(0, 3, 1, 2), size=(self.image_dim, self.image_dim), mode='nearest').squeeze(0).permute(1, 2, 0)
        # normalise normals where the magnitude is greater than 0
        norms = torch.norm(normals, dim=-1, keepdim=True)
        mask = norms.squeeze(-1) > 0
        normals[mask] = normals[mask] / norms[mask]
        mask = mask.reshape(-1)
        # invert y axis of normals to match nerfstudio convention
        normals[:, :, 1] = -normals[:, :, 1]
        normals = normals.reshape(-1, 3) # N x 3
        env_map_image_height, env_map_image_width = environment_map.shape[:2] # 64x128
        environment_map = torch.tensor(environment_map).reshape(-1, 3).float() # M x 3
        light_colours = environment_map.unsqueeze(0).repeat(normals.shape[0], 1, 1) # N x M x 3
        specular = torch.ones_like(normals) * specular_term # N x 3
        albedo = 1 - specular # N x 3
        shiniess = torch.ones_like(normals[:, 0]).squeeze() * self.metadata['shininess'] # N
        albedo[~mask] = 0
        specular[~mask] = 0
        shiniess[~mask] = 0

        rendered_image = self.blinn_phong_shader(albedo=albedo,
                                            normals=normals,
                                            light_directions=self.light_directions,
                                            light_colors=light_colours,
                                            specular=specular,
                                            shininess=shiniess,
                                            view_directions=self.view_directions,
                                            detach_normals=True)
        
        rendered_image = rendered_image.reshape(self.image_dim, self.image_dim, 3)
        rendered_image = linear_to_sRGB(rendered_image, use_quantile=True)

        data['image'] = rendered_image
        data['mask'] = (mask.reshape(self.image_dim, self.image_dim))
                    
        return data