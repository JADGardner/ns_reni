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
import pyexr

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

    exclude_batch_keys_from_device: List[str] = ["image", "mask", "normal", "albedo", "specular", "shininess"]
    cameras: Cameras

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        split: str = "train",
    ):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)
        self.split = split

        self.metadata["env_height"] = self._dataparser_outputs.metadata["env_height"]
        self.metadata["env_width"] = self._dataparser_outputs.metadata["env_width"]
        self.metadata["model_filenames"] = self._dataparser_outputs.metadata["model_filenames"]
        self.metadata["environment_maps_filenames"] = self._dataparser_outputs.metadata["environment_maps_filenames"]
        self.metadata["normal_cam_transforms"] = self._dataparser_outputs.metadata["normal_cam_transforms"]
        self.metadata["normal_map_resolution"] = self._dataparser_outputs.metadata["normal_map_resolution"]
        self.metadata["specular_terms"] = self._dataparser_outputs.metadata["specular_terms"]
        self.metadata["render_metadata"] = self._dataparser_outputs.metadata["render_metadata"]
        self.metadata["render_filenames"] = self._dataparser_outputs.metadata["render_filenames"]
        self.image_dim = self.metadata["normal_map_resolution"]

        # load all the environment maps
        self.metadata["environment_maps"] = []
        for environment_map_path in self.metadata["environment_maps_filenames"]:
            # environment_map = imageio.v2.imread(environment_map_path)
            environment_map = imageio.imread(environment_map_path).astype("float32")
            # make any inf values equal to max non inf
            environment_map[environment_map == np.inf] = np.nanmax(environment_map[environment_map != np.inf])
            # make any values less than zero equal to min non negative
            environment_map[environment_map <= 0] = np.nanmin(environment_map[environment_map > 0])
            environment_map = torch.tensor(environment_map).float()
            self.metadata["environment_maps"].append(environment_map)

        camera_rays = dataparser_outputs.cameras.generate_rays(0) # TODO currently same for all images, this might not be the case
        self.view_directions = camera_rays.directions.reshape(-1, 3) # N x 3
        self.view_directions = self.view_directions / torch.norm(self.view_directions, dim=-1, keepdim=True)

        ray_sampler_config = EquirectangularSamplerConfig(width=self.metadata["env_width"], apply_random_rotation=False, remove_lower_hemisphere=False)
        ray_sampler = ray_sampler_config.setup()
        illumination_samples = ray_sampler.generate_direction_samples()
        light_directions = illumination_samples.frustums.directions.unsqueeze(0).repeat(self.image_dim**2, 1, 1) # N x M x 3
        # ensure light directions are normalized
        self.light_directions = light_directions / torch.norm(light_directions, dim=-1, keepdim=True)
        self.blinn_phong_shader = BlinnPhongShader()

    def __len__(self):
        return len(self.metadata["render_metadata"])
    
    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        normal_filename = self._dataparser_outputs.image_filenames[image_idx]
        normals = imageio.v2.imread(normal_filename)
        normals = torch.tensor(normals).float()
        normals = F.interpolate(normals.unsqueeze(0).permute(0, 3, 1, 2), size=(self.image_dim, self.image_dim), mode='nearest').squeeze(0).permute(1, 2, 0)
        # normalise normals where the magnitude is greater than 0
        norms = torch.norm(normals, dim=-1, keepdim=True)
        mask = norms.squeeze(-1) > 0
        normals[mask] = normals[mask] / norms[mask]
        # invert y axis of normals to match nerfstudio convention
        normals[:, :, 1] = -normals[:, :, 1]
        return normals

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        data = {}

        normals = self.get_image(image_idx)
        specular_term = self.metadata["render_metadata"][image_idx]["specular_term"]

        norms = torch.norm(normals, dim=-1, keepdim=True)
        mask = norms.squeeze(-1) > 0
        mask = mask.reshape(-1)
        normals = normals.reshape(-1, 3) # N x 3

        specular = torch.ones_like(normals) * specular_term # N x 3
        albedo = 1 - specular # N x 3
        shininess = torch.ones_like(normals[:, 0]).squeeze() * self.metadata['shininess'] # N
        albedo[~mask] = 0
        specular[~mask] = 0
        shininess[~mask] = 0

        if len(self.metadata["render_filenames"]) > 0:
            rendered_image_path = self.metadata["render_filenames"][image_idx]
            rendered_image = pyexr.read(str(rendered_image_path))
            rendered_image = torch.tensor(rendered_image).float()
            # only use rgb not rgba
            rendered_image = rendered_image[:, :, :3]
        else:
            environment_map = self.metadata["environment_maps"][self.metadata["render_metadata"][image_idx]["environment_map_idx"]]
            environment_map = environment_map.reshape(-1, 3).float() # M x 3
            light_colours = environment_map.unsqueeze(0).repeat(normals.shape[0], 1, 1) # N x M x 3
            rendered_image = self.blinn_phong_shader(albedo=albedo,
                                                normals=normals,
                                                light_directions=self.light_directions,
                                                light_colors=light_colours,
                                                specular=specular,
                                                shininess=shininess,
                                                view_directions=self.view_directions,
                                                detach_normals=True)
        
            rendered_image = rendered_image.reshape(self.image_dim, self.image_dim, 3)
        
        # rendered_image = np.array(rendered_image)
        # rendered_image[rendered_image == np.inf] = np.nanmax(rendered_image[rendered_image != np.inf])
        # # make any values less than zero equal to min non negative
        # rendered_image[rendered_image <= 0] = np.nanmin(rendered_image[rendered_image > 0])
        # rendered_image = torch.tensor(rendered_image).float()
        
        data['image_idx'] = image_idx
        data['image'] = rendered_image
        data['normal'] = normals.reshape(self.image_dim, self.image_dim, 3)
        data['mask'] = (mask.reshape(self.image_dim, self.image_dim, 1))
        data['albedo'] = albedo.reshape(self.image_dim, self.image_dim, 3)
        data['specular'] = specular.reshape(self.image_dim, self.image_dim, 3)
        data['shininess'] = shininess.reshape(self.image_dim, self.image_dim)
                    
        return data