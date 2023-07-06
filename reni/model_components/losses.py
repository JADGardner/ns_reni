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
Collection of Losses.
"""
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import masked_reduction, normalized_depth_scale_and_shift

import torch
from torch import nn

class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, model_output, ground_truth, sineweight):
        MSE = (((model_output - ground_truth) ** 2) * sineweight).mean(0).sum()
        return MSE


class KLD(nn.Module):
    def __init__(self, Z_dims=1):
        super(KLD, self).__init__()
        self.Z_dims = Z_dims

    def forward(self, mu, log_var):
        kld = -0.5 * ((1 + log_var - mu.pow(2) - log_var.exp()).view(mu.shape[0], -1)).sum(1)
        kld /= self.Z_dims
        kld = kld.sum(0)
        return kld
