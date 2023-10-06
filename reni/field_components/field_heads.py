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
Collection of render heads
"""
from enum import Enum


class RENIFieldHeadNames(Enum):
    """Possible RENI field outputs"""

    RGB = "rgb"
    MU = "mu"
    LOG_VAR = "log_var"
    SUN_POSITION = "sun_position"
    HDR = "hdr"
    LDR = "ldr"
    SH_COEFFS = "sh_coeffs"
    ALBEDO = "albedo"
    SPECULAR = "specular"
    SHININESS = "shininess"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS = "semantics"
    PRED_NORMALS = "pred_normals"
    DENSITY = "density"
    NORMALS = "normals"
