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
Optimizers class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from nerfstudio.configs import base_config
from nerfstudio.utils import writer
from nerfstudio.engine.optimizers import Optimizers


class RESGANOptimizers(Optimizers):
    """A set of optimizers.

    Args:
        config: The optimizer configuration object.
        param_groups: A dictionary of parameter groups to optimize.
    """

    def __init__(self, config: Dict[str, Any], param_groups: Dict[str, List[Parameter]]) -> None:
        super().__init__(config, param_groups)

    def optimizer_step(self, param_group_name: str, grad_scaler: Optional[GradScaler] = None) -> None:
        """Fetch and step corresponding optimizer using grad scaler if provided.

        Args:
            param_group_name: name of optimizer to step forward
            grad_scaler: GradScaler to use, if None then standard step is used
        """
        optimizer = self.optimizers[param_group_name]

        if grad_scaler:
            max_norm = self.config[param_group_name]["optimizer"].max_norm
            if max_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups, max_norm)
            if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
        else:
            optimizer.step()

    def scheduler_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding scheduler.

        Args:
            param_group_name: name of scheduler to step forward
        """
        if "scheduler" in self.config[param_group_name]:
            self.schedulers[param_group_name].step()