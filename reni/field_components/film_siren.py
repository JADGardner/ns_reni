# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Siren MLP"""

import numpy as np
import torch
from torch import nn

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class CustomMappingNetwork(nn.Module):
    def __init__(self, in_features, map_hidden_layers, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = []

        for _ in range(map_hidden_layers):
            self.network.append(nn.Linear(in_features, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = map_hidden_dim

        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))

        self.network = nn.Sequential(*self.network)

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor")]
        phase_shifts = frequencies_offsets[..., torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor") :]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return torch.sin(freq * x + phase_shift)

