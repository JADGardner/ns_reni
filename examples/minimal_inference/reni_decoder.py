# Copyright 2026 The University of York.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PyTorch-only inference for the released thesis RENI++ decoder."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


ARTIFACT_FORMAT_VERSION = 1
MODEL_TYPE = "reni-vnjoint-ortho-so2-two-bracket"


@dataclass(frozen=True)
class ReniDecoderConfig:
    """Architecture and HDR decoding constants for the released decoder."""

    latent_dim: int = 100
    axis_of_invariance: int = 2
    hidden_features: int = 128
    num_attention_heads: int = 8
    num_attention_layers: int = 6
    out_features: int = 6
    num_frequencies: int = 2
    min_frequency_exponent: float = 0.0
    max_frequency_exponent: float = 2.0
    include_direction_input: bool = True
    m_ldr: float = 16.0
    m_log: float = 10000.0
    blend_tau: float = 0.95
    blend_delta: float = 0.02

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ReniDecoderConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VNLinear(nn.Module):
    """Linear mixing of vector-neuron channels."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...ic,oi->...oc", x, self.weight)


class VNReLU(nn.Module):
    """Vector-neuron ReLU used to predict the shared planar frame."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.U = nn.Parameter(torch.empty(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        q = torch.einsum("...ic,oi->...oc", x, self.W)
        k = torch.einsum("...ic,oi->...oc", x, self.U)
        qk = (q * k).sum(dim=-1, keepdim=True)
        k_norm = torch.sqrt((k**2).sum(dim=-1, keepdim=True).clamp(min=self.eps))
        q_projected = q - (q * (k / k_norm)).sum(dim=-1, keepdim=True) * k
        return torch.where(qk >= 0.0, q, q_projected)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        direction_input_dim: int,
        conditioning_input_dim: int,
        latent_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(direction_input_dim, latent_dim)
        self.key = nn.Linear(conditioning_input_dim, latent_dim)
        self.value = nn.Linear(conditioning_input_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size = query.size(0)
        q = (
            self.query(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attention = torch.einsum("bnqk,bnkh->bnqh", q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        output = torch.einsum("bnqh,bnhv->bnqv", attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1)
        return self.fc_out(output)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        direction_input_dim: int,
        conditioning_input_dim: int,
        latent_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(
            direction_input_dim,
            conditioning_input_dim,
            latent_dim,
            num_heads,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, directional_input: Tensor, conditioning_input: Tensor) -> Tensor:
        attention = self.mha(
            directional_input,
            conditioning_input,
            conditioning_input,
        )
        output = self.norm1(attention + directional_input)
        return self.norm2(self.fc(output) + output)


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        conditioning_input_dim: int,
        hidden_features: int,
        num_heads: int,
        num_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.residual_projection = nn.Linear(in_dim, hidden_features)
        self.layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_features,
                    conditioning_input_dim,
                    hidden_features,
                    num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_features, out_dim)

    def forward(self, x: Tensor, conditioning_input: Tensor) -> Tensor:
        x = self.residual_projection(x)
        for layer in self.layers:
            x = layer(x, conditioning_input)
        return torch.sigmoid(self.fc(x))


class ReniDecoder(nn.Module):
    """The reusable RENI++ prior without Nerfstudio or training state."""

    def __init__(self, config: ReniDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReniDecoderConfig()
        if self.config.axis_of_invariance != 2:
            raise ValueError("The released minimal decoder expects a z-up SO(2) axis")

        planar_dim = 2
        directional_dim = self.config.latent_dim + 2
        encoded_directional_dim = directional_dim * (
            2 * self.config.num_frequencies + int(self.config.include_direction_input)
        )
        conditioning_dim = self.config.latent_dim * 3

        self.vn_joint_frame = nn.Sequential(
            VNLinear(self.config.latent_dim, planar_dim),
            VNReLU(planar_dim),
        )
        self.network = Decoder(
            in_dim=encoded_directional_dim,
            conditioning_input_dim=conditioning_dim,
            hidden_features=self.config.hidden_features,
            num_heads=self.config.num_attention_heads,
            num_layers=self.config.num_attention_layers,
            out_dim=self.config.out_features,
        )

    @staticmethod
    def _orthonormalise_frame(frame: Tensor, eps: float = 1e-6) -> Tensor:
        rows = []
        for index in range(frame.shape[-2]):
            vector = frame[..., index, :]
            for unit in rows:
                vector = vector - (vector * unit).sum(-1, keepdim=True) * unit
            rows.append(vector / vector.norm(dim=-1, keepdim=True).clamp(min=eps))
        return torch.stack(rows, dim=-2)

    def _invariant_inputs(
        self, latent: Tensor, directions: Tensor
    ) -> tuple[Tensor, Tensor]:
        z_planar = latent[..., :2]
        d_planar = directions[..., :2]
        frame = self._orthonormalise_frame(self.vn_joint_frame(z_planar))
        z_planar_invariant = torch.einsum("bnc,boc->bno", z_planar, frame)
        z_axis = latent[..., 2].unsqueeze(-1)
        conditioning = torch.cat((z_planar_invariant, z_axis), dim=-1).flatten(1)
        inner_product = (z_planar * d_planar.unsqueeze(1)).sum(dim=-1)
        direction_axis = directions[..., 2].unsqueeze(-1)
        direction_planar_norm = d_planar.norm(dim=-1, keepdim=True)
        directional = torch.cat(
            (inner_product, direction_axis, direction_planar_norm), dim=-1
        )
        return directional, conditioning

    def _encode_directions(self, directions: Tensor) -> Tensor:
        config = self.config
        scaled = 2.0 * torch.pi * directions
        frequencies = 2.0 ** torch.linspace(
            config.min_frequency_exponent,
            config.max_frequency_exponent,
            config.num_frequencies,
            device=directions.device,
        )
        scaled = (scaled[..., None] * frequencies).flatten(-2)
        encoded = torch.sin(torch.cat((scaled, scaled + torch.pi / 2.0), dim=-1))
        if config.include_direction_input:
            encoded = torch.cat((encoded, directions), dim=-1)
        return encoded

    def _decode_flat(self, latent: Tensor, directions: Tensor) -> Tensor:
        directional, conditioning = self._invariant_inputs(latent, directions)
        return self.network(
            self._encode_directions(directional),
            conditioning,
        )

    @staticmethod
    def _prepare_inputs(
        latent: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor, bool]:
        latent_unbatched = latent.ndim == 2
        directions_unbatched = directions.ndim == 2
        if latent_unbatched:
            latent = latent.unsqueeze(0)
        if directions_unbatched:
            directions = directions.unsqueeze(0)
        if latent.ndim != 3 or latent.shape[-1] != 3:
            raise ValueError("latent must have shape [D, 3] or [B, D, 3]")
        if directions.ndim != 3 or directions.shape[-1] != 3:
            raise ValueError("directions must have shape [N, 3] or [B, N, 3]")
        batch = max(latent.shape[0], directions.shape[0])
        if latent.shape[0] not in (1, batch) or directions.shape[0] not in (1, batch):
            raise ValueError("latent and direction batch dimensions cannot broadcast")
        latent = latent.expand(batch, -1, -1)
        directions = directions.expand(batch, -1, -1)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return latent, directions, latent_unbatched and directions_unbatched

    def decode_brackets(
        self,
        latent: Tensor,
        directions: Tensor,
        chunk_size: int = 65536,
    ) -> Tensor:
        """Evaluate the two bounded RGB brackets at query directions."""
        latent, directions, squeeze = self._prepare_inputs(latent, directions)
        batch, num_directions = directions.shape[:2]
        chunks = []
        for start in range(0, num_directions, chunk_size):
            direction_chunk = directions[:, start : start + chunk_size]
            count = direction_chunk.shape[1]
            repeated_latent = (
                latent[:, None]
                .expand(batch, count, -1, -1)
                .reshape(batch * count, self.config.latent_dim, 3)
            )
            flat_directions = direction_chunk.reshape(batch * count, 3)
            chunks.append(
                self._decode_flat(repeated_latent, flat_directions).reshape(
                    batch, count, self.config.out_features
                )
            )
        brackets = torch.cat(chunks, dim=1)
        return brackets.squeeze(0) if squeeze else brackets

    def forward(
        self,
        latent: Tensor,
        directions: Tensor,
        chunk_size: int = 65536,
    ) -> Tensor:
        """Evaluate linear HDR RGB at query directions."""
        brackets = self.decode_brackets(latent, directions, chunk_size)
        return two_bracket_to_linear(
            brackets,
            m_ldr=self.config.m_ldr,
            m_log=self.config.m_log,
            tau=self.config.blend_tau,
            delta=self.config.blend_delta,
        )

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "ReniDecoder":
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError("Unsupported RENI decoder artifact format")
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unsupported model type: {payload.get('model_type')}")
        model = cls(ReniDecoderConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model


def inverse_reinhard_extended(tonemapped: Tensor, m_ldr: float) -> Tensor:
    discriminant = (1.0 - tonemapped) ** 2 + 4.0 * tonemapped / (m_ldr**2)
    return (
        2.0
        * tonemapped
        / ((1.0 - tonemapped) + torch.sqrt(discriminant.clamp_min(0.0)))
    )


def inverse_log_tonemap(tonemapped: Tensor, m_log: float) -> Tensor:
    return torch.expm1(tonemapped * math.log1p(m_log))


def two_bracket_to_linear(
    brackets: Tensor,
    m_ldr: float = 16.0,
    m_log: float = 10000.0,
    tau: float = 0.95,
    delta: float = 0.02,
) -> Tensor:
    """Reconstruct linear HDR from the LDR and log RGB brackets."""
    ldr = inverse_reinhard_extended(brackets[..., :3], m_ldr)
    log = inverse_log_tonemap(brackets[..., 3:6], m_log)
    weight = torch.sigmoid((ldr.max(dim=-1, keepdim=True).values - tau) / delta)
    return (1.0 - weight) * ldr + weight * log


def equirectangular_directions(
    height: int,
    width: int | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return row-major, z-up unit directions for a 2:1 ERP."""
    width = width or 2 * height
    if width != 2 * height:
        raise ValueError("The reference renderer expects a 2:1 ERP")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    coord_x = (x - width / 2.0) / height
    coord_y = -((y - height / 2.0) / height)
    theta = -torch.pi * coord_x
    phi = torch.pi * (0.5 - coord_y)
    camera_x = -torch.sin(theta) * torch.sin(phi)
    camera_y = torch.cos(phi)
    camera_z = -torch.cos(theta) * torch.sin(phi)
    world = torch.stack((camera_x, camera_z, camera_y), dim=-1)
    return world.reshape(-1, 3)
