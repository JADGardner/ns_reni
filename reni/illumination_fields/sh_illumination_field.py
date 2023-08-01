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

"""RENI field"""

from typing import Literal, Type, Union, Optional, Dict, Union, Tuple, Any
from dataclasses import dataclass, field
import wget
import zipfile
import os
import contextlib
import torch.nn.functional as F

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding, Encoding

from reni.illumination_fields.base_spherical_field import SphericalField, SphericalFieldConfig
from reni.field_components.field_heads import RENIFieldHeadNames


def factorial(x):
    """
    Calculate the factorial of the given number.

    Args:
        x (int): Input number.

    Returns:
        float: Factorial of the input number.
    """
    if x == 0:
        return 1.0
    return x * factorial(x - 1)


def P(l, m, x, device):
    """
    Associated Legendre polynomial function, used in the computation of spherical harmonics.

    Args:
        l (int): Degree of the polynomial.
        m (int): Order of the polynomial.
        x (Tensor): Input tensor.
        device (torch.device): Device on which computations will be performed.

    Returns:
        Tensor: Resulting tensor after performing associated Legendre polynomial computation.
    """
    pmm = 1.0
    if m > 0:
        somx2 = torch.sqrt((1.0 - x) * (1.0 + x)).to(device)
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

    if l == m:
        return pmm * torch.ones(x.shape).to(device)

    pmmp1 = x * (2.0 * m + 1.0) * pmm

    if l == m + 1:
        return pmmp1

    pll = torch.zeros(x.shape).to(device)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def shTerms(lmax):
    """
    Calculate the number of spherical harmonics terms for a given order.

    Args:
        lmax (int): Maximum degree/order of the spherical harmonic.

    Returns:
        int: Number of spherical harmonics terms.
    """
    return (lmax + 1) * (lmax + 1)


def K(l, m, device):
    """
    Normalization constant for the spherical harmonics computation.

    Args:
        l (int): Degree of the polynomial.
        m (int): Order of the polynomial.
        device (torch.device): Device on which computations will be performed.

    Returns:
        Tensor: Normalization constant.
    """
    return torch.sqrt(
        torch.tensor(
            ((2 * l + 1) * factorial(l - m))
            / (4 * torch.pi * factorial(l + m))
        )
    ).to(device)


def shIndex(l, m):
    """
    Calculate the index in the flattened spherical harmonics array.

    Args:
        l (int): Degree of the polynomial.
        m (int): Order of the polynomial.

    Returns:
        int: Index in the flattened spherical harmonics array.
    """
    return l * l + l + m


def SH(l, m, theta, phi, device):
    """
    Calculate the spherical harmonics.

    Args:
        l (int): Degree of the polynomial.
        m (int): Order of the polynomial.
        theta (Tensor): Colatitude angle.
        phi (Tensor): Longitude angle.
        device (torch.device): Device on which computations will be performed.

    Returns:
        Tensor: Resulting tensor after performing spherical harmonics computation.
    """
    sqrt2 = np.sqrt(2.0)
    if m == 0:
        return (
            K(l, m, device)
            * P(l, m, torch.cos(theta), device)
            * torch.ones(phi.shape).to(device)
        )
    elif m > 0:
        return (
            sqrt2
            * K(l, m, device)
            * torch.cos(m * phi)
            * P(l, m, torch.cos(theta), device)
        )
    else:
        return (
            sqrt2
            * K(l, -m, device)
            * torch.sin(-m * phi)
            * P(l, -m, torch.cos(theta), device)
        )


def shEvaluate(theta, phi, lmax, device):
    """
    Evaluate spherical harmonics for given angles and maximum degree.

    Args:
        theta (Tensor): Colatitude angle [K]
        phi (Tensor): Longitude angle [K]
        lmax (int): Maximum degree/order of the spherical harmonic.
        device (torch.device): Device on which computations will be performed.

    Returns:
        Tensor: Spherical harmonics coefficients tensor.
    """
    coeffsMatrix = torch.zeros((theta.shape[0], shTerms(lmax))).to(device)

    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            coeffsMatrix[:, index] = SH(l, m, theta, phi, device)
    return coeffsMatrix


def xy2ll(x, y, width, height):
    """
    Convert from image coordinates to latitude and longitude.

    Args:
        x (Tensor): X coordinates.
        y (Tensor): Y coordinates.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        tuple: Latitude and longitude tensors.
    """
    def yLocToLat(yLoc, height):
        return yLoc / (float(height) / torch.pi)

    def xLocToLon(xLoc, width):
        return xLoc / (float(width) / (torch.pi * 2))

    return yLocToLat(y, height), xLocToLon(x, width)


def getCoefficientsMatrix(xres, lmax, device):
    """
    Compute the matrix of spherical harmonics coefficients.

    Args:
        xres (int): Resolution of the x-dimension.
        lmax (int): Maximum degree/order of the spherical harmonic.
        device (torch.device): Device on which computations will be performed.

    Returns:
        Tensor: Matrix of spherical harmonics coefficients.
    """
    yres = int(xres / 2)
    # setup fast vectorisation
    x = torch.arange(0, xres).to(device)
    y = torch.arange(0, yres).reshape(yres, 1).to(device)

    # get lat and lon of shape [H*W]
    lat, lon = xy2ll(x, y, xres, yres)
    # lat is shape [H, 1] and lon is shape [W]
    lat = lat.repeat(1, xres).reshape(-1)
    lon = lon.unsqueeze(0).repeat(yres, 1).reshape(-1)
    
    # Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
    Ylm = shEvaluate(lat, lon, lmax, device)
    return Ylm


def sh_lmax_from_terms(terms):
    """
    Calculate the maximum degree/order of the spherical harmonic from the number of terms.

    Args:
        terms (int): Number of spherical harmonics terms.

    Returns:
        int: Maximum degree/order of the spherical harmonic.
    """
    return int(torch.sqrt(terms) - 1)


def shReconstructSignal(coeffs, width, device):
    """
    Reconstruct the signal from spherical harmonics coefficients.

    Args:
        coeffs (Tensor): Spherical harmonics coefficients [N, 3]
        width (int): Width of the image.
        device (torch.device): Device on which computations will be performed.

    Returns:
        Tensor: Reconstructed signal tensor (radiance map)
    """
    lmax = sh_lmax_from_terms(torch.tensor(coeffs.shape[0]).to(device))
    sh_basis_matrix = getCoefficientsMatrix(width, lmax, device) # (H*W, N)
    sh_basis_matrix = sh_basis_matrix.reshape(width//2, width, -1) # (H, W, N)
    return torch.einsum("ijk,kl->ijl", sh_basis_matrix, coeffs)  # (H, W, 3)

def calc_num_sh_coeffs(order):
    """
    Calculate the number of spherical harmonics coefficients for a given order.

    Args:
        order (int): Order of the spherical harmonic.

    Returns:
        int: Number of spherical harmonics coefficients.
    """
    coeffs = 0
    for i in range(order + 1):
        coeffs += 2 * i + 1
    return coeffs

def get_sh_order(ndims):
    """
    Calculate the order of the spherical harmonic from the number of dimensions.

    Args:
        ndims (int): Number of dimensions.

    Returns:
        int: Order of the spherical harmonic.
    """

    order = 0
    while calc_num_sh_coeffs(order) < ndims:
        order += 1
    return order

def getSolidAngleMap(width):
    height = int(width / 2)
    solid_angle = getSolidAngle(torch.arange(0, height).float(), width)
    return solid_angle.unsqueeze(1).repeat(1, width)


def getSolidAngle(y, width, is3D=False):
    height = int(width / 2)
    pi2OverWidth = (torch.tensor(2 * np.pi) / width)
    piOverHeight = torch.tensor(np.pi) / height
    theta = (1.0 - ((y + 0.5) / height)) * torch.tensor(np.pi)
    return pi2OverWidth * (
        torch.cos(theta - (piOverHeight / 2.0)) - torch.cos(theta + (piOverHeight / 2.0))
    )


def getCoefficientsFromImage(ibl, lmax=2, resizeWidth=None, filterAmount=None, device=torch.device("cpu")):
    # Resize if necessary (I recommend it for large images)
    if resizeWidth is not None:
        ibl = ibl.permute(2, 0, 1)
        ibl = F.interpolate(ibl.unsqueeze(0), size=(resizeWidth // 2, resizeWidth), mode='bilinear', align_corners=False).squeeze(0)
        ibl = ibl.permute(1, 2, 0)
    elif ibl.shape[1] > 1000:
        ibl = ibl.permute(2, 0, 1)
        ibl = F.interpolate(ibl.unsqueeze(0), size=(1000 // 2, 1000), mode='bilinear', align_corners=False).squeeze(0)
        ibl = ibl.permute(1, 2, 0)
        
    xres = ibl.shape[1]
    yres = ibl.shape[0]

    # Pre-filtering, windowing
    if filterAmount is not None:
        ibl = blurIBL(ibl, amount=filterAmount)

    # Compute sh coefficients
    sh_basis_matrix = getCoefficientsMatrix(xres, lmax, device) # [H * W, N]
    sh_basis_matrix = sh_basis_matrix.reshape(yres, xres, -1) # [H, W, N]

    # Sampling weights
    solidAngles = getSolidAngleMap(xres)
    solidAngles = solidAngles.to(device)
    
    # Project IBL into SH basis
    nCoeffs = shTerms(lmax)
    iblCoeffs = torch.zeros((nCoeffs, 3), device=device)
    for i in range(0, shTerms(lmax)):
        iblCoeffs[i, 0] = torch.sum(ibl[:, :, 0] * sh_basis_matrix[:, :, i] * solidAngles)
        iblCoeffs[i, 1] = torch.sum(ibl[:, :, 1] * sh_basis_matrix[:, :, i] * solidAngles)
        iblCoeffs[i, 2] = torch.sum(ibl[:, :, 2] * sh_basis_matrix[:, :, i] * solidAngles)

    return iblCoeffs

def get_spherical_harmonic_representation(img, nBands):
    # img: (H, W, 3), nBands: int
    iblCoeffs = getCoefficientsFromImage(img, nBands)
    sh_radiance_map = shReconstructSignal(
        iblCoeffs, width=img.shape[1]
    )
    sh_radiance_map = torch.from_numpy(sh_radiance_map)
    return sh_radiance_map


@dataclass
class SphericalHarmonicIlluminationFieldConfig(SphericalFieldConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SphericalHarmonicIlluminationField)
    """target class to instantiate"""
    spherical_harmonic_order: int = 2
    """Spherical harmonic order"""

class SphericalHarmonicIlluminationField(SphericalField):
    """Base class for illumination fields."""

    def __init__(
        self,
        config: SphericalHarmonicIlluminationFieldConfig,
        num_train_data: int,
        num_eval_data: int,
        normalisations: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.num_train_data = num_train_data
        self.num_eval_data = num_eval_data
        self.spherical_harmonic_order = config.spherical_harmonic_order
        self.num_sh_coeffs = calc_num_sh_coeffs(self.spherical_harmonic_order)
        self.fixed_decoder = False

        self.min_max = normalisations["min_max"] if "min_max" in normalisations else None
        self.log_domain = normalisations["log_domain"] if "log_domain" in normalisations else False

        train_latent_codes = self.init_latent_codes(self.num_train_data)
        eval_latent_codes = self.init_latent_codes(self.num_eval_data)

        self.register_parameter("train_latent_codes", train_latent_codes)
        self.register_parameter("eval_latent_codes", eval_latent_codes)


    @contextlib.contextmanager
    def hold_decoder_fixed(self):
        """Context manager to fix the decoder weights

        Example usage:
        ```
        with instance_of_RENIField.hold_decoder_fixed():
            # do stuff
        ```
        """
        prev_decoder_state = self.fixed_decoder
        self.fixed_decoder = True
        try:
            yield
        finally:
            self.fixed_decoder = prev_decoder_state

    def sample_latent(self, idx):
        """Sample the latent code at a given index

        Args:
        idx (int): Index of the latent variable to sample

        Returns:
        tuple (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing the sampled spherical harmonics coefficients
        """

        if self.training and not self.fixed_decoder:
            return self.train_latent_codes[idx, :, :]
        else:
            return self.eval_latent_codes[idx, :, :]


    def init_latent_codes(self, num_latents: int):
        """Initializes the spherical harmonics coefficients
        
        """
        return torch.nn.Parameter(torch.zeros(num_latents, self.num_sh_coeffs, 3))
        
    
    def reset_eval_latents(self):
        """Resets the eval latents"""
        eval_latent_codes = self.init_latent_codes(self.num_eval_data).type_as(self.eval_latent_codes)
        self.eval_latent_codes.data = eval_latent_codes.data

    def get_outputs(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None], latent_codes: Union[torch.Tensor, None]) -> Dict[RENIFieldHeadNames, TensorType]:
        """Returns the outputs of the field.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
            latent_codes: [1, latent_dim, 3]

        Returns:
            Dict[RENIFieldHeadNames, TensorType]: A dictionary containing the outputs of the field.
        """
        # we want to batch over camera_indices as these correspond to unique latent codes
        camera_indices = ray_samples.camera_indices.squeeze() # [num_rays]

        if latent_codes is None:
            sh_coeffs = self.sample_latent(camera_indices) # [num_rays, num_sh_coeffs, 3]
        else:
            sh_coeffs = latent_codes.repeat(ray_samples.shape[0], 1, 1) # [num_rays, num_sh_coeffs, 3]

        directions = ray_samples.frustums.directions # [num_rays, 3] # each has unique latent code defined by camera index

        if rotation is not None:
            rotation = rotation.T
            directions = torch.matmul(ray_bundle.directions, rotation) # [num_rays, 3]

        # convert from cartesian to spherical coordinates with y-up convention
        theta = torch.acos(directions[:, 2]) # [num_rays]
        phi = torch.atan2(directions[:, 0], directions[:, 1]) # [num_rays]

        # evaluate spherical harmonics
        sh_basis_matrix = shEvaluate(theta, phi, self.spherical_harmonic_order, device=camera_indices.device) # [num_rays, num_sh_coeffs]

        # get radiance
        radiance = torch.einsum("ij,ijk->ik", sh_basis_matrix, sh_coeffs) # [num_rays, 3]

        outputs = {
            RENIFieldHeadNames.RGB: radiance,
            RENIFieldHeadNames.SH_COEFFS: sh_coeffs,
        }

        return outputs
    

    def forward(self, ray_samples: RaySamples, rotation: Union[torch.Tensor, None] = None, latent_codes: Union[torch.Tensor, None] = None) -> Dict[RENIFieldHeadNames, TensorType]:
        """Evaluates spherical field for a given ray bundle and rotation.

        Args:
            ray_samples: [num_rays]
            rotation: [3, 3]
            latent_codes: [1, latent_dim, 3]

        Returns:
            Dict[RENIFieldHeadNames, TensorType]: A dictionary containing the outputs of the field.
        """
        return self.get_outputs(ray_samples=ray_samples, rotation=rotation, latent_codes=latent_codes)