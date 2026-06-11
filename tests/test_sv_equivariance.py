# Copyright 2025 The University of York. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Tests for rotation equivariance of the Spherical Voronoi illumination field.

Proves the mathematical property:
    f(ω; RZ) = f(R⁻¹ω; Z)

Where:
- f is the environment map value at direction ω
- Z is the latent code
- R is a rotation matrix
"""

import pytest
import torch
import torch.nn.functional as F
import math

from reni.illumination_fields.sv_illumination_field import (
    SphericalVoronoiField,
    SphericalVoronoiFieldConfig,
    VNGeometryBranch,
    VNAppearanceBranch,
)
from reni.field_components.sv_primitives import soft_spherical_voronoi_batched


def random_rotation_matrix(device=None) -> torch.Tensor:
    """Generate a random rotation matrix using QR decomposition."""
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    # Ensure it's a proper rotation (det = 1)
    Q = Q * torch.sign(torch.linalg.det(Q))
    return Q


def rotation_matrix_y(angle: float, device=None) -> torch.Tensor:
    """Generate rotation matrix around Y-axis (azimuth rotation)."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=torch.float32, device=device)


class TestGeometryBranchEquivariance:
    """Test that geometry branch is rotation-equivariant."""
    
    def test_so3_equivariance(self):
        """Φ_geo(RZ) = R · Φ_geo(Z) for any rotation R."""
        branch = VNGeometryBranch(
            latent_dim=16,
            num_sites=50,
            hidden_dim=32,
            num_layers=2,
        )
        branch.eval()
        
        # Random latent code
        z = torch.randn(4, 16, 3)  # [batch, latent_dim, 3]
        
        # Random rotation
        R = random_rotation_matrix()
        
        with torch.no_grad():
            # Option 1: Rotate input
            z_rotated = torch.einsum('ij,...j->...i', R, z)
            sites_from_rotated = branch(z_rotated)
            
            # Option 2: Rotate output
            sites_original = branch(z)
            sites_rotated = torch.einsum('ij,...j->...i', R, sites_original)
        
        # They should be equal (up to normalization)
        # Note: Both are unit-normalized, so we compare directions
        assert torch.allclose(sites_from_rotated, sites_rotated, atol=1e-5), \
            f"Max diff: {(sites_from_rotated - sites_rotated).abs().max()}"
    
    def test_so2_y_axis_equivariance(self):
        """Equivariance holds for Y-axis rotations (azimuth)."""
        branch = VNGeometryBranch(
            latent_dim=16,
            num_sites=50,
            hidden_dim=32,
            num_layers=2,
        )
        branch.eval()
        
        z = torch.randn(4, 16, 3)
        R = rotation_matrix_y(math.pi / 3)  # 60 degrees
        
        with torch.no_grad():
            z_rotated = torch.einsum('ij,...j->...i', R, z)
            sites_from_rotated = branch(z_rotated)
            
            sites_original = branch(z)
            sites_rotated = torch.einsum('ij,...j->...i', R, sites_original)
        
        assert torch.allclose(sites_from_rotated, sites_rotated, atol=1e-5)


class TestAppearanceBranchInvariance:
    """Test that appearance branch is rotation-invariant."""
    
    def test_so3_invariance(self):
        """Φ_app(RZ) = Φ_app(Z) for any rotation R."""
        branch = VNAppearanceBranch(
            latent_dim=16,
            num_sites=50,
            num_channels=3,
            hidden_dim=64,
            num_layers=2,
        )
        branch.eval()
        
        z = torch.randn(4, 16, 3)
        R = random_rotation_matrix()
        
        with torch.no_grad():
            colors_original, temps_original = branch(z)
            
            z_rotated = torch.einsum('ij,...j->...i', R, z)
            colors_rotated, temps_rotated = branch(z_rotated)
        
        # Colors and temperatures should be identical
        assert torch.allclose(colors_original, colors_rotated, atol=1e-5), \
            f"Colors max diff: {(colors_original - colors_rotated).abs().max()}"
        assert torch.allclose(temps_original, temps_rotated, atol=1e-5), \
            f"Temps max diff: {(temps_original - temps_rotated).abs().max()}"


class TestFullFieldEquivariance:
    """Test the full field satisfies f(ω; RZ) = f(R⁻¹ω; Z)."""
    
    @pytest.fixture
    def field(self):
        config = SphericalVoronoiFieldConfig(
            latent_dim=16,
            num_sites=50,
            geometry_hidden_dim=32,
            geometry_num_layers=2,
            appearance_hidden_dim=64,
            appearance_num_layers=2,
        )
        return SphericalVoronoiField(
            config=config,
            num_train_data=1,
            num_eval_data=1,
        )
    
    def test_direct_rendering_equivariance(self, field):
        """
        Test the core equivariance property using direct decode→render.
        
        f(ω; RZ) = f(R⁻¹ω; Z)
        
        This is the mathematical proof from the implementation plan.
        """
        field.eval()
        
        # Random latent code
        z = torch.randn(1, 16, 3)
        
        # Random directions to query
        directions = F.normalize(torch.randn(1, 100, 3), dim=-1)
        
        # Random rotation
        R = random_rotation_matrix()
        R_inv = R.T  # Orthogonal matrix inverse is transpose
        
        with torch.no_grad():
            # Option 1: Use rotated latent, query at ω
            z_rotated = torch.einsum('ij,...j->...i', R, z)
            sites1, colors1, temps1 = field.decode(z_rotated)
            rgb1 = field.render(directions, sites1, colors1, temps1)
            
            # Option 2: Use original latent, query at R⁻¹ω
            sites2, colors2, temps2 = field.decode(z)
            dirs_rotated = torch.einsum('ij,...j->...i', R_inv, directions)
            rgb2 = field.render(dirs_rotated, sites2, colors2, temps2)
        
        # These should be equal: f(ω; RZ) = f(R⁻¹ω; Z)
        assert torch.allclose(rgb1, rgb2, atol=1e-4), \
            f"Equivariance violated! Max diff: {(rgb1 - rgb2).abs().max()}"
    
    def test_equivariance_multiple_rotations(self, field):
        """Test equivariance holds for multiple random rotations."""
        field.eval()
        
        z = torch.randn(1, 16, 3)
        directions = F.normalize(torch.randn(1, 50, 3), dim=-1)
        
        for _ in range(10):
            R = random_rotation_matrix()
            R_inv = R.T
            
            with torch.no_grad():
                z_rotated = torch.einsum('ij,...j->...i', R, z)
                sites1, colors1, temps1 = field.decode(z_rotated)
                rgb1 = field.render(directions, sites1, colors1, temps1)
                
                sites2, colors2, temps2 = field.decode(z)
                dirs_rotated = torch.einsum('ij,...j->...i', R_inv, directions)
                rgb2 = field.render(dirs_rotated, sites2, colors2, temps2)
            
            assert torch.allclose(rgb1, rgb2, atol=1e-4)
    
    def test_azimuth_rotation_equivariance(self, field):
        """Test specifically for Y-axis (azimuth) rotations."""
        field.eval()
        
        z = torch.randn(1, 16, 3)
        directions = F.normalize(torch.randn(1, 50, 3), dim=-1)
        
        for angle in [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]:
            R = rotation_matrix_y(angle)
            R_inv = R.T
            
            with torch.no_grad():
                z_rotated = torch.einsum('ij,...j->...i', R, z)
                sites1, colors1, temps1 = field.decode(z_rotated)
                rgb1 = field.render(directions, sites1, colors1, temps1)
                
                sites2, colors2, temps2 = field.decode(z)
                dirs_rotated = torch.einsum('ij,...j->...i', R_inv, directions)
                rgb2 = field.render(dirs_rotated, sites2, colors2, temps2)
            
            assert torch.allclose(rgb1, rgb2, atol=1e-4), \
                f"Failed at angle {angle}: max diff {(rgb1 - rgb2).abs().max()}"


class TestSVFormulaEquivariance:
    """Test that the SV formula itself respects equivariance when sites rotate."""
    
    def test_sv_formula_rotation_property(self):
        """
        Prove: (Rs · ω) = (s · R⁻¹ω)
        
        This is the key property that makes SV naturally equivariant.
        """
        # Random sites and directions
        sites = F.normalize(torch.randn(10, 3), dim=-1)
        directions = F.normalize(torch.randn(100, 3), dim=-1)
        colors = torch.randn(10, 3) * 0.5 + 0.5  # Positive colors
        temps = torch.ones(10) * 10.0
        
        # Random rotation
        R = random_rotation_matrix()
        R_inv = R.T
        
        # Option 1: Rotate sites
        sites_rotated = torch.einsum('ij,kj->ki', R, sites)
        rgb1 = soft_spherical_voronoi_batched(
            directions.unsqueeze(0),
            sites_rotated.unsqueeze(0),
            colors.unsqueeze(0),
            temps.unsqueeze(0),
        )
        
        # Option 2: Counter-rotate directions
        dirs_rotated = torch.einsum('ij,kj->ki', R_inv, directions)
        rgb2 = soft_spherical_voronoi_batched(
            dirs_rotated.unsqueeze(0),
            sites.unsqueeze(0),
            colors.unsqueeze(0),
            temps.unsqueeze(0),
        )
        
        # These should be equal
        assert torch.allclose(rgb1, rgb2, atol=1e-5), \
            f"SV formula rotation property violated! Max diff: {(rgb1 - rgb2).abs().max()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
