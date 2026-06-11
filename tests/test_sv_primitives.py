# Copyright 2025 The University of York. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Unit tests for Spherical Voronoi primitives.
"""

import pytest
import torch
import torch.nn.functional as F
import math

from reni.field_components.sv_primitives import (
    fibonacci_sphere_sites,
    soft_spherical_voronoi,
    soft_spherical_voronoi_batched,
    SphericalVoronoiRepresentation,
    fit_sv_to_envmap,
)


class TestFibonacciSphere:
    """Tests for Fibonacci sphere site initialization."""
    
    def test_correct_shape(self):
        """Sites should have shape [K, 3]."""
        sites = fibonacci_sphere_sites(100)
        assert sites.shape == (100, 3)
    
    def test_unit_norm(self):
        """All sites should be on unit sphere."""
        sites = fibonacci_sphere_sites(100)
        norms = torch.norm(sites, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_reasonable_distribution(self):
        """Sites should be roughly uniformly distributed (mean near origin)."""
        sites = fibonacci_sphere_sites(1000)
        mean = sites.mean(dim=0)
        # Mean should be close to [0, 0, 0] for uniform distribution
        assert torch.allclose(mean, torch.zeros(3), atol=0.1)
    
    def test_device_placement(self):
        """Sites should be on specified device."""
        if torch.cuda.is_available():
            sites = fibonacci_sphere_sites(100, device=torch.device('cuda'))
            assert sites.device.type == 'cuda'


class TestSoftSphericalVoronoi:
    """Tests for the core SV formula."""
    
    @pytest.fixture
    def simple_setup(self):
        """Create simple test case with 4 cardinal sites."""
        sites = torch.tensor([
            [1., 0., 0.],   # +X
            [-1., 0., 0.],  # -X
            [0., 1., 0.],   # +Y
            [0., 0., 1.],   # +Z
        ])
        colors = torch.tensor([
            [1., 0., 0.],  # Red
            [0., 1., 0.],  # Green
            [0., 0., 1.],  # Blue
            [1., 1., 1.],  # White
        ])
        temperatures = torch.tensor([10., 10., 10., 10.])
        return sites, colors, temperatures
    
    def test_output_shape(self, simple_setup):
        """Output should match direction shape with color channels."""
        sites, colors, temperatures = simple_setup
        directions = torch.randn(32, 64, 3)
        directions = F.normalize(directions, dim=-1)
        
        output = soft_spherical_voronoi(directions, sites, colors, temperatures)
        assert output.shape == (32, 64, 3)
    
    def test_gradient_flow_sites(self, simple_setup):
        """Gradients should flow to sites."""
        sites, colors, temperatures = simple_setup
        sites.requires_grad_(True)
        
        directions = F.normalize(torch.randn(10, 3), dim=-1)
        output = soft_spherical_voronoi(directions, sites, colors, temperatures)
        loss = output.sum()
        loss.backward()
        
        assert sites.grad is not None
        assert not torch.all(sites.grad == 0)
    
    def test_gradient_flow_colors(self, simple_setup):
        """Gradients should flow to colors."""
        sites, colors, temperatures = simple_setup
        colors.requires_grad_(True)
        
        directions = F.normalize(torch.randn(10, 3), dim=-1)
        output = soft_spherical_voronoi(directions, sites, colors, temperatures)
        loss = output.sum()
        loss.backward()
        
        assert colors.grad is not None
        assert not torch.all(colors.grad == 0)
    
    def test_gradient_flow_temperatures(self, simple_setup):
        """Gradients should flow to temperatures."""
        sites, colors, temperatures = simple_setup
        temperatures.requires_grad_(True)
        
        directions = F.normalize(torch.randn(10, 3), dim=-1)
        output = soft_spherical_voronoi(directions, sites, colors, temperatures)
        loss = output.sum()
        loss.backward()
        
        assert temperatures.grad is not None
        assert not torch.all(temperatures.grad == 0)
    
    def test_high_temperature_is_hard_voronoi(self, simple_setup):
        """Very high temperature should approximate hard Voronoi."""
        sites, colors, _ = simple_setup
        high_temp = torch.tensor([1000., 1000., 1000., 1000.])
        
        # Query direction close to +X site
        direction = torch.tensor([[0.99, 0.1, 0.1]])
        direction = F.normalize(direction, dim=-1)
        
        output = soft_spherical_voronoi(direction, sites, colors, high_temp)
        
        # Should be almost exactly red (color of +X site)
        assert torch.allclose(output, torch.tensor([[1., 0., 0.]]), atol=0.01)
    
    def test_low_temperature_is_smooth(self, simple_setup):
        """Low temperature should give smooth interpolation."""
        sites, colors, _ = simple_setup
        low_temp = torch.tensor([0.1, 0.1, 0.1, 0.1])
        
        direction = torch.tensor([[1., 0., 0.]])  # Exactly at +X site
        direction = F.normalize(direction, dim=-1)
        
        output = soft_spherical_voronoi(direction, sites, colors, low_temp)
        
        # Should NOT be exactly red - should be mixed
        assert not torch.allclose(output, torch.tensor([[1., 0., 0.]]), atol=0.1)
    
    def test_numerical_stability_extreme_temperature(self):
        """Should not overflow/NaN with extreme temperatures."""
        sites = fibonacci_sphere_sites(10)
        colors = torch.randn(10, 3)
        temperatures = torch.full((10,), 1000.)  # Very high
        
        directions = F.normalize(torch.randn(100, 3), dim=-1)
        output = soft_spherical_voronoi(directions, sites, colors, temperatures)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSphericalVoronoiRepresentation:
    """Tests for the learnable SV representation module."""
    
    def test_initialization(self):
        """Module should initialize correctly."""
        sv = SphericalVoronoiRepresentation(num_sites=100, num_channels=3)
        assert sv.sites.shape == (100, 3)
        assert sv.colors.shape == (100, 3)
    
    def test_forward_shape(self):
        """Forward pass should return correct shape."""
        sv = SphericalVoronoiRepresentation(num_sites=50)
        directions = F.normalize(torch.randn(32, 64, 3), dim=-1)
        output = sv(directions)
        assert output.shape == (32, 64, 3)
    
    def test_render_equirectangular_shape(self):
        """Equirectangular render should have correct shape."""
        sv = SphericalVoronoiRepresentation(num_sites=50)
        envmap = sv.render_equirectangular(height=128, width=256)
        assert envmap.shape == (128, 256, 3)
    
    def test_sites_remain_on_sphere(self):
        """Normalized sites should always be on unit sphere."""
        sv = SphericalVoronoiRepresentation(num_sites=50)
        # Simulate some gradient updates
        sv.sites.data += torch.randn_like(sv.sites) * 0.5
        
        normalized = sv.get_normalized_sites()
        norms = torch.norm(normalized, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_temperature_always_positive(self):
        """Temperature values should always be positive."""
        sv = SphericalVoronoiRepresentation(num_sites=50, log_temperature=True)
        temps = sv.temperature_values
        assert (temps > 0).all()
        
        # Even with negative log-temperatures
        sv.log_temperatures.data = torch.full_like(sv.log_temperatures, -10.)
        temps = sv.temperature_values
        assert (temps > 0).all()
    
    def test_trainable_parameters(self):
        """All parameters should be trainable."""
        sv = SphericalVoronoiRepresentation(num_sites=50)
        
        # Check that we have the expected parameters
        param_names = [name for name, _ in sv.named_parameters()]
        assert 'sites' in param_names
        assert 'colors' in param_names
        assert 'log_temperatures' in param_names


class TestBatchedSV:
    """Tests for batched SV computation."""
    
    def test_batched_shape(self):
        """Batched output should have correct shape."""
        B, N, K, C = 4, 100, 50, 3
        directions = F.normalize(torch.randn(B, N, 3), dim=-1)
        sites = F.normalize(torch.randn(B, K, 3), dim=-1)
        colors = torch.randn(B, K, C)
        temperatures = torch.ones(B, K) * 10.
        
        output = soft_spherical_voronoi_batched(
            directions, sites, colors, temperatures
        )
        assert output.shape == (B, N, C)
    
    def test_batched_matches_unbatched(self):
        """Batched version should give same result as looped unbatched."""
        B, N, K = 3, 50, 20
        directions = F.normalize(torch.randn(B, N, 3), dim=-1)
        sites = F.normalize(torch.randn(B, K, 3), dim=-1)
        colors = torch.randn(B, K, 3)
        temperatures = torch.ones(B, K) * 10.
        
        batched_result = soft_spherical_voronoi_batched(
            directions, sites, colors, temperatures
        )
        
        # Compare with loop
        for b in range(B):
            single_result = soft_spherical_voronoi(
                directions[b], sites[b], colors[b], temperatures[b]
            )
            assert torch.allclose(batched_result[b], single_result, atol=1e-5)


class TestFitting:
    """Tests for environment map fitting."""
    
    def test_fit_reduces_loss(self):
        """Fitting should reduce loss over iterations or start with very low loss."""
        # Create a simple target (single color)
        target = torch.ones(32, 64, 3) * 0.5
        
        sv, metrics = fit_sv_to_envmap(
            target,
            num_sites=20,
            num_iterations=50,
            verbose=False,
        )
        
        # Either loss should decrease OR initial loss is already very low (good init)
        initial_loss = metrics['losses'][0]
        final_loss = metrics['losses'][-1]
        assert final_loss < initial_loss or initial_loss < 1e-6
    
    def test_fit_returns_valid_psnr(self):
        """PSNR should be a valid finite number."""
        target = torch.rand(32, 64, 3)
        
        sv, metrics = fit_sv_to_envmap(
            target,
            num_sites=30,
            num_iterations=100,
            verbose=False,
        )
        
        assert not math.isnan(metrics['final_psnr'])
        assert not math.isinf(metrics['final_psnr'])
        assert metrics['final_psnr'] > 0  # Should be positive for reasonable fit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
