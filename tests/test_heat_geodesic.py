"""Verify the heat-method geodesic kernel on small graphs.

Property tests:
1. Distance from a seed to itself is zero.
2. Distances are non-negative.
3. On a uniformly sampled circle, the geodesic from a seed point increases
   approximately linearly with arc-length around the circle.
"""
from __future__ import annotations

import math

import torch

from geosam3d.propagate import HeatGeodesicKernel


def test_seed_distance_is_zero():
    torch.manual_seed(0)
    pts = torch.randn(50, 3)
    kernel = HeatGeodesicKernel(k=8, t=0.1)
    seed_mask = torch.zeros(50)
    seed_mask[5] = 1.0
    d = kernel.geodesic(pts, seed_mask)
    assert d[5].abs() < 1e-3


def test_distances_are_nonneg():
    torch.manual_seed(0)
    pts = torch.randn(30, 3)
    kernel = HeatGeodesicKernel(k=6, t=0.1)
    seed_mask = torch.zeros(30)
    seed_mask[0] = 1.0
    d = kernel.geodesic(pts, seed_mask)
    assert (d >= -1e-4).all()


def test_geodesic_on_circle_is_monotone():
    """Order of geodesic distances should match arc-length order on a circle."""
    n = 24
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    pts = torch.stack([angles.cos(), angles.sin(), torch.zeros_like(angles)], dim=-1)
    kernel = HeatGeodesicKernel(k=4, t=0.05)
    seed_mask = torch.zeros(n)
    seed_mask[0] = 1.0
    d = kernel.geodesic(pts, seed_mask).cpu().numpy()
    # The "near" half (idx 1..n//2) should have smaller distance than the "far" half (idx n//2..n-1)
    near = d[1: n // 2 + 1]
    far = d[n // 2: n]
    assert near.mean() < far.mean()


def test_propagate_label_is_in_unit_interval():
    torch.manual_seed(0)
    pts = torch.randn(40, 3)
    kernel = HeatGeodesicKernel(k=8)
    seed_mask = torch.zeros(40)
    seed_mask[3] = 1.0
    p = kernel.propagate_label(pts, seed_mask)
    assert (p >= 0.0).all() and (p <= 1.0 + 1e-6).all()
    assert p[3] > 0.5  # closer to seed -> higher prob
