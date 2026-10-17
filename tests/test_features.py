"""Terrain feature tests: slope, noise-aware roughness, significance-gated steps."""

from __future__ import annotations

import math

import numpy as np
from map_helpers import cell_xy, emap_from

from terranav.perception.features import terrain_features

X, Y = cell_xy()
INTERIOR = (slice(3, -3), slice(3, -3))


def test_plane_15deg():
    # Plane rising at 15 deg along the direction 30 deg from +x. Adjacent-cell jumps are at
    # most tan(15 deg) * 0.1 * sqrt(2) = 0.0379 m (diagonal), so step < 0.04.
    t = math.tan(math.radians(15.0))
    z = t * (X * math.cos(math.radians(30.0)) + Y * math.sin(math.radians(30.0)))
    f = terrain_features(emap_from(z))
    np.testing.assert_allclose(f.slope_rad[INTERIOR], math.radians(15.0), atol=1e-6)
    assert np.nanmax(f.roughness_m[INTERIOR]) < 1e-6
    assert np.nanmax(f.step_m[INTERIOR]) < 0.04


def test_roughness_without_and_with_noise_subtraction():
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 0.02, X.shape)
    # Fitted values carry (almost) no modelled noise -> roughness ~ the noise sigma.
    # Median over ~13k cells: within 5% (each cell's estimate has ~14% spread, 26 dof).
    rough = terrain_features(emap_from(z)).roughness_m[INTERIOR]
    assert abs(np.median(rough) - 0.02) < 0.001
    # Modelled noise variance 0.02^2 on every value -> residual^2 - 0.02^2 is ~0 on average:
    # about half the cells clip to 0 (median 0), the rest are small (chi^2_26 spread).
    rough = terrain_features(emap_from(z, z_mean_var=0.02**2, noise_var=0.02**2)).roughness_m
    assert np.median(rough[INTERIOR]) < 0.001
    assert np.mean(rough[INTERIOR]) < 0.006


def test_step_edge():
    # z = 0 for x < 6 m (cols <= 59), 0.2 m for cols >= 60. Only pairs straddling the edge
    # (endpoints in cols 59 and 60) jump; a cell gets the step if a jumping pair has an
    # endpoint within r_step = 0.2 m (2 cells): cols 57..62. Elsewhere step = 0 exactly.
    z = np.where(X >= 6.0, 0.2, 0.0)
    step = terrain_features(emap_from(z)).step_m
    rows = slice(3, -3)
    np.testing.assert_allclose(step[rows, 57:63], 0.2, atol=1e-12)
    assert np.all(step[rows, 3:57] == 0.0) and np.all(step[rows, 63:-3] == 0.0)


def test_step_significance_gate():
    # Bounded noise |z| <= 0.02 -> any jump <= 0.04 m, but the gate is
    # 3 * sqrt(0.02^2 + 0.02^2) = 0.085 m: no jump is significant, step = 0 everywhere.
    rng = np.random.default_rng(2)
    z = rng.uniform(-0.02, 0.02, X.shape)
    step = terrain_features(emap_from(z, z_mean_var=0.02**2)).step_m
    assert np.all(step[INTERIOR] == 0.0)


def test_unknown_and_unsupported_cells_are_nan():
    known = X < 6.0
    f = terrain_features(emap_from(0.1 * X, known=known))
    for arr in (f.slope_rad, f.roughness_m, f.step_m):
        assert np.all(np.isnan(arr[~known]))  # unknown cells
        assert np.all(np.isfinite(arr[3:-3, 3:59]))
    isolated = np.zeros(X.shape, dtype=bool)
    isolated[60, 60:63] = True  # 3 known cells: support < 8
    f = terrain_features(emap_from(np.zeros(X.shape), known=isolated))
    assert np.all(np.isnan(f.slope_rad))
