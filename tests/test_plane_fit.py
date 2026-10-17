"""Windowed weighted plane fit tests (hand-derived)."""

from __future__ import annotations

import numpy as np
from map_helpers import cell_xy, emap_from

from terranav.perception.plane_fit import MIN_SUPPORT, plane_fit

X, Y = cell_xy()
# Disk of radius 3 cells (0.3 m): rows dy = 0: 7 cells, |dy| = 1, 2: 5 each, |dy| = 3: 1 each.
FULL_SUPPORT = 7 + 2 * 5 + 2 * 5 + 2 * 1  # 29


def test_exact_tilted_rotated_plane():
    z = 0.3 * X - 0.2 * Y + 0.1
    fit = plane_fit(emap_from(z))
    interior = (slice(3, -3), slice(3, -3))
    np.testing.assert_allclose(fit.a, 0.3, atol=1e-9)  # exact everywhere, edges included
    np.testing.assert_allclose(fit.b, -0.2, atol=1e-9)
    np.testing.assert_allclose(fit.c, z, atol=1e-9)
    assert np.all(fit.support[interior] == FULL_SUPPORT)
    assert np.nanmax(fit.residual_m) < 1e-6


def test_partial_support_near_unknown_region():
    # Known only for x < 6 m (columns 0..59). Column 59 (last known) keeps dx <= 0:
    # 4 + 2*3 + 2*3 + 2*1 = 18 cells. Column 60 (unknown) sees dx <= -1: 3 + 2*2 + 2*2 = 11.
    # Column 62 sees only dx = -3 at dy = 0: 1 cell -> below MIN_SUPPORT -> NaN.
    z = 0.25 * X + 0.05 * Y
    fit = plane_fit(emap_from(z, known=X < 6.0))
    row = 60
    assert fit.support[row, 59] == 18 and fit.support[row, 60] == 11 and fit.support[row, 62] == 1
    for col in (59, 60):
        assert abs(fit.a[row, col] - 0.25) < 1e-9 and abs(fit.b[row, col] - 0.05) < 1e-9
    assert np.isnan(fit.a[row, 62]) and np.isnan(fit.residual_m[row, 62])


def test_support_below_minimum_is_nan():
    known = np.zeros(X.shape, dtype=bool)
    known[60, 58:63] = True  # 5 cells in a line
    known[58:63, 60] = True  # plus a cross: 9 cells in total, but at most 9 in any disk
    fit = plane_fit(emap_from(0.1 * X, known=known), min_support=10)
    assert np.all(np.isnan(fit.a)) and np.all(np.isnan(fit.residual_m))
    assert MIN_SUPPORT == 8
    fit8 = plane_fit(emap_from(0.1 * X, known=known))
    assert abs(fit8.a[60, 60] - 0.1) < 1e-9  # 9 >= 8 at the cross centre


def test_residual_of_iid_noise_estimates_sigma():
    # Unbiased (dof-corrected) residual RMS of pure noise is ~sigma. With 29 cells per fit
    # and 26 dof, each cell's estimate has ~14% relative spread; the median over ~13k
    # interior cells is within a few percent of sigma.
    rng = np.random.default_rng(0)
    fit = plane_fit(emap_from(rng.normal(0.0, 0.02, X.shape)))
    assert abs(np.median(fit.residual_m[3:-3, 3:-3]) - 0.02) < 0.001
