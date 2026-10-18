"""Terrain feature tests: slope, noise-aware roughness, significance-gated steps."""

from __future__ import annotations

import math

import numpy as np
import pytest
from map_helpers import cell_xy, emap_from

from terranav.perception.features import terrain_features, vertical_span
from terranav.perception.traversability import R_STEP, TraversabilityConfig, traversability

X, Y = cell_xy()
INTERIOR = (slice(3, -3), slice(3, -3))


def test_plane_15deg():
    # Plane rising at 15 deg along the direction 30 deg from +x. Step pairs are at most 0.2 m
    # apart, so jumps are at most 0.2 * tan(15 deg) = 0.0536 m.
    t = math.tan(math.radians(15.0))
    z = t * (X * math.cos(math.radians(30.0)) + Y * math.sin(math.radians(30.0)))
    f = terrain_features(emap_from(z))
    np.testing.assert_allclose(f.slope_rad[INTERIOR], math.radians(15.0), atol=1e-6)
    assert np.nanmax(f.roughness_m[INTERIOR]) < 1e-6
    assert np.nanmax(f.step_m[INTERIOR]) <= 0.2 * t + 1e-9


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
    # z = 0 for x < 6 m (cols <= 59), 0.2 m for cols >= 60. Pairs up to 2 cells apart straddle
    # the edge from cols 58..61 (58-60, 59-60, 59-61), and a cell's step comes only from pairs
    # it is an endpoint of: cols 58..61 get 0.2 m, every other column exactly 0.
    z = np.where(X >= 6.0, 0.2, 0.0)
    f = terrain_features(emap_from(z))
    rows = slice(3, -3)
    for step in (f.step_m, f.step_raw_m):
        np.testing.assert_allclose(step[rows, 58:62], 0.2, atol=1e-12)
        assert np.all(step[rows, 3:58] == 0.0) and np.all(step[rows, 62:-3] == 0.0)


def test_box_edge_inside_a_cell_is_seen_two_cells_apart():
    # A 0.24 m box edge at x = 6.05 m, the middle of column 60: that cell's mean mixes ground
    # and top, 0.12 m. Adjacent jumps are 0.12 m (< step_crit 0.2), but the pair (59, 61), two
    # cells apart, sees the whole 0.24 m: cols 59 and 61 get step 0.24 and are step-lethal.
    z = np.select([X < 6.0, X < 6.1], [0.0, 0.12], 0.24)
    emap = emap_from(z)
    f = terrain_features(emap)
    rows = slice(3, -3)
    np.testing.assert_allclose(f.step_m[rows, 59], 0.24, atol=1e-12)
    np.testing.assert_allclose(f.step_m[rows, 61], 0.24, atol=1e-12)
    np.testing.assert_allclose(f.step_m[rows, 60], 0.12, atol=1e-12)  # its pairs: +-0.12 only
    t = traversability(emap, f, TraversabilityConfig())
    assert np.all(t.core_reason[rows, 59] & R_STEP) and np.all(t.core_reason[rows, 61] & R_STEP)


def test_30deg_plane_has_no_lethal_step():
    # Steepest pair jump: 0.2 m * tan(30 deg) = 0.115 m < step_crit 0.2 (the slope rule, not the
    # step rule, governs slopes).
    z = math.tan(math.radians(30.0)) * (X * math.cos(0.4) + Y * math.sin(0.4))
    emap = emap_from(z)
    f = terrain_features(emap)
    assert np.nanmax(f.step_m) <= 0.2 * math.tan(math.radians(30.0)) + 1e-9
    t = traversability(emap, f, TraversabilityConfig())
    assert not (t.core_reason & R_STEP).any()


def test_noisy_far_flat_ground_has_no_lethal_step():
    # Far range: each cell mean has sigma 0.05 m. Bounded noise |z| <= 0.08 m gives raw jumps up
    # to 0.16 m, but the gate 3 * sqrt(2) * 0.05 = 0.212 m rejects every one: no step-lethal cell.
    rng = np.random.default_rng(3)
    emap = emap_from(rng.uniform(-0.08, 0.08, X.shape), z_mean_var=0.05**2)
    f = terrain_features(emap)
    assert np.all(f.step_m[INTERIOR] == 0.0) and np.nanmax(f.step_raw_m) > 0.1
    t = traversability(emap, f, TraversabilityConfig())
    assert not (t.core_reason & R_STEP).any()


def test_step_significance_gate():
    # Bounded noise |z| <= 0.02 -> any jump <= 0.04 m, but the gate is
    # 3 * sqrt(0.02^2 + 0.02^2) = 0.085 m: no jump is significant, step = 0 everywhere.
    rng = np.random.default_rng(2)
    z = rng.uniform(-0.02, 0.02, X.shape)
    f = terrain_features(emap_from(z, z_mean_var=0.02**2))
    assert np.all(f.step_m[INTERIOR] == 0.0)
    assert np.nanmax(f.step_raw_m[INTERIOR]) > 0.0  # the raw jumps are there, just not significant


def test_unknown_and_unsupported_cells_are_nan():
    known = X < 6.0
    f = terrain_features(emap_from(0.1 * X, known=known))
    for arr in (f.slope_rad, f.roughness_m, f.step_m, f.step_raw_m):
        assert np.all(np.isnan(arr[~known]))  # unknown cells
        assert np.all(np.isfinite(arr[3:-3, 3:59]))
    isolated = np.zeros(X.shape, dtype=bool)
    isolated[60, 60:63] = True  # 3 known cells: support < 8
    f = terrain_features(emap_from(np.zeros(X.shape), known=isolated))
    assert np.all(np.isnan(f.slope_rad))


def test_sub_critical_edge_is_a_step_not_roughness():
    # 0.15 m step at x = 6 m (cols 59 | 60): pairs up to 2 cells apart give step 0.15 m on
    # cols 58..61 (>= ROUGH_EDGE_STEP_M = 0.10, < step_crit 0.20). Plane fits within 3 cells of
    # those endpoints (cols 55..64) straddle the edge; their roughness is zeroed. Nothing is
    # lethal: step 0.15 < 0.20, roughness 0 instead of the edge's residual.
    z = np.where(X >= 6.0, 0.15, 0.0)
    emap = emap_from(z)
    f = terrain_features(emap)
    rows = slice(3, -3)
    np.testing.assert_allclose(f.step_m[rows, 58:62], 0.15, atol=1e-12)
    assert np.all(f.roughness_m[rows, 55:65] == 0.0)
    t = traversability(emap, f, TraversabilityConfig())
    assert not t.core_reason[rows, 3:-3].any()


def test_roughness_away_from_edges_unchanged():
    # iid noise (sigma 0.02) on x < 4 m, and a 0.15 m edge at x = 8 m. Cells farther than
    # 0.3 m (fit radius) + 0.2 m (pair reach) from the edge keep exactly the roughness they
    # have without the edge.
    rng = np.random.default_rng(4)
    noise = np.where(X < 4.0, rng.normal(0.0, 0.02, X.shape), 0.0)
    with_edge = terrain_features(emap_from(noise + np.where(X >= 8.0, 0.15, 0.0)))
    without = terrain_features(emap_from(noise))
    far = (slice(3, -3), slice(3, 70))  # x < 7.0 m: >= 0.5 m from the edge
    # Equal up to float rounding: plane fits centre heights on the map's weighted mean, which
    # the edge shifts (differences ~1e-17).
    np.testing.assert_allclose(with_edge.roughness_m[far], without.roughness_m[far], atol=1e-12)
    assert np.median(with_edge.roughness_m[3:-3, 3:37]) > 0.015  # texture still measured


# --- vertical-surface (in-cell span) step -------------------------------------------------


def ground_with_cells(extra: dict, sigma: float, rows=slice(40, 80), cols=slice(40, 80)):
    """One ground point (z = 0) at the centre of every cell in a 40 x 40 block, plus `extra`
    points per cell {(row, col): z-values}; all points with the given sigma."""
    from map_helpers import MAP

    from terranav.perception.elevation_map import fuse_points

    rr, cc = np.mgrid[rows, cols]
    pts = [np.column_stack([(cc.ravel() + 0.5) * 0.1, (rr.ravel() + 0.5) * 0.1, np.zeros(rr.size)])]
    for (r, c), zs in extra.items():
        zs = np.asarray(zs, dtype=float)
        pts.append(
            np.column_stack(
                [np.full(zs.size, (c + 0.5) * 0.1), np.full(zs.size, (r + 0.5) * 0.1), zs]
            )
        )
    p = np.vstack(pts)
    return fuse_points(p, np.full(len(p), sigma), MAP)


@pytest.mark.parametrize(("height", "lethal"), [(0.24, True), (0.10, False)])
def test_face_inside_a_cell(height, lethal):
    # Near range (sigma 0.007 m): a vertical face inside cell (60, 60) gives 12 extra points from
    # 0 to `height`. Span = height >> 2 * 0.007 * sqrt(2 ln 13) = 0.032 m: significant, so the
    # cell's step is the full face height, lethal at 0.24 m, not at 0.10 m.
    emap = ground_with_cells({(60, 60): np.linspace(0.0, height, 12)}, sigma=0.007)
    f = terrain_features(emap)
    assert f.step_span_m[60, 60] == pytest.approx(height, abs=1e-12)
    assert f.step_m[60, 60] == pytest.approx(height, abs=1e-12)
    t = traversability(emap, f, TraversabilityConfig())
    assert bool(t.core_reason[60, 60] & R_STEP) is lethal


def test_far_range_noise_rarely_reads_as_a_face():
    # Far range: 10,000 flat cells (100 x 100), 3 points each, sigma 0.04 m (seeded). The span
    # threshold is 2 * 0.04 * sqrt(2 ln 3) = 0.119 m. Pure noise must almost never produce a
    # LETHAL step (>= 0.20 m = 5 sigma for a 3-point range): asserted < 1%. The rate at which
    # the significance test fires at all (span > 0.119 m, a sub-critical step) is reported by
    # the assertion message for reference.
    from map_helpers import MAP

    from terranav.perception.elevation_map import fuse_points

    rng = np.random.default_rng(5)
    rr, cc = np.mgrid[10:110, 10:110]
    x = np.repeat((cc.ravel() + 0.5) * 0.1, 3)
    y = np.repeat((rr.ravel() + 0.5) * 0.1, 3)
    z = rng.normal(0.0, 0.04, x.size)
    emap = fuse_points(np.column_stack([x, y, z]), np.full(x.size, 0.04), MAP)
    span = vertical_span(emap)[10:110, 10:110]
    fired = float(np.mean(span > 0))
    lethal = float(np.mean(span >= TraversabilityConfig().step_crit_m))
    assert lethal < 0.01, f"lethal {lethal:.4f}, significant {fired:.4f}"
    assert fired < 0.15, f"significant {fired:.4f}"


def test_filled_and_ground_truth_cells_have_no_span_step():
    # Single-point cells (count 1, as in ground truth) never use the span rule.
    emap = emap_from(np.zeros(X.shape))
    assert np.all(vertical_span(emap) == 0.0)
