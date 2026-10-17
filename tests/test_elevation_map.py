"""2.5D elevation map fusion tests. Expected values are derived by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terranav.frames import T_world_cam
from terranav.perception.elevation_map import (
    OBSERVED,
    UNKNOWN,
    ElevationMap,
    fuse,
    fuse_points,
)
from terranav.sim.sensor import capture
from terranav.types import GridSpec, SensorConfig, World

SMALL = GridSpec(resolution=1.0, size_x=4.0, size_y=3.0, origin=(0.0, 0.0))  # (ny, nx) = (3, 4)
MAP = GridSpec(resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
TERRAIN = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))


def pts(*xyz) -> np.ndarray:
    return np.array(xyz, dtype=np.float64).reshape(-1, 3)


def test_per_cell_statistics():
    # Cell (row 0, col 0): z = 1, 2, 4 with equal sigma -> count 3, min 1, max 4,
    # mean 7/3, population variance ((1-7/3)^2 + (2-7/3)^2 + (4-7/3)^2) / 3 = 14/9.
    # Cell (row 1, col 2): z = 0 (sigma 0.1, w = 100) and z = 1 (sigma 0.2, w = 25):
    # inverse-variance mean = 25 / 125 = 0.2; noise_var = (0.01 + 0.04) / 2 = 0.025;
    # population variance of {0, 1} = 0.25.
    p = pts([0.2, 0.3, 1.0], [0.7, 0.9, 2.0], [0.5, 0.5, 4.0], [2.5, 1.5, 0.0], [2.2, 1.1, 1.0])
    sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.2])
    m = fuse_points(p, sigma, SMALL)
    assert isinstance(m, ElevationMap)
    assert m.count.dtype == np.int32 and m.count.shape == (3, 4)
    assert m.count[0, 0] == 3 and m.count[1, 2] == 2 and m.count.sum() == 5
    assert (m.z_min[0, 0], m.z_max[0, 0]) == (1.0, 4.0)
    assert m.z_mean[0, 0] == pytest.approx(7 / 3, abs=1e-12)
    assert m.z_var[0, 0] == pytest.approx(14 / 9, abs=1e-12)
    assert m.noise_var[0, 0] == pytest.approx(0.01, abs=1e-15)
    assert m.z_mean[1, 2] == pytest.approx(0.2, abs=1e-12)
    assert m.noise_var[1, 2] == pytest.approx(0.025, abs=1e-15)
    # Variance of the weighted mean = 1 / sum(1 / sigma^2): 1 / (3 * 100) and 1 / (100 + 25).
    assert m.z_mean_var[0, 0] == pytest.approx(1 / 300, abs=1e-15)
    assert m.z_mean_var[1, 2] == pytest.approx(1 / 125, abs=1e-15)
    assert m.z_var[1, 2] == pytest.approx(0.25, abs=1e-12)
    assert m.source[0, 0] == OBSERVED and m.source[1, 2] == OBSERVED


def test_single_point_cell_has_zero_variance():
    m = fuse_points(pts([3.5, 2.5, 0.7]), np.array([0.05]), SMALL)
    assert m.count[2, 3] == 1
    assert m.z_mean[2, 3] == m.z_min[2, 3] == m.z_max[2, 3] == 0.7
    assert m.z_var[2, 3] == 0.0


def test_points_outside_half_open_map_dropped():
    # The map covers x in [0, 4), y in [0, 3): x = 4.0 and y = 3.0 lie outside; x = 3.999 is in
    # the last column.
    p = pts(
        [-0.1, 1.0, 0.0], [4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [1.0, -1e-9, 0.0], [3.999, 2.0, 0.5]
    )
    m = fuse_points(p, np.full(5, 0.1), SMALL)
    assert m.count.sum() == 1
    assert m.count[2, 3] == 1 and m.z_mean[2, 3] == 0.5


def test_point_on_cell_boundary_goes_to_higher_index_cell():
    # floor binning: x = 1.0 is the lower edge of column 1, not the upper edge of column 0.
    m = fuse_points(pts([1.0, 2.0, 0.3]), np.array([0.1]), SMALL)
    assert m.count[2, 1] == 1 and m.count[2, 0] == 0


def test_unknown_cells():
    m = fuse_points(pts([0.5, 0.5, 1.0]), np.array([0.1]), SMALL)
    unknown = m.count == 0
    assert unknown.sum() == 11
    for arr in (m.z_min, m.z_max, m.z_mean, m.z_mean_var, m.z_var, m.noise_var):
        assert np.all(np.isnan(arr[unknown]))
    assert np.all(m.source[unknown] == UNKNOWN)


def test_order_independence():
    rng = np.random.default_rng(0)
    p = np.column_stack(
        [rng.uniform(0, 12, 20_000), rng.uniform(0, 12, 20_000), rng.normal(0, 1, 20_000)]
    )
    sigma = rng.uniform(0.001, 0.1, 20_000)
    a = fuse_points(p, sigma, MAP)
    perm = rng.permutation(len(p))
    b = fuse_points(p[perm], sigma[perm], MAP)
    np.testing.assert_array_equal(a.count, b.count)
    np.testing.assert_array_equal(a.z_min, b.z_min)
    np.testing.assert_array_equal(a.z_max, b.z_max)
    np.testing.assert_allclose(a.z_mean, b.z_mean, atol=1e-12, equal_nan=True)


def test_arrays_read_only():
    m = fuse_points(pts([0.5, 0.5, 1.0]), np.array([0.1]), SMALL)
    for arr in (m.count, m.z_min, m.z_max, m.z_mean, m.z_var, m.noise_var, m.source):
        with pytest.raises(ValueError):
            arr.flat[0] = 0


def test_fuse_flat_world_end_to_end():
    # Noise and dropout off: every observed cell's mean height is the flat ground, 0. Points
    # come from float32 depth (relative error ~6e-8), so 1e-6 m is a loose bound.
    world = World(np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (), "flat", 0)
    sensor = SensorConfig(width=64, height=48, noise_s0_m=0.0, noise_s1_per_m=0.0, dropout_prob=0.0)
    frames = []
    for i, (x, y, yaw_deg) in enumerate([(2.0, 6.0, 0.0), (10.0, 6.0, 180.0)]):
        T = T_world_cam(np.array([x, y, 0.5]), math.radians(yaw_deg), sensor.pitch_down_rad)
        frames.append(capture(world, T, sensor, np.random.default_rng((0, i)))[0])
    m = fuse(frames, sensor, MAP)
    observed = m.count > 0
    assert observed.sum() > 100
    np.testing.assert_allclose(m.z_mean[observed], 0.0, atol=1e-6)
    assert np.all(m.source[observed] == OBSERVED)
