"""Backprojection tests. Expected values are derived by hand or from the ray-caster's
independent ground truth (RaycastResult.points_world), never from backproject itself."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from terranav.frames import T_world_cam
from terranav.perception.backproject import backproject
from terranav.sim.raycast import cast_frame
from terranav.sim.scenario import generate_scenario
from terranav.sim.sensor import capture
from terranav.types import CameraIntrinsics, DepthFrame, GridSpec, SensorConfig, World

GRID = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
# Noise off, dropout off, grazing cutoff 90 deg: valid depth_z == ideal depth_z (float32).
EXACT = SensorConfig(
    noise_s0_m=0.0, noise_s1_per_m=0.0, dropout_prob=0.0, grazing_cutoff_rad=math.pi / 2
)


def test_round_trip_reproduces_raycast_points():
    # capture() stores depth_z as float32: relative rounding error <= 2^-24 ~ 6e-8, i.e.
    # <= 5e-7 m at 8 m, times ||d_cam|| <= ~1.5 for the corner rays -> < 1e-6 m. A 1e-5 m
    # tolerance therefore leaves a 10x margin over float32 precision.
    sc = generate_scenario("mixed", 0)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    T = T_world_cam(np.array([sx, sy, 0.5]), math.atan2(gy - sy, gx - sx), math.radians(25.0))
    frame, _ = capture(sc.world, T, EXACT, np.random.default_rng((0, 0)))
    ray = cast_frame(sc.world, T, EXACT.intrinsics(), EXACT.max_depth_z_m)

    points, vu = backproject(frame)
    assert len(points) > 10_000
    expected = ray.points_world[vu[:, 0], vu[:, 1]]
    np.testing.assert_allclose(points, expected, atol=1e-5)


def test_hand_derived_points_and_nan_exclusion():
    # Camera at (1, 2, 3), yaw 0, pitch 0: optical z -> world +x, x (right) -> -y,
    # y (down) -> -z. Pixel (u, v) at depth d lands at
    #   (1 + d, 2 - d (u - cx)/fx, 3 - d (v - cy)/fy).
    intr = CameraIntrinsics.from_hfov(5, 3, math.radians(90.0))  # fx = 2.5, cx = 2, cy = 1
    depth = np.full((3, 5), np.nan, dtype=np.float32)
    depth[1, 2] = 2.0  # centre pixel
    depth[0, 4] = 4.0  # top-right corner
    depth[2, 0] = 1.0  # bottom-left corner
    T = T_world_cam(np.array([1.0, 2.0, 3.0]), 0.0, 0.0)
    points, vu = backproject(DepthFrame(depth, intr, T))

    assert points.shape == (3, 3) and points.dtype == np.float64
    assert vu.shape == (3, 2) and np.issubdtype(vu.dtype, np.integer)
    by_pixel = {tuple(p): q for p, q in zip(vu.tolist(), points, strict=True)}
    assert set(by_pixel) == {(1, 2), (0, 4), (2, 0)}
    np.testing.assert_allclose(by_pixel[(1, 2)], [3.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(by_pixel[(0, 4)], [5.0, 2.0 - 4.0 * 2 / 2.5, 3.0 + 4.0 / 2.5])
    np.testing.assert_allclose(by_pixel[(2, 0)], [2.0, 2.0 + 2 / 2.5, 3.0 - 1 / 2.5])


def test_count_matches_valid_pixels():
    sc = generate_scenario("rough", 3)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    T = T_world_cam(np.array([sx, sy, 0.5]), math.atan2(gy - sy, gx - sx), math.radians(25.0))
    frame, _ = capture(sc.world, T, SensorConfig(), np.random.default_rng((3, 0)))
    points, vu = backproject(frame)
    valid = ~np.isnan(frame.depth_z)
    assert (~valid).any()
    assert len(points) == len(vu) == int(valid.sum())
    assert np.all(valid[vu[:, 0], vu[:, 1]])
    assert np.all(np.isfinite(points))


def test_all_nan_frame_gives_empty_arrays():
    intr = CameraIntrinsics.from_hfov(4, 3, math.radians(90.0))
    frame = DepthFrame(np.full((3, 4), np.nan, dtype=np.float32), intr, np.eye(4))
    points, vu = backproject(frame)
    assert points.shape == (0, 3) and vu.shape == (0, 2)


SMALL_SENSOR = SensorConfig(
    width=64,
    height=48,
    noise_s0_m=0.0,
    noise_s1_per_m=0.0,
    dropout_prob=0.0,
    grazing_cutoff_rad=math.pi / 2,
)
FLAT = World(np.zeros((GRID.ny, GRID.nx)), GRID, (), "flat", 0)


@settings(deadline=None, max_examples=40)
@given(
    x=st.floats(1.5, 10.5),
    y=st.floats(1.5, 10.5),
    z=st.floats(0.2, 1.0),
    yaw_rad=st.floats(-math.pi, math.pi),
    pitch_down_rad=st.floats(math.radians(10.0), math.radians(80.0)),
)
def test_flat_ground_backprojects_to_z0(x, y, z, yaw_rad, pitch_down_rad):
    T = T_world_cam(np.array([x, y, z]), yaw_rad, pitch_down_rad)
    frame, _ = capture(FLAT, T, SMALL_SENSOR, np.random.default_rng(0))
    points, _ = backproject(frame)
    # The bottom image row looks >= 10 deg + half the vertical FOV (~35 deg) = 45 deg below
    # the horizon, so it meets the ground within z / tan(45 deg) <= 1 m horizontally: still
    # on the map from x, y in [1.5, 10.5], and well within 8 m depth.
    assert len(points) > 0
    np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-4)
