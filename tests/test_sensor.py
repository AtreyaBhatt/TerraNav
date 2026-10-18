"""Depth sensor model tests. Expected values are derived by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terranav.frames import T_world_cam, pixel_rays_cam
from terranav.sim.raycast import HIT_BOX, HIT_NONE, RaycastResult, cast_frame
from terranav.sim.scenario import generate_scenario
from terranav.sim.sensor import (
    BEYOND_MAX,
    DROPOUT,
    GRAZING,
    MISS,
    capture,
    simulate_depth,
)
from terranav.types import Box, CameraIntrinsics, GridSpec, SensorConfig, World

GRID = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
HFOV_RAD = math.radians(87.0)
BIG = CameraIntrinsics.from_hfov(321, 241, HFOV_RAD)  # cx = 160, cy = 120 (integer)
HUGE = CameraIntrinsics.from_hfov(1001, 1001, HFOV_RAD)  # ~1e6 pixels for statistics
QUIET = SensorConfig(
    noise_s0_m=0.0, noise_s1_per_m=0.0, dropout_prob=0.0, grazing_cutoff_rad=math.pi / 2
)


def pose(x, y, z, yaw_deg=0.0, pitch_down_deg=0.0) -> np.ndarray:
    return T_world_cam(np.array([x, y, z]), math.radians(yaw_deg), math.radians(pitch_down_deg))


def flat_world(boxes: tuple[Box, ...] = ()) -> World:
    return World(np.zeros((GRID.ny, GRID.nx)), GRID, boxes, "test", 0)


def wall_ray(intr: CameraIntrinsics, depth_z: float, T: np.ndarray) -> RaycastResult:
    """Synthetic hit on a plane perpendicular to the optical axis at the given depth_z.

    With z = 1 camera rays d, the hit is depth_z * d, so range_t = depth_z * ||d||. The
    plane's normal faces the camera: -(optical axis) in world.
    """
    d = pixel_rays_cam(intr)
    range_t = depth_z * np.linalg.norm(d, axis=-1)
    R, o = T[:3, :3], T[:3, 3]
    points = o + (depth_z * d) @ R.T
    normals = np.broadcast_to(-(R @ [0.0, 0.0, 1.0]), d.shape)
    kind = np.full(range_t.shape, HIT_BOX, dtype=np.int8)
    return RaycastResult(range_t, points, normals, kind)


# --- Prerequisite: camera inside a box ----------------------------------------------


@pytest.mark.parametrize("z", [0.5, 1.0])  # strictly inside, and on the top face
def test_cast_frame_rejects_camera_inside_box(z):
    world = flat_world((Box((5.0, 5.0, 0.0), (7.0, 7.0, 1.0)),))
    with pytest.raises(ValueError, match="box"):
        cast_frame(world, pose(6, 6, z), BIG, 8.0)


# --- Noise off -----------------------------------------------------------------------


def test_noise_off_matches_ideal_and_misses():
    world = flat_world((Box((4.0, 5.0, 0.0), (5.0, 7.0, 1.0)),))
    T = pose(2, 6, 0.5, 0, 25)
    ray = cast_frame(world, T, BIG, QUIET.max_depth_z_m)
    frame, dbg = simulate_depth(ray, BIG, T, QUIET, np.random.default_rng(0))
    miss = ray.hit_kind == HIT_NONE
    assert miss.any() and (~miss).any()
    np.testing.assert_array_equal(np.isnan(frame.depth_z), miss)
    np.testing.assert_array_equal(frame.depth_z[~miss], dbg.depth_z_ideal[~miss].astype(np.float32))
    np.testing.assert_array_equal(dbg.invalid_reason[miss], MISS)
    np.testing.assert_array_equal(dbg.invalid_reason[~miss], 0)


# --- Noise and dropout statistics ----------------------------------------------------


def test_noise_statistics_at_constant_depth():
    # sigma(2 m) = 0.001 + 0.0015 * 2^2 = 0.007 m. N = 1001^2 ~ 1e6 samples.
    # Sample std has relative standard error ~ 1/sqrt(2N) ~ 0.07 %, so a 3 % band is ~40 SE.
    # The sample mean has standard error sigma/sqrt(N); 4 SE fails with p ~ 6e-5.
    T = pose(0, 0, 10.0)
    cfg = SensorConfig(dropout_prob=0.0, grazing_cutoff_rad=math.pi / 2)
    frame, _ = simulate_depth(wall_ray(HUGE, 2.0, T), HUGE, T, cfg, np.random.default_rng(1))
    err = frame.depth_z.astype(np.float64) - 2.0
    assert np.all(np.isfinite(err))
    sigma, n = 0.007, err.size
    assert abs(err.std() - sigma) <= 0.03 * sigma
    assert abs(err.mean()) < 4 * sigma / math.sqrt(n)


def test_dropout_rate():
    # Binomial: std of the dropped fraction = sqrt(p (1 - p) / N) = sqrt(0.0196 / 1e6) ~ 0.00014
    # for N ~ 1e6 (~0.0005 at 321 x 241). A +/-0.003 band is > 20 std.
    T = pose(0, 0, 10.0)
    cfg = SensorConfig(noise_s0_m=0.0, noise_s1_per_m=0.0, grazing_cutoff_rad=math.pi / 2)
    frame, dbg = simulate_depth(wall_ray(HUGE, 2.0, T), HUGE, T, cfg, np.random.default_rng(2))
    dropped = dbg.invalid_reason == DROPOUT
    assert abs(dropped.mean() - 0.02) <= 0.003
    np.testing.assert_array_equal(np.isnan(frame.depth_z), dropped)


# --- Grazing -------------------------------------------------------------------------


def grazing_row(beta_deg: float) -> int:
    """Centre-column row whose elevation below the horizon is closest to beta_deg."""
    return round(BIG.cy + BIG.fy * math.tan(math.radians(beta_deg - 25.0)))


def test_grazing_on_flat_ground():
    # Camera 0.5 m above flat ground, pitched down 25 deg. Centre-column pixel v looks
    # beta(v) = 25 deg + atan((v - cy)/fy) below the horizon; on flat ground (normal +z)
    # the incidence angle is 90 deg - beta. Cutoff set explicitly to 80 deg (independent of
    # the default) -> valid iff beta >= 10 deg.
    T = pose(2, 6, 0.5, 0, 25)
    cfg = SensorConfig(
        noise_s0_m=0.0, noise_s1_per_m=0.0, dropout_prob=0.0, grazing_cutoff_rad=math.radians(80.0)
    )
    ray = cast_frame(flat_world(), T, BIG, cfg.max_depth_z_m)
    frame, dbg = simulate_depth(ray, BIG, T, cfg, np.random.default_rng(0))
    u0 = int(BIG.cx)
    for target_beta, valid in ((12.0, True), (8.0, False)):
        v = grazing_row(target_beta)
        off_axis = math.atan((v - BIG.cy) / BIG.fy)
        beta = math.radians(25.0) + off_axis
        incidence_deg = 90.0 - math.degrees(beta)
        # Both rows hit within 8 m depth: depth_z = (0.5 / sin beta) * cos(off_axis).
        depth = 0.5 / math.sin(beta) * math.cos(off_axis)
        assert depth < 8.0
        assert dbg.depth_z_ideal[v, u0] == pytest.approx(depth, abs=1e-9)
        if valid:
            assert incidence_deg < 80.0
            assert frame.depth_z[v, u0] == pytest.approx(depth, rel=1e-6)
            assert dbg.invalid_reason[v, u0] == 0
        else:
            assert incidence_deg > 80.0
            assert np.isnan(frame.depth_z[v, u0])
            assert dbg.invalid_reason[v, u0] & GRAZING


# --- BEYOND_MAX uses depth_z ---------------------------------------------------------


def test_beyond_max_uses_depth_not_range():
    # Camera at (2, 6, 2.5) looking along +x at a wall whose face is at x = 9.5: depth_z =
    # 7.5 m for every pixel that hits it. Centre-row pixel u = 280 has (u - cx)/fx = 120/fx
    # = 0.7095, so range_t = 7.5 * sqrt(1 + 0.7095^2) = 9.19 m > 8 m, yet depth_z < 8 m.
    # Its lateral offset 7.5 * 0.7095 = 5.32 m keeps the hit at y = 0.68, on the wall.
    wall = Box((9.5, 0.0, 0.0), (10.0, 12.0, 5.0))
    T = pose(2, 6, 2.5)
    v, u = int(BIG.cy), 280
    ratio = (u - BIG.cx) / BIG.fx
    range_t = 7.5 * math.sqrt(1 + ratio**2)
    assert range_t > 8.0
    frame, dbg = capture(flat_world((wall,)), T, QUIET, np.random.default_rng(0))
    assert dbg.invalid_reason[v, u] == 0
    assert frame.depth_z[v, u] == pytest.approx(7.5, rel=1e-6)


def test_beyond_max_flags_noisy_depth_over_limit():
    # Constant ideal depth 7.99 m just under the limit; sigma(7.99) ~ 0.097 m, so a large
    # share of pixels is pushed past 8 m by noise and must be flagged BEYOND_MAX.
    T = pose(0, 0, 10.0)
    cfg = SensorConfig(dropout_prob=0.0, grazing_cutoff_rad=math.pi / 2)
    frame, dbg = simulate_depth(wall_ray(BIG, 7.99, T), BIG, T, cfg, np.random.default_rng(3))
    beyond = (dbg.invalid_reason & BEYOND_MAX) != 0
    assert 0.3 < beyond.mean() < 0.7
    assert np.all(np.isnan(frame.depth_z[beyond]))
    assert np.nanmax(frame.depth_z) <= 8.0


# --- Consistency, determinism, types ---------------------------------------------------


def mixed_capture(seed: int):
    sc = generate_scenario("mixed", 0)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    T = T_world_cam(np.array([sx, sy, 0.5]), math.atan2(gy - sy, gx - sx), math.radians(25.0))
    return capture(sc.world, T, SensorConfig(), np.random.default_rng(seed))


def test_invalid_reason_consistent_with_nan():
    frame, dbg = mixed_capture(0)
    np.testing.assert_array_equal(np.isnan(frame.depth_z), dbg.invalid_reason != 0)
    assert dbg.invalid_reason.dtype == np.uint8
    assert np.all(dbg.invalid_reason < 16)
    assert dbg.depth_z_ideal.dtype == np.float64
    # MISS excludes every other reason.
    miss = (dbg.invalid_reason & MISS) != 0
    np.testing.assert_array_equal(dbg.invalid_reason[miss], MISS)
    # Dropout only hits pixels that were otherwise valid.
    drop = (dbg.invalid_reason & DROPOUT) != 0
    assert drop.any()
    np.testing.assert_array_equal(dbg.invalid_reason[drop], DROPOUT)


@pytest.mark.slow  # three full-resolution captures
def test_determinism_and_global_rng_untouched():
    before = np.random.get_state()
    a, da = mixed_capture(7)
    b, db = mixed_capture(7)
    c, _ = mixed_capture(8)
    after = np.random.get_state()
    assert np.array_equal(a.depth_z, b.depth_z, equal_nan=True)
    assert np.array_equal(da.invalid_reason, db.invalid_reason)
    assert not np.array_equal(a.depth_z, c.depth_z, equal_nan=True)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_output_types_and_read_only():
    frame, dbg = mixed_capture(0)
    assert frame.depth_z.dtype == np.float32
    assert frame.depth_z.shape == (240, 320)
    assert frame.T_world_cam.dtype == np.float64
    for arr in (frame.depth_z, dbg.depth_z_ideal, dbg.invalid_reason):
        with pytest.raises(ValueError):
            arr.flat[0] = 0
