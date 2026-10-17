"""Frame convention tests. Expected values are derived by hand (see docs/frames.md)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from terranav.frames import R_wc, T_world_cam, pixel_rays_cam, transform_points
from terranav.types import CameraIntrinsics, GridSpec, World

OPTICAL_Z = np.array([0.0, 0.0, 1.0])


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def world_dir_of_pixel(
    intr: CameraIntrinsics, yaw_rad: float, pitch_down_rad: float, u: int, v: int
):
    rays = pixel_rays_cam(intr)
    return unit(R_wc(yaw_rad, pitch_down_rad) @ rays[v, u])


# --- Orientation -------------------------------------------------------------


def test_identity_orientation_optical_axis_is_world_x():
    np.testing.assert_allclose(R_wc(0.0, 0.0) @ OPTICAL_Z, [1.0, 0.0, 0.0], atol=1e-12)


def test_identity_orientation_camera_right_is_world_minus_y_and_down_is_minus_z():
    R = R_wc(0.0, 0.0)
    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)


def test_pitch_down_25deg_tilts_axis_toward_minus_z():
    pitch = math.radians(25.0)
    axis = R_wc(0.0, pitch) @ OPTICAL_Z
    assert axis[0] == pytest.approx(math.cos(pitch), abs=1e-12)
    assert axis[1] == pytest.approx(0.0, abs=1e-12)
    assert axis[2] == pytest.approx(-math.sin(pitch), abs=1e-12)
    # Worked example in docs/frames.md.
    np.testing.assert_allclose(axis, [0.906, 0.0, -0.423], atol=5e-4)


def test_yaw_90deg_optical_axis_is_world_y():
    axis = R_wc(math.pi / 2, 0.0) @ OPTICAL_Z
    np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-12)


@given(
    yaw_rad=st.floats(min_value=-math.pi, max_value=math.pi),
    pitch_down_rad=st.floats(min_value=-math.pi / 2, max_value=math.pi / 2),
)
def test_R_wc_is_proper_rotation(yaw_rad, pitch_down_rad):
    R = R_wc(yaw_rad, pitch_down_rad)
    assert R.dtype == np.float64
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


# --- Pixel rays --------------------------------------------------------------


def test_pixel_right_of_center_maps_to_world_minus_y():
    intr = CameraIntrinsics.from_hfov(320, 240, math.radians(87.0))
    for pitch in (0.0, math.radians(25.0)):
        d = world_dir_of_pixel(intr, 0.0, pitch, u=300, v=120)
        assert d[1] < 0.0


def test_pixel_below_center_has_lower_world_z_than_axis():
    intr = CameraIntrinsics.from_hfov(321, 241, math.radians(87.0))  # odd: (160,120) is centre
    for pitch in (0.0, math.radians(25.0)):
        axis = R_wc(0.0, pitch) @ OPTICAL_Z
        d = world_dir_of_pixel(intr, 0.0, pitch, u=160, v=200)
        assert d[2] < axis[2]


def test_center_pixel_ray_equals_optical_axis():
    # Odd image size so that (u, v) = (cx, cy) is an actual pixel centre.
    intr = CameraIntrinsics.from_hfov(321, 241, math.radians(87.0))
    assert (intr.cx, intr.cy) == (160.0, 120.0)
    rays = pixel_rays_cam(intr)
    np.testing.assert_allclose(rays[120, 160], [0.0, 0.0, 1.0], atol=1e-15)

    yaw, pitch = math.radians(30.0), math.radians(25.0)
    expected_axis = [
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ]
    np.testing.assert_allclose(
        world_dir_of_pixel(intr, yaw, pitch, u=160, v=120), expected_axis, atol=1e-12
    )


def test_even_image_center_falls_between_pixels():
    intr = CameraIntrinsics.from_hfov(320, 240, math.radians(87.0))
    rays = pixel_rays_cam(intr)
    assert rays.shape == (240, 320, 3)
    assert rays.dtype == np.float64
    np.testing.assert_array_equal(rays[..., 2], 1.0)
    # Pixels 159 and 160 are symmetric about cx = 159.5: x = -/+ 0.5 / fx.
    assert rays[0, 159, 0] == pytest.approx(-0.5 / intr.fx)
    assert rays[0, 160, 0] == pytest.approx(0.5 / intr.fx)
    assert rays[119, 0, 1] == pytest.approx(-0.5 / intr.fy)
    assert rays[120, 0, 1] == pytest.approx(0.5 / intr.fy)


# --- Poses -------------------------------------------------------------------


def test_T_world_cam_maps_camera_origin_to_position():
    pos = np.array([3.0, 4.0, 0.5])
    T = T_world_cam(pos, math.radians(40.0), math.radians(25.0))
    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    np.testing.assert_allclose(transform_points(T, np.zeros((1, 3))), [pos], atol=1e-12)
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_T_world_cam_point_on_optical_axis():
    # yaw=0, pitch_down=25 deg, 2 m ahead along optical axis.
    pos = np.array([1.0, 2.0, 0.5])
    T = T_world_cam(pos, 0.0, math.radians(25.0))
    c, s = math.cos(math.radians(25.0)), math.sin(math.radians(25.0))
    expected = [1.0 + 2.0 * c, 2.0, 0.5 - 2.0 * s]
    np.testing.assert_allclose(transform_points(T, np.array([[0.0, 0.0, 2.0]])), [expected])


# --- Intrinsics --------------------------------------------------------------


def test_from_hfov_default_camera():
    intr = CameraIntrinsics.from_hfov(320, 240, math.radians(87.0))
    # 160 / tan(43.5 deg) = 160 / 0.94896 = 168.60
    assert intr.fx == pytest.approx(168.6, abs=0.05)
    assert intr.fy == intr.fx
    assert intr.cx == 159.5
    assert intr.cy == 119.5
    np.testing.assert_allclose(
        intr.K, [[intr.fx, 0.0, 159.5], [0.0, intr.fy, 119.5], [0.0, 0.0, 1.0]]
    )


# --- Grid / World ------------------------------------------------------------


def test_gridspec_derived_sizes():
    g = GridSpec(resolution=0.05, size_x=12.0, size_y=6.0, origin=(0.0, 0.0))
    assert (g.nx, g.ny) == (240, 120)


def test_terrain_and_map_grids_differ():
    # Guard against confusing the 0.05 m terrain grid with the 0.1 m map grid.
    terrain = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
    map_grid = GridSpec(resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
    assert (terrain.ny, terrain.nx) == (240, 240)
    assert (map_grid.ny, map_grid.nx) == (120, 120)
    assert terrain.nx != map_grid.nx
    assert terrain.ny != map_grid.ny


def test_gridspec_rc_to_xy_to_rc_round_trip():
    g = GridSpec(resolution=0.05, size_x=12.0, size_y=6.0, origin=(-1.0, 2.0))
    row, col = np.meshgrid(np.arange(g.ny), np.arange(g.nx), indexing="ij")
    x, y = g.rc_to_xy(row, col)
    row2, col2 = g.xy_to_rc(x, y)
    np.testing.assert_array_equal(row2, row)
    np.testing.assert_array_equal(col2, col)


def test_gridspec_xy_to_rc_known_values():
    g = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
    # x=0.12 -> col=2 (0.10..0.15); y=0.07 -> row=1 (0.05..0.10).
    row, col = g.xy_to_rc(0.12, 0.07)
    assert (int(row), int(col)) == (1, 2)
    x, y = g.rc_to_xy(1, 2)
    assert float(x) == pytest.approx(0.125)
    assert float(y) == pytest.approx(0.075)


def test_gridspec_xy_to_rc_to_xy_within_half_cell():
    g = GridSpec(resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 12.0, size=(1000, 2))
    x, y = g.rc_to_xy(*g.xy_to_rc(pts[:, 0], pts[:, 1]))
    assert np.all(np.abs(x - pts[:, 0]) <= 0.05 + 1e-12)
    assert np.all(np.abs(y - pts[:, 1]) <= 0.05 + 1e-12)


def test_world_accepts_matching_shape():
    g = GridSpec(resolution=0.5, size_x=4.0, size_y=2.0, origin=(0.0, 0.0))
    w = World(np.zeros((4, 8)), g, (), "flat", 0)
    assert w.heightfield.shape == (g.ny, g.nx)


def test_world_rejects_wrong_shape():
    g = GridSpec(resolution=0.5, size_x=4.0, size_y=2.0, origin=(0.0, 0.0))  # (ny, nx) = (4, 8)
    with pytest.raises(ValueError, match="shape"):
        World(np.zeros((8, 4)), g, (), "flat", 0)


def test_world_heightfield_is_read_only_copy():
    g = GridSpec(resolution=0.5, size_x=4.0, size_y=2.0, origin=(0.0, 0.0))
    src = np.zeros((4, 8))
    w = World(src, g, (), "flat", 0)
    src[0, 0] = 1.0  # mutating the caller's array must not leak into the World
    assert w.heightfield[0, 0] == 0.0
    with pytest.raises(ValueError):
        w.heightfield[0, 0] = 1.0
