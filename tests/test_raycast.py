"""Ray-caster tests. Expected values are derived by hand (closed-form geometry)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terranav.frames import T_world_cam
from terranav.sim.raycast import HIT_BOX, HIT_NONE, HIT_TERRAIN, cast_frame, range_to_depth
from terranav.types import Box, CameraIntrinsics, GridSpec, World

GRID = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
HFOV_RAD = math.radians(87.0)
# Odd sizes so the optical axis passes through an actual pixel centre (cx, cy integers).
BIG = CameraIntrinsics.from_hfov(321, 241, HFOV_RAD)  # cx = 160, cy = 120
SMALL = CameraIntrinsics.from_hfov(41, 31, HFOV_RAD)  # cx = 20, cy = 15
# max_depth_z_m that cuts nothing: no point of the 12 x 12 m map is farther than ~17 m
# from a camera over it, and depth_z <= range_t. (Larger values only lengthen the march.)
FAR = 20.0


def flat_world(h: float = 0.0, boxes: tuple[Box, ...] = ()) -> World:
    return World(np.full((GRID.ny, GRID.nx), h), GRID, boxes, "test", 0)


def pose(x, y, z, yaw_deg=0.0, pitch_down_deg=0.0) -> np.ndarray:
    return T_world_cam(np.array([x, y, z]), math.radians(yaw_deg), math.radians(pitch_down_deg))


def centre(intr: CameraIntrinsics) -> tuple[int, int]:
    """(v, u) index of the optical-axis pixel for odd-sized test images."""
    return int(intr.cy), int(intr.cx)


# --- Result structure ----------------------------------------------------------


def test_result_shapes_dtypes_and_read_only():
    res = cast_frame(flat_world(), pose(2, 6, 0.5, 0, 25), SMALL, FAR)
    H, W = SMALL.height, SMALL.width
    assert res.range_t.shape == (H, W) and res.range_t.dtype == np.float64
    assert res.points_world.shape == (H, W, 3) and res.points_world.dtype == np.float64
    assert res.normals_world.shape == (H, W, 3) and res.normals_world.dtype == np.float64
    assert res.hit_kind.shape == (H, W) and res.hit_kind.dtype == np.int8
    for arr in (res.range_t, res.points_world, res.normals_world, res.hit_kind):
        with pytest.raises(ValueError):
            arr.flat[0] = 0


def test_misses_are_consistent():
    res = cast_frame(flat_world(), pose(2, 6, 0.5, 0, 25), SMALL, FAR)
    miss = res.hit_kind == HIT_NONE
    assert miss.any() and (~miss).any()
    assert np.all(np.isnan(res.range_t[miss]))
    assert np.all(np.isnan(res.points_world[miss]))
    assert np.all(np.isnan(res.normals_world[miss]))
    assert np.all(np.isfinite(res.range_t[~miss]))
    assert np.all(np.isfinite(res.points_world[~miss]))
    np.testing.assert_allclose(np.linalg.norm(res.normals_world[~miss], axis=-1), 1.0)


# --- Flat terrain ----------------------------------------------------------------


def test_flat_ground_centre_pixel_and_centre_column():
    # Camera 0.5 m above h = 0, pitched down 25 deg. A ray depressed by angle a below the
    # horizontal hits the ground at range 0.5 / sin(a). Pixel v in the centre column is
    # depressed by 25 deg + atan((v - cy) / fy).
    res = cast_frame(flat_world(), pose(2, 6, 0.5, 0, 25), BIG, FAR)
    v0, u0 = centre(BIG)
    expected = 0.5 / math.sin(math.radians(25.0))
    assert expected == pytest.approx(1.1831, abs=1e-4)
    assert res.range_t[v0, u0] == pytest.approx(expected, abs=1e-9)
    depth = range_to_depth(res.range_t, BIG)
    assert depth[v0, u0] == pytest.approx(res.range_t[v0, u0], abs=1e-12)
    for v in (60, 90, 120, 150, 200, 240):
        a = math.radians(25.0) + math.atan((v - BIG.cy) / BIG.fy)
        assert res.range_t[v, u0] == pytest.approx(0.5 / math.sin(a), abs=1e-9), v
        assert res.hit_kind[v, u0] == HIT_TERRAIN
        np.testing.assert_allclose(res.normals_world[v, u0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(res.points_world[v, u0, 2], 0.0, atol=1e-9)
        # depth_z = range_t * cos(angle off the optical axis) for the centre column.
        off_axis = math.atan((v - BIG.cy) / BIG.fy)
        assert depth[v, u0] == pytest.approx(res.range_t[v, u0] * math.cos(off_axis), abs=1e-9)


def test_rows_above_horizon_miss():
    # Pixel rows with (v - cy)/fy <= -tan(25 deg) look at or above the horizon.
    res = cast_frame(flat_world(), pose(6, 6, 0.5, 0, 25), BIG, FAR)
    v_horizon = BIG.cy - BIG.fy * math.tan(math.radians(25.0))  # ~41.1
    above = np.arange(BIG.height) < v_horizon - 0.5
    assert above.sum() >= 40
    assert np.all(np.isnan(res.range_t[above]))
    assert np.all(res.hit_kind[above] == HIT_NONE)


def test_hit_beyond_max_depth_is_nan():
    # Pitch 5 deg from 0.5 m: centre ray hits at range = depth_z = 0.5 / sin(5 deg) = 5.737 m.
    T = pose(1, 6, 0.5, 0, 5)
    v0, u0 = centre(SMALL)
    d = 0.5 / math.sin(math.radians(5.0))
    assert cast_frame(flat_world(), T, SMALL, 6.0).range_t[v0, u0] == pytest.approx(d, abs=1e-9)
    cut = cast_frame(flat_world(), T, SMALL, 5.0)
    assert np.isnan(cut.range_t[v0, u0])
    assert cut.hit_kind[v0, u0] == HIT_NONE
    # Nothing that survives the cut has depth_z beyond it.
    assert np.nanmax(range_to_depth(cut.range_t, SMALL)) <= 5.0


def test_camera_at_or_below_terrain_raises():
    with pytest.raises(ValueError):
        cast_frame(flat_world(1.0), pose(6, 6, 0.5), SMALL, FAR)
    with pytest.raises(ValueError):
        cast_frame(flat_world(1.0), pose(6, 6, 1.0), SMALL, FAR)


# --- Inclined plane -------------------------------------------------------------


def test_inclined_plane_matches_closed_form():
    # h = tan(s) * x is reproduced exactly by bilinear interpolation of cell-centre samples.
    s, p = math.radians(10.0), math.radians(20.0)
    x_centres = (np.arange(GRID.nx) + 0.5) * GRID.resolution
    hf = np.tile(math.tan(s) * x_centres, (GRID.ny, 1))
    world = World(hf, GRID, (), "test", 0)
    x0, z0 = 2.0, 1.5
    res = cast_frame(world, pose(x0, 6, z0, 0, 20), SMALL, FAR)
    v0, u0 = centre(SMALL)
    # Ray: x = x0 + t cos p, z = z0 - t sin p; plane: z = tan(s) x.
    t = (z0 - math.tan(s) * x0) / (math.sin(p) + math.tan(s) * math.cos(p))
    assert res.range_t[v0, u0] == pytest.approx(t, abs=1e-4)
    n = np.array([-math.tan(s), 0.0, 1.0]) / math.hypot(math.tan(s), 1.0)
    np.testing.assert_allclose(res.normals_world[v0, u0], n, atol=1e-9)


# --- Orientation guard ----------------------------------------------------------


def pillar_world() -> World:
    # 2 m pillar on cells whose centres lie in x in [5.85, 6.15], y in [7.85, 8.15]:
    # columns 117..122, rows 157..162. Bilinear height reaches 0.5 m at x = 5.825 + 0.25 * 0.05.
    hf = np.zeros((GRID.ny, GRID.nx))
    hf[157:163, 117:123] = 2.0
    return World(hf, GRID, (), "test", 0)


def test_pillar_hit_looking_along_x():
    res = cast_frame(pillar_world(), pose(2, 8, 0.5, 0, 0), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_TERRAIN
    np.testing.assert_allclose(res.points_world[v0, u0], [5.8375, 8.0, 0.5], atol=1e-6)


def test_pillar_hit_looking_along_y():
    # Facing +y from (6, 2): the near face ramps from y = 7.825 to 7.875.
    res = cast_frame(pillar_world(), pose(6, 2, 0.5, 90, 0), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_TERRAIN
    np.testing.assert_allclose(res.points_world[v0, u0], [6.0, 7.8375, 0.5], atol=1e-6)


def test_pillar_not_at_swapped_location():
    # Looking along x at y = 6 must pass x = 8 without hitting anything (a swapped pillar
    # would sit at x = 8, y = 6).
    res = cast_frame(pillar_world(), pose(2, 6, 0.5, 0, 0), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_NONE


# --- Boxes -----------------------------------------------------------------------


@pytest.mark.filterwarnings("error")
def test_box_face_hit():
    # Pitch 0, centre ray along +x at z = 0.5, y = 6: parallel to the y and z slabs and
    # inside both, enters the x slab at x = 5.
    box = Box((5.0, 5.0, 0.0), (6.0, 7.0, 1.0))
    res = cast_frame(flat_world(boxes=(box,)), pose(2, 6, 0.5, 0, 0), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_BOX
    assert res.range_t[v0, u0] == pytest.approx(3.0, abs=1e-12)
    np.testing.assert_allclose(res.normals_world[v0, u0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(res.points_world[v0, u0], [5.0, 6.0, 0.5], atol=1e-12)


@pytest.mark.filterwarnings("error")
def test_box_parallel_ray_outside_slab_misses():
    box = Box((5.0, 7.0, 0.0), (6.0, 8.0, 1.0))  # y slab [7, 8]; ray at y = 6, d_y = 0
    res = cast_frame(flat_world(boxes=(box,)), pose(2, 6, 0.5, 0, 0), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_NONE


@pytest.mark.filterwarnings("error")
def test_box_origin_on_slab_plane_no_nan():
    # Camera y = 6.0 lies exactly on the box's y = 6 plane; centre ray has d_y = 0.
    box = Box((5.0, 6.0, 0.0), (6.0, 7.0, 1.0))
    res = cast_frame(flat_world(boxes=(box,)), pose(2, 6, 0.5, 0, 0), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_BOX
    assert res.range_t[v0, u0] == pytest.approx(3.0, abs=1e-12)
    hit = res.hit_kind != HIT_NONE
    assert np.all(np.isfinite(res.points_world[hit]))
    assert np.all(np.isfinite(res.normals_world[hit]))


@pytest.mark.filterwarnings("error")
def test_box_behind_camera_misses():
    box = Box((0.0, 5.0, 0.0), (1.0, 7.0, 1.0))  # behind a camera at x = 2 facing +x
    res = cast_frame(flat_world(boxes=(box,)), pose(2, 6, 0.5, 0, 0), SMALL, FAR)
    assert not np.any(res.hit_kind == HIT_BOX)


def test_box_occludes_terrain():
    # Centre ray (pitch 25 deg) would hit the ground at x = 2 + 0.5 / tan(25 deg) = 3.07;
    # the box face at x = 2.5 is reached first, at range 0.5 / cos(25 deg), z = 0.267.
    box = Box((2.5, 5.5, 0.0), (2.8, 6.5, 1.0))
    res = cast_frame(flat_world(boxes=(box,)), pose(2, 6, 0.5, 0, 25), SMALL, FAR)
    v0, u0 = centre(SMALL)
    assert res.hit_kind[v0, u0] == HIT_BOX
    assert res.range_t[v0, u0] == pytest.approx(0.5 / math.cos(math.radians(25.0)), abs=1e-12)
    np.testing.assert_allclose(res.normals_world[v0, u0], [-1.0, 0.0, 0.0])


def test_box_top_face_normal():
    # Looking straight-ish down onto a box top: pitch 60 deg from z = 2 over a 0.5 m box.
    box = Box((2.5, 5.0, 0.0), (4.0, 7.0, 0.5))
    res = cast_frame(flat_world(boxes=(box,)), pose(2, 6, 2.0, 0, 60), SMALL, FAR)
    v0, u0 = centre(SMALL)
    # Centre ray drops 1.5 m to the top at range 1.5 / sin(60 deg); x = 2 + 1.5/tan(60) = 2.87.
    assert res.hit_kind[v0, u0] == HIT_BOX
    assert res.range_t[v0, u0] == pytest.approx(1.5 / math.sin(math.radians(60.0)), abs=1e-12)
    np.testing.assert_allclose(res.normals_world[v0, u0], [0.0, 0.0, 1.0])
