"""World/scenario generator tests. Expected values are derived by hand from the parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import ndimage

from terranav.config import SimConfig
from terranav.perception.backproject import backproject
from terranav.sim.scenario import (
    FAMILIES,
    POSE_CLEARANCE_M,
    easy_mask,
    generate_scenario,
    scan_poses,
)
from terranav.sim.sensor import capture
from terranav.sim.world import add_ramp, add_rough_patch, add_terrace, box_on_terrain
from terranav.types import Box, GridSpec, Scenario, SensorConfig, World

CFG = SimConfig()
GRID = CFG.terrain_grid
DISK_R = 0.75
KEEP_OUT_M = 1.0
SEEDS = range(8)
SEEDS_50 = range(50)


def cell_centers(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of every cell centre, each (ny, nx), computed by hand from the grid layout."""
    x = grid.origin[0] + (np.arange(grid.nx) + 0.5) * grid.resolution
    y = grid.origin[1] + (np.arange(grid.ny) + 0.5) * grid.resolution
    return np.meshgrid(x, y, indexing="xy")


def disk_mask(grid: GridSpec, center_xy, radius: float) -> np.ndarray:
    x, y = cell_centers(grid)
    return (x - center_xy[0]) ** 2 + (y - center_xy[1]) ** 2 <= radius**2


def rect_disk_distance(box, center_xy) -> float:
    """Distance from a disk centre to the closest point of the box's x-y footprint."""
    cx = min(max(center_xy[0], box.min_xyz[0]), box.max_xyz[0])
    cy = min(max(center_xy[1], box.min_xyz[1]), box.max_xyz[1])
    return math.hypot(center_xy[0] - cx, center_xy[1] - cy)


def zeros() -> np.ndarray:
    return np.zeros((GRID.ny, GRID.nx))


# --- Determinism / RNG hygiene -----------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_same_seed_is_deterministic(family):
    a = generate_scenario(family, 123, CFG)
    b = generate_scenario(family, 123, CFG)
    assert np.array_equal(a.world.heightfield, b.world.heightfield)
    assert a.world.boxes == b.world.boxes
    assert a.start_xy == b.start_xy
    assert a.goal_xy == b.goal_xy
    assert a.params == b.params


@pytest.mark.parametrize("family", [f for f in FAMILIES if f != "flat_obstacles"])
def test_different_seeds_give_different_heightfields(family):
    a = generate_scenario(family, 1, CFG)
    b = generate_scenario(family, 2, CFG)
    assert not np.array_equal(a.world.heightfield, b.world.heightfield)


def test_flat_obstacles_different_seeds_give_different_boxes():
    a = generate_scenario("flat_obstacles", 1, CFG)
    b = generate_scenario("flat_obstacles", 2, CFG)
    assert a.world.boxes != b.world.boxes


def test_same_seed_different_family_gives_different_worlds():
    worlds = [generate_scenario(f, 7, CFG).world for f in FAMILIES]
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            same_hf = np.array_equal(worlds[i].heightfield, worlds[j].heightfield)
            assert not (same_hf and worlds[i].boxes == worlds[j].boxes), (FAMILIES[i], FAMILIES[j])


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_gives_same_start_goal_across_families(seed):
    # Start/goal are the first rng draws, so the task is shared by every family.
    scenarios = [generate_scenario(f, seed, CFG) for f in FAMILIES]
    assert len({sc.start_xy for sc in scenarios}) == 1
    assert len({sc.goal_xy for sc in scenarios}) == 1


def test_generation_does_not_touch_global_rng():
    before = np.random.get_state()
    for family in FAMILIES:
        generate_scenario(family, 5, CFG)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# --- Scenario invariants -----------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_start_goal_disks_flat_and_clear(family, seed):
    sc = generate_scenario(family, seed, CFG)
    hf = sc.world.heightfield
    assert sc.world.grid == GRID
    assert hf.shape == (240, 240)
    for p in (sc.start_xy, sc.goal_xy):
        assert np.std(hf[disk_mask(GRID, p, DISK_R)]) < 1e-9
        for box in sc.world.boxes:
            assert rect_disk_distance(box, p) >= DISK_R
    assert 0.5 <= sc.start_xy[0] <= 1.5 and 2.0 <= sc.start_xy[1] <= 10.0
    assert 10.5 <= sc.goal_xy[0] <= 11.5 and 2.0 <= sc.goal_xy[1] <= 10.0


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.slow  # multi-seed
@pytest.mark.parametrize("seed", SEEDS_50)
def test_start_goal_surroundings_at_base_height(family, seed):
    sc = generate_scenario(family, seed, CFG)
    for p in (sc.start_xy, sc.goal_xy):
        assert np.all(sc.world.heightfield[disk_mask(GRID, p, DISK_R + KEEP_OUT_M)] == 0.0)
        for box in sc.world.boxes:
            assert rect_disk_distance(box, p) >= DISK_R + KEEP_OUT_M


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_boxes_inside_map_and_well_formed(family, seed):
    sc = generate_scenario(family, seed, CFG)
    for box in sc.world.boxes:
        assert box.min_xyz[0] >= 0.0 and box.max_xyz[0] <= 12.0
        assert box.min_xyz[1] >= 0.0 and box.max_xyz[1] <= 12.0
        assert box.min_xyz[2] < box.max_xyz[2]


def test_flat_obstacles_has_boxes_and_flat_terrain():
    sc = generate_scenario("flat_obstacles", 0, CFG)
    assert 4 <= len(sc.world.boxes) <= 8
    assert np.all(sc.world.heightfield == 0.0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.slow  # multi-seed
@pytest.mark.parametrize("seed", SEEDS_50)
def test_goal_reachable_over_easy_terrain(family, seed):
    sc = generate_scenario(family, seed, CFG)
    four_connected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labels, _ = ndimage.label(easy_mask(sc), structure=four_connected)
    sr, sc_ = (int(np.floor(v / GRID.resolution)) for v in sc.start_xy[::-1])
    gr, gc = (int(np.floor(v / GRID.resolution)) for v in sc.goal_xy[::-1])
    assert labels[sr, sc_] != 0
    assert labels[sr, sc_] == labels[gr, gc]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.slow  # multi-seed
@pytest.mark.parametrize("seed", SEEDS_50)
def test_straight_line_is_obstructed(family, seed):
    sc = generate_scenario(family, seed, CFG)
    s, g = np.array(sc.start_xy), np.array(sc.goal_xy)
    length = np.linalg.norm(g - s)
    dist = np.arange(0.0, length + 1e-9, 0.05)
    pts = s + (dist / length)[:, None] * (g - s)
    col = np.floor(pts[:, 0] / GRID.resolution).astype(int)
    row = np.floor(pts[:, 1] / GRID.resolution).astype(int)
    hit = np.abs(sc.world.heightfield[row, col]) > 0.02
    for b in sc.world.boxes:
        hit |= (
            (b.min_xyz[0] <= pts[:, 0])
            & (pts[:, 0] <= b.max_xyz[0])
            & (b.min_xyz[1] <= pts[:, 1])
            & (pts[:, 1] <= b.max_xyz[1])
        )
    assert hit.mean() >= 0.15


def feature_bbox(f) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of a logged feature's footprint, derived from its params."""
    if f["kind"] in ("ramp", "terrace"):
        (cx, cy), (length, width) = f["center_xy"], f["size_xy"]
        c, s = abs(math.cos(f["yaw_rad"])), abs(math.sin(f["yaw_rad"]))
        hx, hy = (length * c + width * s) / 2, (length * s + width * c) / 2
        return cx - hx, cx + hx, cy - hy, cy + hy
    if f["kind"] == "rough":
        (cx, cy), r = f["center_xy"], f["radius"]
        return cx - r, cx + r, cy - r, cy + r
    return f["min_xy"][0], f["max_xy"][0], f["min_xy"][1], f["max_xy"][1]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.slow  # multi-seed
@pytest.mark.parametrize("seed", SEEDS_50)
def test_features_do_not_overlap_and_stay_in_band(family, seed):
    feats = generate_scenario(family, seed, CFG).params["features"]
    lo, hi = {"flat_obstacles": (4, 8), "mixed": (3, 5)}.get(family, (3, 3))
    assert lo <= len(feats) <= hi
    boxes = [feature_bbox(f) for f in feats]
    for i, a in enumerate(boxes):
        assert 3.0 - 1e-9 <= a[0] and a[1] <= 9.0 + 1e-9
        for b in boxes[i + 1 :]:
            assert not (a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3])


@pytest.mark.slow  # 60 hypothesis seeds
@settings(deadline=None, max_examples=60)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1), family=st.sampled_from(FAMILIES))
def test_generation_succeeds_for_any_seed(seed, family):
    sc = generate_scenario(family, seed, CFG)
    assert np.all(np.isfinite(sc.world.heightfield))
    assert sc.family == family and sc.seed == seed


# --- Primitives --------------------------------------------------------------


def test_primitives_return_new_arrays():
    hf = zeros()
    noise = np.random.default_rng(0).standard_normal(hf.shape)
    outs = [
        add_ramp(hf, GRID, (6.0, 6.0), (3.0, 2.0), 0.0, 10.0, 0.4),
        add_rough_patch(hf, GRID, (6.0, 6.0), 1.0, 0.05, 0.2, noise),
        add_terrace(hf, GRID, (6.0, 6.0), (4.0, 3.0), 0.0, 0.1, 2, 0.5, 1),
    ]
    assert np.all(hf == 0.0)
    for out in outs:
        assert out is not hf and out.shape == hf.shape


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, 135.0])
def test_add_ramp_face_gradients_match_slope(yaw_deg):
    # 4 x 4 m footprint, 10 deg: peak = 2 * tan(10 deg) = 0.35 m < h_max, so four planar faces.
    slope_deg, center = 10.0, np.array([6.0, 6.0])
    yaw = math.radians(yaw_deg)
    hf = add_ramp(zeros(), GRID, tuple(center), (4.0, 4.0), yaw, slope_deg, 0.5)
    gy, gx = np.gradient(hf)  # (d/drow, d/dcol) = (d/dy, d/dx), per cell
    gx, gy = gx / GRID.resolution, gy / GRID.resolution
    t = math.tan(math.radians(slope_deg))
    a_hat = np.array([math.cos(yaw), math.sin(yaw)])  # local +a in world
    b_hat = np.array([-math.sin(yaw), math.cos(yaw)])  # local +b in world
    # Sample point 1.2 m from centre on each face (0.8 m from its outer edge, 0.85 m from
    # the diagonal ridges); height rises toward the centre, so gradient = -tan * outward.
    for outward in (a_hat, -a_hat, b_hat, -b_hat):
        mask = disk_mask(GRID, center + 1.2 * outward, 0.3)
        np.testing.assert_allclose(gx[mask], -t * outward[0], atol=1e-6)
        np.testing.assert_allclose(gy[mask], -t * outward[1], atol=1e-6)


def test_add_ramp_peak_is_h_max_when_footprint_large():
    # min half-size 2 m * tan(20 deg) = 0.73 m > h_max = 0.4 m -> flat top at 0.4 m.
    hf = add_ramp(zeros(), GRID, (6.0, 6.0), (4.0, 4.0), math.radians(20.0), 20.0, 0.4)
    assert hf.max() == pytest.approx(0.4, abs=1e-12)
    assert np.all(hf[disk_mask(GRID, (6.0, 6.0), 0.5)] == pytest.approx(0.4, abs=1e-12))


def test_add_ramp_small_footprint_peak_below_h_max():
    # 1 x 1 m, 20 deg: peak <= 0.5 * tan(20 deg) = 0.182 m < h_max = 0.4 m.
    hf = add_ramp(zeros(), GRID, (6.0, 6.0), (1.0, 1.0), 0.0, 20.0, 0.4)
    assert 0.0 < hf.max() <= 0.5 * math.tan(math.radians(20.0)) + 1e-12


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, 135.0])
def test_add_ramp_has_no_discontinuity(yaw_deg):
    slope_deg = 20.0
    hf = add_ramp(zeros(), GRID, (6.0, 6.0), (4.0, 3.0), math.radians(yaw_deg), slope_deg, 0.4)
    limit = math.tan(math.radians(slope_deg)) * GRID.resolution + 1e-12
    assert np.abs(np.diff(hf, axis=0)).max() <= limit
    assert np.abs(np.diff(hf, axis=1)).max() <= limit


@pytest.mark.parametrize("direction", [1, -1])
def test_add_terrace_axis_aligned_steps(direction):
    # 4 x 3 m footprint centred at (6, 6), yaw 0: x in [4, 8], y in [4.5, 7.5].
    h, depth, n = 0.15, 0.5, 3
    base = np.random.default_rng(1).uniform(-0.1, 0.1, size=(GRID.ny, GRID.nx))
    hf = add_terrace(base, GRID, (6.0, 6.0), (4.0, 3.0), 0.0, h, n, depth, direction)
    diff = hf - base
    x, y = cell_centers(GRID)
    outside = (np.abs(x - 6.0) > 2.0) | (np.abs(y - 6.0) > 1.5)
    np.testing.assert_array_equal(hf[outside], base[outside])
    # Row 120 (y = 6.025): d = min(2 - |x - 6|, 1.475). Ring edges at d = 0, 0.5, 1.0, i.e.
    # x = 4.0, 4.5, 5.0 (columns 79|80, 89|90, 99|100) and x = 8.0, 7.5, 7.0 on the far side.
    row = 120
    for k, col in enumerate([80, 90, 100]):
        assert diff[row, col] - diff[row, col - 1] == pytest.approx(direction * h, abs=1e-12)
        assert diff[row, col] == pytest.approx(direction * (k + 1) * h, abs=1e-12)
    for col in [140, 150, 160]:  # x = 7.0, 7.5, 8.0 edges between col-1 | col
        assert diff[row, col] - diff[row, col - 1] == pytest.approx(-direction * h, abs=1e-12)
    # Centre plateau: needs d >= (n - 1) * depth = 1.0, i.e. |x - 6| <= 1 and |y - 6| <= 0.5.
    # Within 0.45 m of the centre, d >= 1.5 - 0.45 = 1.05.
    centre = disk_mask(GRID, (6.0, 6.0), 0.45)
    np.testing.assert_allclose(diff[centre], direction * n * h, atol=1e-12)


@pytest.mark.parametrize("direction", [1, -1])
def test_add_terrace_rotated_jumps_are_one_step(direction):
    h, depth, n = 0.12, 0.4, 3
    hf = add_terrace(
        zeros(), GRID, (6.0, 6.0), (4.0, 3.5), math.radians(30.0), h, n, depth, direction
    )
    for axis in (0, 1):
        jumps = np.abs(np.diff(hf, axis=axis))
        nonzero = jumps[jumps > 1e-12]
        assert nonzero.size > 0
        np.testing.assert_allclose(nonzero, h, atol=1e-12)
    # min half-size 1.75 m >= (n - 1) * depth = 0.8 m, so the centre reaches n steps.
    centre = disk_mask(GRID, (6.0, 6.0), 0.5)
    np.testing.assert_allclose(hf[centre], direction * n * h, atol=1e-12)
    assert np.all(direction * hf >= 0.0)


def test_add_rough_patch_unchanged_outside_radius():
    rng = np.random.default_rng(3)
    base = rng.uniform(-0.1, 0.1, size=(GRID.ny, GRID.nx))
    noise = rng.standard_normal(base.shape)
    center, radius = (6.0, 5.0), 1.2
    hf = add_rough_patch(base, GRID, center, radius, 0.05, 0.2, noise)
    outside = ~disk_mask(GRID, center, radius)
    assert np.array_equal(hf[outside], base[outside])
    assert not np.array_equal(hf[~outside], base[~outside])


def test_box_sits_on_terrain():
    # One-step mesa over x in [6, 8], y in [4, 8] at z = 0.2. A footprint straddling its
    # x = 6 edge must get min_z = 0.0 and max_z = 0.2 + height.
    hf = add_terrace(zeros(), GRID, (7.0, 6.0), (2.0, 4.0), 0.0, 0.2, 1, 1.0, 1)
    box = box_on_terrain(GRID, hf, (5.5, 5.0), (6.5, 6.0), 0.5)
    assert box.min_xyz == pytest.approx((5.5, 5.0, 0.0))
    assert box.max_xyz == pytest.approx((6.5, 6.0, 0.7))


# --- Scan poses (placeholder survey pass) ------------------------------------------


def test_line_poses_skip_boxes():
    # Flat world, start (1, 2), goal (11, 7): direction (10, 5), yaw = atan2(5, 10). A box
    # straddles the 0.6 fraction point (7, 5): that pose is skipped, the other five remain.
    box = Box((6.5, 4.5, 0.0), (7.5, 5.5, 0.8))
    sc = Scenario(World(zeros(), GRID, (box,), "test", 0), (1.0, 2.0), (11.0, 7.0), "test", 0, {})
    poses = scan_poses(sc, k=6, strategy="line")
    assert len(poses) == 5
    yaw, pitch = math.atan2(5.0, 10.0), math.radians(25.0)
    axis = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)]
    for T, frac in zip(poses, [0.0, 0.2, 0.4, 0.8, 1.0], strict=True):
        np.testing.assert_allclose(T[:3, 3], [1 + 10 * frac, 2 + 5 * frac, 0.5], atol=1e-12)
        np.testing.assert_allclose(T[:3, 2], axis, atol=1e-12)  # optical z axis in world


def test_line_poses_follow_terrain():
    # One-step mesa 0.2 m high (2 x 2 m, flat top) under the 0.4 fraction point (5, 6) of a
    # start (1, 6) -> goal (11, 6) line: that pose sits 0.5 m above the mesa top.
    hf = add_terrace(zeros(), GRID, (5.0, 6.0), (2.0, 2.0), 0.0, 0.2, 1, 1.0, 1)
    sc = Scenario(World(hf, GRID, (), "test", 0), (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    zs = [T[2, 3] for T in scan_poses(sc, k=6, strategy="line")]
    np.testing.assert_allclose(zs, [0.5, 0.5, 0.7, 0.5, 0.5, 0.5], atol=1e-12)


def test_flank_poses_on_flat_ground():
    # Start (1, 6) -> goal (11, 6): heading 0, left normal +y. Rows at y = 6 +/- 2.5, along-
    # line fractions 0.1, 0.45, 0.8 -> x = 2, 5.5, 9. Left row (y = 8.5) yaws -30 deg (toward
    # the line), right row (y = 3.5) +30 deg. Order: by fraction, left before right.
    sc = Scenario(World(zeros(), GRID, (), "test", 0), (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    poses = scan_poses(sc, strategy="flank")
    expected_xy = [(2.0, 8.5), (2.0, 3.5), (5.5, 8.5), (5.5, 3.5), (9.0, 8.5), (9.0, 3.5)]
    p = math.radians(25.0)
    for i, (T, (x, y)) in enumerate(zip(poses, expected_xy, strict=True)):
        yaw = math.radians(-30.0 if i % 2 == 0 else 30.0)
        np.testing.assert_allclose(T[:3, 3], [x, y, 0.5], atol=1e-12)
        axis = [math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), -math.sin(p)]
        np.testing.assert_allclose(T[:3, 2], axis, atol=1e-12)


def clearance_ok(sc, x: float, y: float) -> bool:
    """Independent check: every cell within POSE_CLEARANCE_M of (x, y)'s cell is easy, and
    (x, y) is >= POSE_CLEARANCE_M from the map border. Uses a Euclidean distance transform
    of the non-easy cells (the area outside the map counts as non-easy)."""
    if not (0.5 <= x <= 11.5 and 0.5 <= y <= 11.5):
        return False
    easy = np.pad(easy_mask(sc), 1, constant_values=False)
    dist_cells = ndimage.distance_transform_edt(easy)[1:-1, 1:-1]
    r, c = int(y // GRID.resolution), int(x // GRID.resolution)
    return dist_cells[r, c] > POSE_CLEARANCE_M / GRID.resolution


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.slow  # multi-seed
@pytest.mark.parametrize("seed", SEEDS_50)
def test_flank_poses_placed_on_clear_easy_ground(family, seed):
    sc = generate_scenario(family, seed, CFG)
    poses = scan_poses(sc)  # default "flank+start"; must not raise
    assert len(poses) == 7
    for T in poses:
        x, y, z = T[:3, 3]
        assert clearance_ok(sc, x, y)
        r, c = int(y // GRID.resolution), int(x // GRID.resolution)
        # Clear easy ground is flat to within a few cm, so 0.5 m above the local cell height.
        assert abs(z - 0.5 - sc.world.heightfield[r, c]) < 0.05


@pytest.mark.slow  # captures from every pose of 10 scenarios
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [0, 1])
def test_capture_succeeds_from_every_flank_pose(family, seed):
    # Low resolution keeps this fast; what matters is that cast_frame accepts every pose
    # (above terrain, outside boxes) and returns some valid depth.
    sensor = SensorConfig(width=32, height=24)
    sc = generate_scenario(family, seed, CFG)
    for i, T in enumerate(scan_poses(sc, sensor=sensor)):
        frame, _ = capture(sc.world, T, sensor, np.random.default_rng((seed, i)))
        assert np.isfinite(frame.depth_z).any()


def test_flank_plus_start_poses():
    # Default strategy: frame 0 at the start (yaw toward the goal, 0.5 m up), then "flank".
    sc = Scenario(World(zeros(), GRID, (), "test", 0), (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    poses = scan_poses(sc)
    assert len(poses) == 7
    p = math.radians(25.0)
    np.testing.assert_allclose(poses[0][:3, 3], [1.0, 6.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(poses[0][:3, 2], [math.cos(p), 0.0, -math.sin(p)], atol=1e-12)
    for a, b in zip(poses[1:], scan_poses(sc, strategy="flank"), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        scan_poses(sc, k=6)  # flank+start needs an odd k


def test_start_pose_sees_ground_next_to_footprint():
    # Flat ground, camera 0.5 m up, pitch 25 deg. The bottom image row (v = H - 1) looks
    # 25 deg + atan(((H - 1) - cy) / fy) below the horizon, with fy = (W/2) / tan(hfov/2), so
    # the nearest visible ground lies d = 0.5 / tan(that) ahead: ~0.285 m for 320 x 240 and
    # 87 deg. On flat ground every pixel of a row shares the same forward distance.
    sensor = SensorConfig(noise_s0_m=0.0, noise_s1_per_m=0.0, dropout_prob=0.0)
    fy = (sensor.width / 2) / math.tan(sensor.hfov_rad / 2)
    cy = (sensor.height - 1) / 2
    dip = sensor.pitch_down_rad + math.atan((sensor.height - 1 - cy) / fy)
    d_near = 0.5 / math.tan(dip)
    assert d_near == pytest.approx(0.285, abs=0.001)
    sc = Scenario(World(zeros(), GRID, (), "test", 0), (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    T = scan_poses(sc, sensor=sensor)[0]
    frame, _ = capture(sc.world, T, sensor, np.random.default_rng((0, 0)))
    points, _ = backproject(frame)
    assert (points[:, 0] - 1.0).min() == pytest.approx(d_near, abs=1e-3)
    # So the ground just beyond a 0.4 m footprint is observed: map cells (0.1 m) centred
    # 0.45 and 0.55 m ahead, at y = 5.95 and 6.05.
    for x, y in ((1.45, 5.95), (1.45, 6.05), (1.55, 5.95), (1.55, 6.05)):
        near = (np.abs(points[:, 0] - x) <= 0.05) & (np.abs(points[:, 1] - y) <= 0.05)
        assert near.any(), (x, y)
