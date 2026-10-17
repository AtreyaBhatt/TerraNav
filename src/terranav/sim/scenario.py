"""Deterministic scenario recipes: (family, seed) -> Scenario.

All randomness flows from a single np.random.default_rng(seed). Start and goal are the
FIRST draws, before anything family-specific, so the same seed gives the same start/goal
in every family (families are then directly comparable on the same task).

Features are placed along the start->goal corridor: centre = start + f * (goal - start)
+ lateral * n_hat, f ~ U[0.3, 0.7], lateral ~ N(0, 0.8 m), then clipped so the footprint
stays inside the x band [3, 9] m and the map. In every family a feature is rejected if its
footprint overlaps another feature or comes within KEEP_OUT_MARGIN_M of the start/goal
disks, so those disks stay at base height 0 by construction. A layout is kept only if the
straight start->goal line actually crosses terrain or boxes (see _segment_relevance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import ndimage

from terranav.config import SimConfig
from terranav.frames import T_world_cam
from terranav.sim.world import (
    add_ramp,
    add_rough_patch,
    add_terrace,
    bilinear_sample,
    box_on_terrain,
    cell_centers_xy,
)
from terranav.types import GridSpec, Scenario, SensorConfig, World

FAMILIES = ("flat_obstacles", "slopes", "rough", "steps", "mixed")

FEATURE_BAND_X = (3.0, 9.0)  # metres; feature footprints start inside this x band
TASK_Y_RANGE = (2.0, 10.0)  # metres; start/goal y
START_X_RANGE = (0.75, 1.25)
GOAL_X_RANGE = (10.75, 11.25)
CLEAR_RADIUS_M = 0.75  # start/goal disks: base height 0, feature-free
KEEP_OUT_MARGIN_M = 1.0  # no feature footprint within this distance of a disk edge
CORRIDOR_FRACTION = (0.3, 0.7)
CORRIDOR_LATERAL_SIGMA_M = 0.8
RELEVANCE_MIN_FRACTION = 0.15  # of straight-line samples on terrain (|h| > 2 cm) or boxes
RELEVANCE_HEIGHT_M = 0.02
RELEVANCE_STEP_M = 0.05
MAX_FEATURE_ATTEMPTS = 50
# Layout attempts per scenario. Worst family (rough) succeeds ~30% per attempt, so 50
# attempts fail with probability ~1e-8; exhaustion raises rather than returning a bad layout.
MAX_SCENARIO_ATTEMPTS = 50


# ILLUSTRATIVE parameter ranges: chosen by eye to give varied, plausible scenes for a small
# ground robot. Not derived from any real dataset. Integer ranges are inclusive (lo, hi);
# float ranges are uniform (lo, hi).
@dataclass(frozen=True)
class FamilyRanges:
    n_features: tuple[int, int]
    kinds: tuple[str, ...]  # each feature's kind is drawn uniformly from these
    ramp_length_m: tuple[float, float] = (1.5, 3.0)
    ramp_width_m: tuple[float, float] = (1.0, 2.0)
    ramp_slope_deg: tuple[float, float] = (8.0, 25.0)
    ramp_h_max_m: tuple[float, float] = (0.3, 0.6)
    # Patches below ~3 cm RMS rarely clear the 2 cm relevance threshold along the line.
    rough_radius_m: tuple[float, float] = (0.8, 1.5)
    rough_rms_m: tuple[float, float] = (0.03, 0.08)
    rough_corr_len_m: tuple[float, float] = (0.1, 0.4)
    terrace_step_height_m: tuple[float, float] = (0.05, 0.2)
    # Terraces are kept <= 2.6 m a side so three fit side by side, non-overlapping, near the
    # corridor (larger ones made steps seeds 4, 22, 42, 43 unplaceable).
    terrace_n_steps: tuple[int, int] = (1, 2)
    terrace_step_depth_m: tuple[float, float] = (0.2, 0.4)
    terrace_top_m: tuple[float, float] = (0.4, 1.0)  # side of the flat top, before the rings
    box_size_m: tuple[float, float] = (0.4, 1.5)
    box_height_m: tuple[float, float] = (0.2, 1.0)


FAMILY_RANGES: dict[str, FamilyRanges] = {
    "flat_obstacles": FamilyRanges(n_features=(4, 8), kinds=("box",)),
    "slopes": FamilyRanges(n_features=(3, 3), kinds=("ramp",)),
    "rough": FamilyRanges(n_features=(3, 3), kinds=("rough",)),
    "steps": FamilyRanges(n_features=(3, 3), kinds=("terrace",)),
    "mixed": FamilyRanges(n_features=(3, 5), kinds=("ramp", "rough", "terrace", "box")),
}

BBox = tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax), metres


@dataclass(frozen=True)
class _Ctx:
    rng: np.random.Generator
    grid: GridSpec
    ranges: FamilyRanges
    start: tuple[float, float]
    goal: tuple[float, float]


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


def _count(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _corridor_point(ctx: _Ctx, half_x: float, half_y: float) -> tuple[float, float]:
    """Centre near the start->goal segment, clipped so a (2*half_x, 2*half_y) box fits."""
    s, g = np.asarray(ctx.start), np.asarray(ctx.goal)
    d = g - s
    n_hat = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    p = (
        s
        + _u(ctx.rng, CORRIDOR_FRACTION) * d
        + ctx.rng.normal(0.0, CORRIDOR_LATERAL_SIGMA_M) * n_hat
    )
    oy, sy = ctx.grid.origin[1], ctx.grid.size_y
    x = float(np.clip(p[0], FEATURE_BAND_X[0] + half_x, FEATURE_BAND_X[1] - half_x))
    y = float(np.clip(p[1], oy + half_y, oy + sy - half_y))
    return x, y


def _rotated_half_extents(size_xy: tuple[float, float], yaw_rad: float) -> tuple[float, float]:
    """Half extents along world x and y of a size_xy rectangle rotated by yaw_rad."""
    (length, width), c, s = size_xy, abs(math.cos(yaw_rad)), abs(math.sin(yaw_rad))
    return 0.5 * (length * c + width * s), 0.5 * (length * s + width * c)


def _sample_ramp(ctx: _Ctx) -> tuple[dict, BBox]:
    r = ctx.ranges
    length, width = _u(ctx.rng, r.ramp_length_m), _u(ctx.rng, r.ramp_width_m)
    yaw_rad = _u(ctx.rng, (-math.pi, math.pi))
    hx, hy = _rotated_half_extents((length, width), yaw_rad)
    cx, cy = _corridor_point(ctx, hx, hy)
    p = {
        "center_xy": (cx, cy),
        "size_xy": (length, width),
        "yaw_rad": yaw_rad,
        "slope_deg": _u(ctx.rng, r.ramp_slope_deg),
        "h_max_m": _u(ctx.rng, r.ramp_h_max_m),
    }
    return p, (cx - hx, cx + hx, cy - hy, cy + hy)


def _sample_rough(ctx: _Ctx) -> tuple[dict, BBox]:
    r = ctx.ranges
    radius = _u(ctx.rng, r.rough_radius_m)
    cx, cy = _corridor_point(ctx, radius, radius)
    p = {
        "center_xy": (cx, cy),
        "radius": radius,
        "rms_m": _u(ctx.rng, r.rough_rms_m),
        "corr_len_m": _u(ctx.rng, r.rough_corr_len_m),
    }
    return p, (cx - radius, cx + radius, cy - radius, cy + radius)


def _sample_terrace(ctx: _Ctx) -> tuple[dict, BBox]:
    r = ctx.ranges
    n_steps = _count(ctx.rng, r.terrace_n_steps)
    depth = _u(ctx.rng, r.terrace_step_depth_m)
    rings = 2 * n_steps * depth  # both sides; guarantees the centre reaches n_steps levels
    size = (rings + _u(ctx.rng, r.terrace_top_m), rings + _u(ctx.rng, r.terrace_top_m))
    yaw_rad = _u(ctx.rng, (-math.pi, math.pi))
    hx, hy = _rotated_half_extents(size, yaw_rad)
    cx, cy = _corridor_point(ctx, hx, hy)
    p = {
        "center_xy": (cx, cy),
        "size_xy": size,
        "yaw_rad": yaw_rad,
        "step_height_m": _u(ctx.rng, r.terrace_step_height_m),
        "n_steps": n_steps,
        "step_depth_m": depth,
        "direction": int(ctx.rng.choice([1, -1])),
    }
    return p, (cx - hx, cx + hx, cy - hy, cy + hy)


def _sample_box(ctx: _Ctx) -> tuple[dict, BBox]:
    r = ctx.ranges
    sx, sy = _u(ctx.rng, r.box_size_m), _u(ctx.rng, r.box_size_m)
    height = _u(ctx.rng, r.box_height_m)
    cx, cy = _corridor_point(ctx, sx / 2, sy / 2)
    p = {
        "min_xy": (cx - sx / 2, cy - sy / 2),
        "max_xy": (cx + sx / 2, cy + sy / 2),
        "height_m": height,
    }
    return p, (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2)


_SAMPLERS = {
    "ramp": _sample_ramp,
    "rough": _sample_rough,
    "terrace": _sample_terrace,
    "box": _sample_box,
}


def _overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _hits_disk(bb: BBox, cx: float, cy: float, radius: float) -> bool:
    px, py = min(max(cx, bb[0]), bb[1]), min(max(cy, bb[2]), bb[3])
    return math.hypot(cx - px, cy - py) < radius


def _accept(ctx: _Ctx, bb: BBox, placed: list[tuple[str, dict, BBox]]) -> bool:
    """Placement rules shared by every family and feature kind."""
    oy, sy = ctx.grid.origin[1], ctx.grid.size_y
    eps = 1e-9
    if bb[0] < FEATURE_BAND_X[0] - eps or bb[1] > FEATURE_BAND_X[1] + eps:
        return False  # e.g. a rotated footprint too large for the band
    if bb[2] < oy - eps or bb[3] > oy + sy + eps:
        return False
    keep_out = CLEAR_RADIUS_M + KEEP_OUT_MARGIN_M
    if any(_hits_disk(bb, *c, keep_out) for c in (ctx.start, ctx.goal)):
        return False
    return not any(_overlap(bb, other) for _, _, other in placed)


def _segment_relevance(hf, grid, boxes, start, goal) -> float:
    """Fraction of points every RELEVANCE_STEP_M along start->goal on terrain or in a box."""
    s, g = np.asarray(start), np.asarray(goal)
    length = float(np.linalg.norm(g - s))
    t = np.arange(int(length // RELEVANCE_STEP_M) + 1) * RELEVANCE_STEP_M / length
    pts = s + t[:, None] * (g - s)
    row, col = grid.xy_to_rc(pts[:, 0], pts[:, 1])
    hit = np.abs(hf[row, col]) > RELEVANCE_HEIGHT_M
    for b in boxes:
        hit |= (
            (pts[:, 0] >= b.min_xyz[0])
            & (pts[:, 0] <= b.max_xyz[0])
            & (pts[:, 1] >= b.min_xyz[1])
            & (pts[:, 1] <= b.max_xyz[1])
        )
    return float(hit.mean())


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def generate_scenario(family: str, seed: int, cfg: SimConfig | None = None) -> Scenario:
    """Build the scenario for (family, seed) on cfg.terrain_grid. Deterministic.

    Heights in metres on a base of z = 0; start/goal (x, y) in world metres. The same seed
    gives the same start/goal for every family. Up to MAX_SCENARIO_ATTEMPTS feature layouts
    are tried until the straight start->goal line is >= RELEVANCE_MIN_FRACTION obstructed
    and at least the family's minimum number of features was placed; the number of attempts
    is logged in params. Raises RuntimeError if no attempt succeeds.
    """
    if family not in FAMILY_RANGES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    cfg = cfg if cfg is not None else SimConfig()
    grid = cfg.terrain_grid
    rng = np.random.default_rng(seed)
    # Start/goal are drawn first, before any family-specific draw.
    start = (_u(rng, START_X_RANGE), _u(rng, TASK_Y_RANGE))
    goal = (_u(rng, GOAL_X_RANGE), _u(rng, TASK_Y_RANGE))
    ctx = _Ctx(rng, grid, FAMILY_RANGES[family], start, goal)

    for attempt in range(1, MAX_SCENARIO_ATTEMPTS + 1):
        placed: list[tuple[str, dict, BBox]] = []
        n_wanted = _count(rng, ctx.ranges.n_features)
        for _ in range(n_wanted):
            for _ in range(MAX_FEATURE_ATTEMPTS):
                kinds = ctx.ranges.kinds  # kind re-drawn per attempt: a blocked kind can yield
                kind = kinds[0] if len(kinds) == 1 else str(rng.choice(kinds))
                p, bb = _SAMPLERS[kind](ctx)
                if _accept(ctx, bb, placed):
                    placed.append((kind, p, bb))
                    break

        hf = np.zeros((grid.ny, grid.nx), dtype=np.float64)
        for kind, p, _ in placed:
            if kind == "ramp":
                hf = add_ramp(hf, grid, **p)
            elif kind == "rough":
                hf = add_rough_patch(hf, grid, **p, noise=rng.standard_normal(hf.shape))
            elif kind == "terrace":
                hf = add_terrace(hf, grid, **p)
        boxes = tuple(
            box_on_terrain(grid, hf, p["min_xy"], p["max_xy"], p["height_m"])
            for kind, p, _ in placed
            if kind == "box"
        )
        relevance = _segment_relevance(hf, grid, boxes, start, goal)
        if relevance >= RELEVANCE_MIN_FRACTION and len(placed) >= ctx.ranges.n_features[0]:
            break
    else:
        raise RuntimeError(
            f"{family} seed {seed}: no valid layout in {MAX_SCENARIO_ATTEMPTS} attempts"
        )

    params = {
        "start_xy": start,
        "goal_xy": goal,
        "features": [{"kind": kind, **p} for kind, p, _ in placed],
        "attempts": attempt,
        "relevance": relevance,
    }
    # TODO(M2): reject unsolvable scenarios once the planner exists.
    world = World(hf, grid, boxes, family, seed)
    return Scenario(world, start, goal, family, seed, _freeze(params))


# --- Easy ground -------------------------------------------------------------------

# ILLUSTRATIVE "easy terrain" thresholds: plausible for a small ground robot, not derived
# from any vehicle model. Used for reachability checks and scan-pose placement.
EASY_MAX_SLOPE_DEG = 15.0
EASY_MAX_JUMP_M = 0.03  # largest height difference to any 4-neighbour cell
EASY_BOX_DILATION_M = 0.3  # clearance around box footprints


def easy_mask(scenario: Scenario) -> np.ndarray:
    """(ny, nx) bool on the terrain grid: gentle, step-free cells away from boxes.

    A cell is easy iff its ground-truth gradient magnitude <= tan(EASY_MAX_SLOPE_DEG), its
    height differs from every 4-neighbour by <= EASY_MAX_JUMP_M, and its centre lies outside
    every box footprint dilated by EASY_BOX_DILATION_M (square dilation).
    """
    world = scenario.world
    hf, grid = world.heightfield, world.grid
    gy, gx = np.gradient(hf, grid.resolution)  # d/drow = d/dy, d/dcol = d/dx
    easy = np.hypot(gx, gy) <= math.tan(math.radians(EASY_MAX_SLOPE_DEG))
    pad = np.pad(hf, 1, mode="edge")
    neighbours = (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:])
    easy &= np.max([np.abs(n - hf) for n in neighbours], axis=0) <= EASY_MAX_JUMP_M
    x, y = cell_centers_xy(grid)
    d = EASY_BOX_DILATION_M
    for b in world.boxes:
        easy &= ~(
            (x >= b.min_xyz[0] - d)
            & (x <= b.max_xyz[0] + d)
            & (y >= b.min_xyz[1] - d)
            & (y <= b.max_xyz[1] + d)
        )
    return easy


# --- Scan poses -------------------------------------------------------------------

SCAN_STRATEGIES = ("flank+start", "flank", "line")
DEFAULT_SCAN_K = {"flank+start": 7, "flank": 6, "line": 6}  # poses per strategy if k is None
# Flank survey STARTING VALUES, to be tuned against measured coverage (scripts/eval_scan.py).
FLANK_LATERAL_OFFSET_M = 2.5  # d_lat: distance of each row from the start->goal line
FLANK_TOE_IN_DEG = 30.0  # theta_in: rows look this far inward from the line's heading
FLANK_FRACTIONS = (0.1, 0.8)  # first/last along-line fraction of each row
POSE_CLEARANCE_M = 0.5  # poses need easy ground within this radius and this far from the border
POSE_SEARCH_STEP_M = 0.1
POSE_SEARCH_AWAY_M = 3.0  # how far beyond d_lat the lateral search may go
# If the whole lateral search fails, it is repeated at these along-line fraction shifts, in
# order (needed when one side of the line is boxed in between features and the map edge).
FLANK_FRACTION_SHIFTS = (0.0, -0.05, 0.05, -0.10, 0.10, -0.15, 0.15, -0.20, 0.20, -0.25, 0.25)


def pose_ground_mask(scenario: Scenario) -> np.ndarray:
    """(ny, nx) bool: easy_mask eroded by a POSE_CLEARANCE_M disk (outside the map = not easy).

    Because the erosion treats the area outside the map as not easy, every kept cell is also
    at least POSE_CLEARANCE_M from the map border.
    """
    grid = scenario.world.grid
    r = round(POSE_CLEARANCE_M / grid.resolution)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = xx**2 + yy**2 <= r**2
    return ndimage.binary_erosion(easy_mask(scenario), structure=disk, border_value=0)


def _pose_ok(mask: np.ndarray, grid: GridSpec, x: float, y: float) -> bool:
    x0, y0 = grid.origin
    m = POSE_CLEARANCE_M
    if not (x0 + m <= x <= x0 + grid.size_x - m and y0 + m <= y <= y0 + grid.size_y - m):
        return False
    r, c = grid.xy_to_rc(x, y)
    return bool(mask[int(r), int(c)])


def _camera_pose(scenario: Scenario, x: float, y: float, yaw_rad: float, sensor) -> np.ndarray:
    world = scenario.world
    z = float(bilinear_sample(world.heightfield, world.grid, x, y)) + sensor.mount_height_m
    return T_world_cam(np.array([x, y, z]), yaw_rad, sensor.pitch_down_rad)


# TODO(M2): revisit scan strategy (this is a survey pass with known poses).
def scan_poses(
    scenario: Scenario,
    k: int | None = None,
    strategy: str = "flank+start",
    sensor: SensorConfig | None = None,
    d_lat_m: float = FLANK_LATERAL_OFFSET_M,
    theta_in_deg: float = FLANK_TOE_IN_DEG,
) -> list[np.ndarray]:
    """PLACEHOLDER survey: known camera poses (4x4 float64 T_world_cam) for one scenario.

    Every camera is sensor.mount_height_m above the bilinear terrain at its (x, y) and pitched
    down by sensor.pitch_down_rad. With psi the heading of the start->goal line and n_hat its
    left normal:

    "flank+start" (default, k = 7): one pose at the start (yaw psi), then the k - 1 poses of
      "flank". The start pose sees the ground next to the robot's footprint, which no flank
      pose does; frame 0 is the start pose.
    "flank" (k = 6): two rows of k/2 poses at lateral offsets +d_lat_m (left) and -d_lat_m
      (right) from the line, at along-line fractions linspace(0.1, 0.8, k/2) (0.1, 0.45, 0.8
      for k = 6). Left-row yaw = psi - theta_in, right-row yaw = psi + theta_in (both turned
      toward the line). A pose must lie in pose_ground_mask; otherwise the lateral offset is
      searched in POSE_SEARCH_STEP_M steps, first toward the line (down to 0), then away (up to
      d_lat_m + POSE_SEARCH_AWAY_M). If no offset works, the same search is repeated at the
      shifted fractions FLANK_FRACTION_SHIFTS (clipped to [0, 1]); ValueError if all fail.
      Order: by fraction, left before right.
    "line": comparison baseline. k poses at fractions linspace(0, 1, k) on the line, yaw psi.
      A pose inside a box footprint or off easy_mask is SKIPPED, so k - len(result) poses
      were skipped.

    k counts ALL poses; None selects DEFAULT_SCAN_K[strategy].
    Frame i is captured with the per-frame generator np.random.default_rng((scenario.seed, i)).
    """
    if strategy not in SCAN_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {SCAN_STRATEGIES}")
    k = DEFAULT_SCAN_K[strategy] if k is None else k
    sensor = sensor if sensor is not None else SensorConfig()
    s, g = np.asarray(scenario.start_xy), np.asarray(scenario.goal_xy)
    t_hat = (g - s) / np.linalg.norm(g - s)
    n_hat = np.array([-t_hat[1], t_hat[0]])  # left of the start->goal heading
    psi = math.atan2(t_hat[1], t_hat[0])
    world = scenario.world

    if strategy == "flank+start":
        if k < 3 or k % 2 == 0:
            raise ValueError(f"flank+start needs an odd k >= 3 (start + even flank), got {k}")
        start_pose = _camera_pose(scenario, *s, psi, sensor)
        flank = scan_poses(scenario, k - 1, "flank", sensor, d_lat_m, theta_in_deg)
        return [start_pose, *flank]

    if strategy == "line":
        if k < 2:
            raise ValueError(f"line strategy needs k >= 2, got {k}")
        easy = easy_mask(scenario)
        poses = []
        for frac in np.linspace(0.0, 1.0, k):
            x, y = s + frac * (g - s)
            in_box = any(
                b.min_xyz[0] <= x <= b.max_xyz[0] and b.min_xyz[1] <= y <= b.max_xyz[1]
                for b in world.boxes
            )
            r, c = world.grid.xy_to_rc(x, y)
            if not in_box and easy[int(r), int(c)]:
                poses.append(_camera_pose(scenario, x, y, psi, sensor))
        return poses

    if k < 2 or k % 2:
        raise ValueError(f"flank strategy needs an even k >= 2, got {k}")
    mask = pose_ground_mask(scenario)
    step = POSE_SEARCH_STEP_M
    toward = [d_lat_m - step * j for j in range(1, int(d_lat_m / step + 1e-9) + 1)]
    away = [d_lat_m + step * j for j in range(1, round(POSE_SEARCH_AWAY_M / step) + 1)]
    offsets = [d_lat_m, *toward, *away]
    theta = math.radians(theta_in_deg)
    poses = []
    for frac in np.linspace(*FLANK_FRACTIONS, k // 2):
        for side in (1.0, -1.0):  # left row, then right row
            xy = _flank_position(mask, world.grid, s, g, n_hat, frac, side, offsets)
            if xy is None:
                raise ValueError(
                    f"{scenario.family} seed {scenario.seed}: no valid flank pose near fraction "
                    f"{frac:.2f} on the {'left' if side > 0 else 'right'} side"
                )
            poses.append(_camera_pose(scenario, *xy, psi - side * theta, sensor))
    return poses


def _flank_position(mask, grid, s, g, n_hat, frac, side, offsets) -> tuple[float, float] | None:
    """First valid (x, y) over fraction shifts (outer loop) and lateral offsets (inner)."""
    for shift in FLANK_FRACTION_SHIFTS:
        base = s + float(np.clip(frac + shift, 0.0, 1.0)) * (g - s)
        for d in offsets:
            x, y = base + side * d * n_hat
            if _pose_ok(mask, grid, x, y):
                return float(x), float(y)
    return None
