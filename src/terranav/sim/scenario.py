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

from terranav.config import SimConfig
from terranav.sim.world import (
    add_ramp,
    add_rough_patch,
    add_terrace,
    box_on_terrain,
)
from terranav.types import GridSpec, Scenario, World

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
