"""Independent path evaluator: judges a path against the raw simulated world.

Deliberately does NOT use perception, the traversability/feature definitions or the planning
cost model (enforced by tests/test_architecture.py): a metric built from the definitions the
planners use cannot reveal a flaw in them (docs/decisions.md D18). Everything here comes from
the heightfield, the boxes and the generator's feature parameters (scenario.params).
World frame, metres; path_xy is a polyline of world (x, y) vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from scipy import ndimage

from terranav.sim.world import bilinear_sample
from terranav.types import Scenario

SAMPLE_STEP_M = 0.02  # polyline sampling interval for height / slope / clearance metrics
RISE_WINDOW_M = 0.2  # max_rise compares samples at most this far apart along the path
FEATURE_KINDS = ("ramp", "terrace", "rough")
# ILLUSTRATIVE: a jump between 4-neighbour terrain cells steeper than this, locally, is a STEP
# (height discontinuity), measured by max_rise and excluded from the slope metrics. 40 deg is
# above the 30 deg slope limit (so steeper-than-allowed slopes are still measured as slopes) and
# below the 45 deg of the smallest terrace step (0.05 m over one 0.05 m cell).
# Gap: a sustained slope of 40-45 deg is classed as steps but rises < max_step over 0.2 m, so it
# would not be flagged; the generator's ramps are <= 25 deg.
DISCONTINUITY_SLOPE_DEG = 40.0
# ILLUSTRATIVE: violation slopes are measured on the heightfield smoothed by a Gaussian of this
# std (m), the scale of the robot footprint: a vehicle rests on its contact patch, not on
# 5 cm texture. The raw-gradient maximum is reported as a diagnostic only.
SLOPE_SCALE_M = 0.1


@dataclass(frozen=True)
class PhysicalLimits:
    """ILLUSTRATIVE robot limits: the cost model's values, but checked here on raw ground truth."""

    robot_radius_m: float = 0.3
    max_step_m: float = 0.20
    max_slope_deg: float = 30.0


@dataclass(frozen=True)
class PathMetrics:
    """Ground-truth metrics of one path. Angles in degrees, lengths in metres.

    min_clearance_m: smallest 2D distance from the path to any box footprint (inf if no boxes);
        collision = min_clearance_m < robot radius, at collision_xy.
    max_rise_m: largest |z(s2) - z(s1)| over samples <= RISE_WINDOW_M apart (z = bilinear terrain,
        or the box top inside a box footprint), at max_rise_xy (midpoint of that pair).
    max_slope_deg / mean_slope_deg (length-weighted) / max_pitch_deg (along the heading) /
        max_roll_deg (across it): gradient of the heightfield smoothed at SLOPE_SCALE_M, on
        continuous samples only (samples within the smoothing reach of a discontinuity, see
        DISCONTINUITY_SLOPE_DEG, are steps, measured by max_rise). max_slope_deg decides the
        slope violation. max_slope_raw_deg: the same maximum on the raw (unsmoothed) gradient,
        diagnostic only.
    rough_exposure_m2: sum over rough patches of (path length inside) * rms; length_in_rough_m.
    length_on_features_m: path length inside each feature kind's footprint.
    violations: subset of ("collision", "rise", "slope"); violation = any.
    """

    length_m: float
    min_clearance_m: float
    collision: bool
    collision_xy: tuple[float, float] | None
    max_rise_m: float
    max_rise_xy: tuple[float, float] | None
    max_slope_deg: float
    max_slope_raw_deg: float
    mean_slope_deg: float
    max_pitch_deg: float
    max_roll_deg: float
    rough_exposure_m2: float
    length_in_rough_m: float
    length_on_features_m: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()

    @property
    def violation(self) -> bool:
        return bool(self.violations)


def sample_polyline(
    path_xy: np.ndarray, step_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (M, 2), arc length s (M,), unit heading (M, 2)) every <= step_m along the path."""
    pts, s, head = [], [], []
    s0 = 0.0
    for a, b in pairwise(path_xy):
        seg = float(np.hypot(*(b - a)))
        if seg == 0.0:
            continue
        n = max(1, math.ceil(seg / step_m))
        t = np.arange(n) / n
        pts.append(a + t[:, None] * (b - a))
        s.append(s0 + t * seg)
        head.append(np.broadcast_to((b - a) / seg, (n, 2)))
        s0 += seg
    if not pts:
        p = np.asarray(path_xy[:1], dtype=float)
        return p, np.zeros(1), np.array([[1.0, 0.0]])
    pts.append(np.asarray(path_xy[-1:], dtype=float))
    s.append(np.array([s0]))
    head.append(head[-1][-1:])
    return np.vstack(pts), np.concatenate(s), np.vstack(head)


def _clip_interval(p0: float, d: float, lo: float, hi: float, t0: float, t1: float):
    """Liang-Barsky clip of p0 + t d (t in [t0, t1]) to [lo, hi]; returns the new (t0, t1)."""
    if d == 0.0:
        return (t0, t1) if lo <= p0 <= hi else (1.0, 0.0)
    ta, tb = (lo - p0) / d, (hi - p0) / d
    return max(t0, min(ta, tb)), min(t1, max(ta, tb))


def length_in_rect(path_xy: np.ndarray, center_xy, size_xy, yaw_rad: float) -> float:
    """Exact path length inside a rotated rectangle (size_xy = (length, width) along yaw)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    hl, hw = size_xy[0] / 2, size_xy[1] / 2
    total = 0.0
    for a, b in pairwise(path_xy):
        rel_a, d = a - center_xy, b - a
        la = (rel_a[0] * c + rel_a[1] * s, -rel_a[0] * s + rel_a[1] * c)
        ld = (d[0] * c + d[1] * s, -d[0] * s + d[1] * c)
        t0, t1 = _clip_interval(la[0], ld[0], -hl, hl, 0.0, 1.0)
        t0, t1 = _clip_interval(la[1], ld[1], -hw, hw, t0, t1)
        total += max(0.0, t1 - t0) * float(np.hypot(*d))
    return total


def length_in_disk(path_xy: np.ndarray, center_xy, radius: float) -> float:
    """Exact path length inside a disk."""
    total = 0.0
    for a, b in pairwise(path_xy):
        d, f = b - a, a - np.asarray(center_xy)
        qa, qb, qc = d @ d, 2 * (f @ d), f @ f - radius**2
        disc = qb * qb - 4 * qa * qc
        if qa == 0.0 or disc <= 0.0:
            continue
        r = math.sqrt(disc)
        t0, t1 = max(0.0, (-qb - r) / (2 * qa)), min(1.0, (-qb + r) / (2 * qa))
        total += max(0.0, t1 - t0) * math.sqrt(qa)
    return total


def surface_z(scenario: Scenario, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """GT surface height: bilinear terrain, or the box top inside a box footprint (closed)."""
    world = scenario.world
    z = bilinear_sample(world.heightfield, world.grid, x, y)
    for b in world.boxes:
        inside = (
            (x >= b.min_xyz[0]) & (x <= b.max_xyz[0]) & (y >= b.min_xyz[1]) & (y <= b.max_xyz[1])
        )
        z = np.where(inside, np.maximum(z, b.max_xyz[2]), z)
    return z


def discontinuity_mask(
    hf: np.ndarray,
    resolution_m: float,
    slope_deg: float = DISCONTINUITY_SLOPE_DEG,
    reach_cells: int = 1,
) -> np.ndarray:
    """Terrain cells within reach_cells of a step: |dh| between 4-neighbours > res * tan(slope_deg).
    reach_cells = 1 covers np.gradient's +-1 cell stencil; smoothing widens the reach."""
    thresh = resolution_m * math.tan(math.radians(slope_deg))
    jump = np.zeros(hf.shape, dtype=bool)
    for axis in (0, 1):
        big = np.abs(np.diff(hf, axis=axis)) > thresh
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        jump[tuple(lo)] |= big
        jump[tuple(hi)] |= big
    size = 2 * reach_cells + 1
    return ndimage.binary_dilation(jump, structure=np.ones((size, size), dtype=bool))


def _slope_field(hf: np.ndarray, grid, x, y, smooth_cells: float) -> np.ndarray:
    """(M, 2) world gradient (dz/dx, dz/dy) at the samples, of hf smoothed by smooth_cells."""
    # truncate at 3 sigma: matches the discontinuity reach used for the smoothed field
    if smooth_cells > 0:
        field_ = ndimage.gaussian_filter(hf, smooth_cells, mode="nearest", truncate=3.0)
    else:
        field_ = hf
    gy, gx = np.gradient(field_, grid.resolution)  # d/drow = d/dy, d/dcol = d/dx
    return np.stack([bilinear_sample(gx, grid, x, y), bilinear_sample(gy, grid, x, y)], axis=1)


def evaluate_path(scenario: Scenario, path_xy: np.ndarray, limits: PhysicalLimits) -> PathMetrics:
    """Measure one path against the raw world (see PathMetrics)."""
    path_xy = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    world = scenario.world
    grid, hf = world.grid, world.heightfield
    pts, s, head = sample_polyline(path_xy, SAMPLE_STEP_M)
    x, y = pts[:, 0], pts[:, 1]
    length = float(s[-1])

    clearance, collision_xy = math.inf, None
    for b in world.boxes:
        dx = np.maximum.reduce([b.min_xyz[0] - x, np.zeros_like(x), x - b.max_xyz[0]])
        dy = np.maximum.reduce([b.min_xyz[1] - y, np.zeros_like(y), y - b.max_xyz[1]])
        d = np.hypot(dx, dy)
        i = int(np.argmin(d))
        if d[i] < clearance:
            clearance, collision_xy = float(d[i]), (float(x[i]), float(y[i]))
    collision = clearance < limits.robot_radius_m

    z = surface_z(scenario, x, y)
    rise, rise_xy = 0.0, None
    for k in range(1, int(RISE_WINDOW_M / SAMPLE_STEP_M + 1e-9) + 1):
        if k >= len(z):
            break
        dz = np.abs(z[k:] - z[:-k])
        dz = np.where(s[k:] - s[:-k] <= RISE_WINDOW_M + 1e-9, dz, 0.0)
        i = int(np.argmax(dz))
        if dz[i] > rise:
            rise = float(dz[i])
            rise_xy = (float((x[i] + x[i + k]) / 2), float((y[i] + y[i + k]) / 2))

    sigma_cells = SLOPE_SCALE_M / grid.resolution
    # smoothed field: steps spread over ~3 sigma, so they are excluded that far out
    reach = 1 + math.ceil(3 * sigma_cells)
    g = _slope_field(hf, grid, x, y, sigma_cells)
    cont = (
        bilinear_sample(
            discontinuity_mask(hf, grid.resolution, reach_cells=reach).astype(float), grid, x, y
        )
        == 0
    )
    g_raw = _slope_field(hf, grid, x, y, 0.0)
    cont_raw = (
        bilinear_sample(discontinuity_mask(hf, grid.resolution).astype(float), grid, x, y) == 0
    )
    slope = np.degrees(np.arctan(np.hypot(g[:, 0], g[:, 1])))
    slope_raw = np.degrees(np.arctan(np.hypot(g_raw[:, 0], g_raw[:, 1])))
    pitch = np.degrees(np.arctan(np.abs(np.sum(g * head, axis=1))))
    normal = np.stack([-head[:, 1], head[:, 0]], axis=1)
    roll = np.degrees(np.arctan(np.abs(np.sum(g * normal, axis=1))))
    w = np.append(np.diff(s), 0.0) * cont  # length weight of each continuous sample
    mean_slope = float(np.sum(slope * w) / np.sum(w)) if np.sum(w) > 0 else 0.0

    def cmax(a: np.ndarray) -> float:
        return float(np.max(a[cont])) if cont.any() else 0.0

    on_feat: dict[str, float] = dict.fromkeys(FEATURE_KINDS, 0.0)
    exposure = 0.0
    for f in scenario.params.get("features", ()):
        kind = f["kind"]
        if kind == "rough":
            inside = length_in_disk(path_xy, f["center_xy"], f["radius"])
            exposure += inside * f["rms_m"]
        elif kind in ("ramp", "terrace"):
            inside = length_in_rect(path_xy, f["center_xy"], f["size_xy"], f["yaw_rad"])
        else:
            continue
        on_feat[kind] += inside

    max_slope = cmax(slope)
    violations = tuple(
        name
        for name, bad in (
            ("collision", collision),
            ("rise", rise > limits.max_step_m),
            ("slope", max_slope > limits.max_slope_deg),
        )
        if bad
    )
    return PathMetrics(
        length_m=length,
        min_clearance_m=clearance,
        collision=collision,
        collision_xy=collision_xy if collision else None,
        max_rise_m=rise,
        max_rise_xy=rise_xy,
        max_slope_deg=max_slope,
        max_slope_raw_deg=float(np.max(slope_raw[cont_raw])) if cont_raw.any() else 0.0,
        mean_slope_deg=mean_slope,
        max_pitch_deg=cmax(pitch),
        max_roll_deg=cmax(roll),
        rough_exposure_m2=exposure,
        length_in_rough_m=on_feat["rough"],
        length_on_features_m=on_feat,
        violations=violations,
    )
