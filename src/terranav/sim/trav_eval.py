"""Traversability evaluation: perceived vs ground-truth traversability on the map grid.

Perceived: survey -> elevation map (fuse, fill, footprint) -> features -> traversability.
Ground truth: the same feature and traversability functions on the GT elevation map.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
from scipy import ndimage

from terranav.config import SimConfig
from terranav.perception.features import TerrainFeatures, terrain_features
from terranav.perception.traversability import (
    R_FEATURE,
    R_NO_ESTIMATE,
    R_UNKNOWN,
    Traversability,
    TraversabilityConfig,
    traversability,
)
from terranav.sim.coverage import corridor_mask
from terranav.sim.ground_truth import ground_truth_map
from terranav.sim.map_eval import FOUR_CONNECTED, build_map, survey_frames
from terranav.sim.scan_eval import Job
from terranav.types import GridSpec, SensorConfig

# False-lethal causes, from the PERCEIVED core_reason of the cell itself (first match wins).
FL_CAUSES = ("unknown", "no_estimate", "feature", "dilation")


def traversable_connected(lethal: np.ndarray, grid: GridSpec, start_xy, goal_xy) -> bool:
    """True iff start and goal share a 4-connected component of non-lethal cells."""
    labels, _ = ndimage.label(~lethal, structure=FOUR_CONNECTED)
    rs, cs = grid.xy_to_rc(*start_xy)
    rg, cg = grid.xy_to_rc(*goal_xy)
    a, b = labels[int(rs), int(cs)], labels[int(rg), int(cg)]
    return bool(a != 0 and a == b)


def false_lethal_cause(reason: np.ndarray) -> np.ndarray:
    """(ny, nx) int index into FL_CAUSES for a perceived-lethal cell, from its core_reason."""
    cause = np.full(reason.shape, FL_CAUSES.index("dilation"))
    cause[(reason & R_FEATURE) != 0] = FL_CAUSES.index("feature")
    cause[(reason & R_NO_ESTIMATE) != 0] = FL_CAUSES.index("no_estimate")
    cause[(reason & R_UNKNOWN) != 0] = FL_CAUSES.index("unknown")
    return cause


def perceive(job: Job, sensor: SensorConfig, tcfgs: tuple[TraversabilityConfig, ...]):
    """(scenario, frames, emap, features, [Traversability per config]) for one job."""
    cfg = SimConfig()
    sc, frames = survey_frames(job, None, sensor, cfg)
    emap = build_map(sc, frames, sensor, cfg)
    feats = terrain_features(emap)
    return sc, frames, emap, feats, [traversability(emap, feats, t) for t in tcfgs]


def ground_truth(sc, grid: GridSpec, tcfg: TraversabilityConfig):
    """(GT features, GT Traversability) from the GT elevation map."""
    gt_map = ground_truth_map(sc, grid)
    gt_feats = terrain_features(gt_map)
    return gt_feats, traversability(gt_map, gt_feats, tcfg)


def compare(sc, grid: GridSpec, feats: TerrainFeatures, trav: Traversability, gt_feats, gt) -> dict:
    """Counts and absolute-error sums over the corridor cells (pooled later per family)."""
    corr = corridor_mask(sc, grid)
    gt_core = corr & (gt.core_reason != 0)
    gt_lethal, gt_free = corr & gt.lethal, corr & ~gt.lethal
    fl = gt_free & trav.lethal
    cause = false_lethal_cause(trav.core_reason)
    both_free = gt_free & ~trav.lethal
    out = {
        "gt_core": int(gt_core.sum()),
        "core_false_free": int((gt_core & ~trav.lethal).sum()),
        "gt_lethal": int(gt_lethal.sum()),
        "band_false_free": int((gt_lethal & ~trav.lethal).sum()),
        "gt_free": int(gt_free.sum()),
        "false_lethal": int(fl.sum()),
        "cost_abs_sum": float(np.abs(trav.cost - gt.cost)[both_free].sum()),
        "cost_n": int(both_free.sum()),
        "connected_perceived": traversable_connected(trav.lethal, grid, sc.start_xy, sc.goal_xy),
        "connected_gt": traversable_connected(gt.lethal, grid, sc.start_xy, sc.goal_xy),
    }
    for i, name in enumerate(FL_CAUSES):
        out[f"fl_{name}"] = int((fl & (cause == i)).sum())
    for name, scale in (("slope_rad", np.degrees(1.0)), ("roughness_m", 1.0), ("step_m", 1.0)):
        p, g = getattr(feats, name), getattr(gt_feats, name)
        both = corr & np.isfinite(p) & np.isfinite(g)
        out[f"{name}_abs_sum"] = float(np.abs(p - g)[both].sum() * scale)
        out[f"{name}_n"] = int(both.sum())
    return out


def evaluate_trav_job(
    job: Job, sensor: SensorConfig | None = None, tcfg: TraversabilityConfig | None = None
) -> dict:
    """Corridor counts / error sums for one scenario (see compare)."""
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    tcfg = tcfg if tcfg is not None else TraversabilityConfig()
    sc, _, _, feats, (trav,) = perceive(job, sensor, (tcfg,))
    gt_feats, gt = ground_truth(sc, cfg.map_grid, tcfg)
    return {"family": job[0], "seed": job[1]} | compare(sc, cfg.map_grid, feats, trav, gt_feats, gt)


def trav_latency_s(job: Job, repeats: int = 3) -> float:
    """Median wall time (s) over `repeats` of terrain_features + traversability for one job.

    Survey simulation and map building happen once, outside the timed region. Run serially.
    """
    cfg = SimConfig()
    tcfg = TraversabilityConfig()
    sc, frames = survey_frames(job, None, cfg.sensor, cfg)
    emap = build_map(sc, frames, cfg.sensor, cfg)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        traversability(emap, terrain_features(emap), tcfg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
