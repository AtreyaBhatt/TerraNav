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
from terranav.perception.features import terrain_features
from terranav.perception.traversability import (
    Traversability,
    TraversabilityConfig,
    traversability,
)
from terranav.sim.coverage import corridor_mask
from terranav.sim.ground_truth import ground_truth_map
from terranav.sim.map_eval import FOUR_CONNECTED, build_map, survey_frames
from terranav.sim.scan_eval import Job
from terranav.types import GridSpec, SensorConfig


def traversable_connected(trav: Traversability, grid: GridSpec, start_xy, goal_xy) -> bool:
    """True iff start and goal share a 4-connected component of non-lethal cells."""
    labels, _ = ndimage.label(~trav.lethal, structure=FOUR_CONNECTED)
    rs, cs = grid.xy_to_rc(*start_xy)
    rg, cg = grid.xy_to_rc(*goal_xy)
    a, b = labels[int(rs), int(cs)], labels[int(rg), int(cg)]
    return bool(a != 0 and a == b)


def evaluate_trav_job(
    job: Job, sensor: SensorConfig | None = None, tcfg: TraversabilityConfig | None = None
) -> dict:
    """Counts and absolute-error sums over the corridor cells of one scenario.

    Rates and MAEs are formed by the caller from pooled sums, so families are weighted by
    cells, not by scenarios.
    """
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    tcfg = tcfg if tcfg is not None else TraversabilityConfig()
    grid = cfg.map_grid
    sc, frames = survey_frames(job, None, sensor, cfg)
    emap = build_map(sc, frames, sensor, cfg)
    feats = terrain_features(emap)
    trav = traversability(emap, feats, tcfg)
    gt_map = ground_truth_map(sc, grid)
    gt_feats = terrain_features(gt_map)
    gt = traversability(gt_map, gt_feats, tcfg)

    corr = corridor_mask(sc, grid)
    gt_lethal, gt_free = corr & gt.lethal, corr & ~gt.lethal
    both_free = gt_free & ~trav.lethal
    out = {
        "family": job[0],
        "seed": job[1],
        "gt_lethal": int(gt_lethal.sum()),
        "false_free": int((gt_lethal & ~trav.lethal).sum()),
        "gt_free": int(gt_free.sum()),
        "false_lethal": int((gt_free & trav.lethal).sum()),
        "cost_abs_sum": float(np.abs(trav.cost - gt.cost)[both_free].sum()),
        "cost_n": int(both_free.sum()),
        "connected_perceived": traversable_connected(trav, grid, sc.start_xy, sc.goal_xy),
        "connected_gt": traversable_connected(gt, grid, sc.start_xy, sc.goal_xy),
    }
    for name, scale in (("slope_rad", np.degrees(1.0)), ("roughness_m", 1.0), ("step_m", 1.0)):
        p, g = getattr(feats, name), getattr(gt_feats, name)
        both = corr & np.isfinite(p) & np.isfinite(g)
        out[f"{name}_abs_sum"] = float(np.abs(p - g)[both].sum() * scale)
        out[f"{name}_n"] = int(both.sum())
    return out


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
