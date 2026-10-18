"""Diagnosis of traversability errors (evaluation only).

core false-free cell: GT lethal before dilation, perceived non-lethal (the robot would believe
an actual hazard cell is passable). For each, in the corridor, record what perception saw and
the GT feature values (margins above the critical values). Route blockers: when perceived
start-goal is disconnected but GT is connected, find which perceived core-lethal category, if
removed, reconnects the route. Several TraversabilityConfig variants are evaluated on the same
perception run.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from terranav.config import SimConfig
from terranav.perception.traversability import (
    R_NO_ESTIMATE,
    R_ROUGH,
    R_SLOPE,
    R_STEP,
    R_UNKNOWN,
    TraversabilityConfig,
    dilate_lethal,
)
from terranav.sim.coverage import corridor_mask
from terranav.sim.scan_eval import Job
from terranav.sim.trav_eval import compare, ground_truth, perceive, traversable_connected
from terranav.types import SensorConfig

Variants = Sequence[tuple[str, TraversabilityConfig]]
BLOCKER_BITS = {
    "unknown": R_UNKNOWN,
    "no_estimate": R_NO_ESTIMATE,
    "slope": R_SLOPE,
    "rough": R_ROUGH,
    "step": R_STEP,
}


def route_blockers(reason: np.ndarray, grid, start_xy, goal_xy, robot_radius_m: float) -> list:
    """Smallest sets (size 1, else 2) of core categories whose removal reconnects the route."""
    present = [n for n, bit in BLOCKER_BITS.items() if (reason & bit).any()]
    for size in (1, 2):
        found = []
        for names in combinations(present, size):
            mask = np.uint8(sum(BLOCKER_BITS[n] for n in names))
            core = (reason & ~mask) != 0
            lethal = dilate_lethal(core, grid, robot_radius_m)
            if traversable_connected(lethal, grid, start_xy, goal_xy):
                found.append("+".join(names))
        if found:
            return found
    return ["none of <= 2 categories"]


def diagnose_job(job: Job, variants: Variants, sensor: SensorConfig | None = None) -> dict:
    """Per variant name: corridor counts (trav_eval.compare), core false-free cell records
    (perceived source, GT reasons, perceived and GT feature values, distance to the nearest
    camera), connectivity and route blockers."""
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    grid = cfg.map_grid
    tcfgs = tuple(t for _, t in variants)
    sc, frames, emap, feats, travs = perceive(job, sensor, tcfgs)
    corr = corridor_mask(sc, grid)
    x, y = grid.cell_centers_xy()
    cams = np.array([f.T_world_cam[:2, 3] for f in frames])
    dist_cam = np.min(np.hypot(x[..., None] - cams[:, 0], y[..., None] - cams[:, 1]), axis=-1)

    out = {"family": job[0], "seed": job[1]}
    for (name, tcfg), trav in zip(variants, travs, strict=True):
        gt_feats, gt = ground_truth(sc, grid, tcfg)
        ff = corr & (gt.core_reason != 0) & ~trav.lethal
        conn = traversable_connected(trav.lethal, grid, sc.start_xy, sc.goal_xy)
        conn_gt = traversable_connected(gt.lethal, grid, sc.start_xy, sc.goal_xy)
        blockers = []
        if conn_gt and not conn:
            blockers = route_blockers(
                trav.core_reason, grid, sc.start_xy, sc.goal_xy, tcfg.robot_radius_m
            )
        records = {"source": emap.source[ff], "gt_reason": gt.core_reason[ff]}
        for feat in ("slope_rad", "roughness_m", "step_m", "step_raw_m"):
            records[feat] = getattr(feats, feat)[ff]
        for feat in ("slope_rad", "roughness_m", "step_m"):
            records[f"gt_{feat}"] = getattr(gt_feats, feat)[ff]
        records["dist_cam_m"] = dist_cam[ff]
        out[name] = {
            "counts": compare(sc, grid, feats, trav, gt_feats, gt),
            "records": records,
            "connected": conn,
            "connected_gt": conn_gt,
            "blockers": blockers,
        }
    return out
