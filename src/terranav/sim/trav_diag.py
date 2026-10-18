"""Diagnosis of traversability errors (evaluation only).

core false-free cell: GT lethal before dilation, perceived non-lethal (the robot would believe
an actual hazard cell is passable). For each, in the corridor, record what perception saw.
Route blockers: when perceived start-goal is disconnected but GT is connected, find which
perceived core-lethal category, if removed, reconnects the route.
"""

from __future__ import annotations

from dataclasses import replace
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
from terranav.sim.trav_eval import ground_truth, perceive, traversable_connected
from terranav.types import SensorConfig

GATE_SETTINGS = (True, False)  # step_gate_for_lethal
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


def diagnose_job(job: Job, sensor: SensorConfig | None = None) -> dict:
    """Per gate setting (key "gate_True" / "gate_False"): core false-free cell records,
    connectivity and route blockers."""
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    grid = cfg.map_grid
    base = TraversabilityConfig()
    tcfgs = tuple(replace(base, step_gate_for_lethal=g) for g in GATE_SETTINGS)
    sc, frames, emap, feats, travs = perceive(job, sensor, tcfgs)
    corr = corridor_mask(sc, grid)
    x, y = grid.cell_centers_xy()
    cams = np.array([f.T_world_cam[:2, 3] for f in frames])
    dist_cam = np.min(np.hypot(x[..., None] - cams[:, 0], y[..., None] - cams[:, 1]), axis=-1)

    out = {"family": job[0], "seed": job[1]}
    for gate, tcfg, trav in zip(GATE_SETTINGS, tcfgs, travs, strict=True):
        _, gt = ground_truth(sc, grid, tcfg)
        ff = corr & (gt.core_reason != 0) & ~trav.lethal
        conn = traversable_connected(trav.lethal, grid, sc.start_xy, sc.goal_xy)
        conn_gt = traversable_connected(gt.lethal, grid, sc.start_xy, sc.goal_xy)
        blockers = []
        if conn_gt and not conn:
            blockers = route_blockers(
                trav.core_reason, grid, sc.start_xy, sc.goal_xy, tcfg.robot_radius_m
            )
        out[f"gate_{gate}"] = {
            "source": emap.source[ff],
            "gt_reason": gt.core_reason[ff],
            "slope_rad": feats.slope_rad[ff],
            "roughness_m": feats.roughness_m[ff],
            "step_m": feats.step_m[ff],
            "step_raw_m": feats.step_raw_m[ff],
            "dist_cam_m": dist_cam[ff],
            "connected": conn,
            "connected_gt": conn_gt,
            "blockers": blockers,
        }
    return out
