"""Elevation-map evaluation: capture a survey, build the map, compare it with ground truth.

This module plays the robot's "pipeline" role: it supplies the footprint heights to
perception as proprioception (the robot knows the ground it stands on), using ground truth.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from terranav.config import SimConfig
from terranav.perception.elevation_map import FILLED, OBSERVED, UNKNOWN, ElevationMap, fuse
from terranav.perception.hole_fill import FOOTPRINT_RADIUS_M, fill_gaps, mark_footprint
from terranav.sim.coverage import corridor_mask, reachable_mask
from terranav.sim.ground_truth import ground_truth_elevation
from terranav.sim.scan_eval import Job
from terranav.sim.scenario import CLEAR_RADIUS_M, Scenario, generate_scenario, scan_poses
from terranav.sim.sensor import capture
from terranav.types import DepthFrame, SensorConfig

GOAL_VISIBLE_FRACTION = 0.8  # goal counts as visible if this share of its disk is known
FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def survey_frames(job: Job, k: int | None, sensor: SensorConfig, cfg: SimConfig):
    """(scenario, depth frames) for a job; frame i uses default_rng((seed, i))."""
    family, seed, strategy = job
    sc = generate_scenario(family, seed, cfg)
    poses = scan_poses(sc, k=k, strategy=strategy, sensor=sensor)
    frames = [
        capture(sc.world, T, sensor, np.random.default_rng((seed, i)))[0]
        for i, T in enumerate(poses)
    ]
    return sc, frames


def build_map(
    sc: Scenario, frames: Sequence[DepthFrame], sensor: SensorConfig, cfg: SimConfig
) -> ElevationMap:
    """Perception pipeline: fuse -> fill_gaps -> mark_footprint (proprioceptive heights)."""
    heights = ground_truth_elevation(sc, cfg.map_grid)  # stands in for proprioception
    emap = fill_gaps(fuse(frames, sensor, cfg.map_grid))
    return mark_footprint(emap, sc.start_xy, FOOTPRINT_RADIUS_M, heights)


def known_connected(emap: ElevationMap, start_xy, goal_xy) -> bool:
    """True iff start and goal cells lie in one 4-connected component of known cells
    (OBSERVED | FILLED | FOOTPRINT, i.e. source != UNKNOWN)."""
    labels, _ = ndimage.label(emap.source != UNKNOWN, structure=FOUR_CONNECTED)
    rs, cs = emap.grid.xy_to_rc(*start_xy)
    rg, cg = emap.grid.xy_to_rc(*goal_xy)
    a, b = labels[int(rs), int(cs)], labels[int(rg), int(cg)]
    return bool(a != 0 and a == b)


def evaluate_map_job(job: Job, k: int | None = None, sensor: SensorConfig | None = None) -> dict:
    """Build the elevation map for one (family, seed, strategy) job and measure it.

    Returns coverage (raw = OBSERVED, filled = not UNKNOWN) of the corridor and reachable
    regions, |z_mean - GT| for observed and filled cells (float32 arrays), the known share of
    the goal disk, and whether start and goal are connected through known cells.
    """
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    grid = cfg.map_grid
    sc, frames = survey_frames(job, k, sensor, cfg)
    emap = build_map(sc, frames, sensor, cfg)

    err = np.abs(emap.z_mean - ground_truth_elevation(sc, grid))
    observed, known = emap.source == OBSERVED, emap.source != UNKNOWN
    x, y = grid.cell_centers_xy()
    gx, gy = sc.goal_xy
    goal_disk = (x - gx) ** 2 + (y - gy) ** 2 <= CLEAR_RADIUS_M**2
    out = {"family": job[0], "seed": job[1]}
    for name, region in (
        ("corridor", corridor_mask(sc, grid)),
        ("reachable", reachable_mask(sc, grid)),
    ):
        out[f"{name}_raw"] = float(observed[region].mean())
        out[f"{name}_filled"] = float(known[region].mean())
    out["err_observed"] = err[observed].astype(np.float32)
    out["err_filled"] = err[emap.source == FILLED].astype(np.float32)
    out["goal_known_fraction"] = float(known[goal_disk].mean())
    out["connected"] = known_connected(emap, sc.start_xy, sc.goal_xy)
    return out


def perception_latency_s(job: Job, repeats: int = 3, k: int | None = None) -> float:
    """Median wall time (s) over `repeats` of fuse + fill_gaps + mark_footprint for one job.

    Simulation (scan poses, capture) and ground-truth lookup are done once, outside the
    timed region. Run in a single process for meaningful numbers.
    """
    cfg = SimConfig()
    sc, frames = survey_frames(job, k, cfg.sensor, cfg)
    heights = ground_truth_elevation(sc, cfg.map_grid)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        emap = fill_gaps(fuse(frames, cfg.sensor, cfg.map_grid))
        mark_footprint(emap, sc.start_xy, FOOTPRINT_RADIUS_M, heights)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
