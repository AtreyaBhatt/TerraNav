"""Planning evaluation: run the three planners per scenario and measure their paths.

Path metrics come from the independent evaluator (terranav.sim.evaluate); the GT traversability
cost along the path is reported separately as definition-dependent (terranav.sim.pipeline).
"""

from __future__ import annotations

import statistics
from dataclasses import asdict

import numpy as np

from terranav.config import SimConfig
from terranav.planning.planners import DEFAULT_ALPHA, PLANNERS, plan
from terranav.sim.evaluate import PhysicalLimits, evaluate_path
from terranav.sim.pipeline import gt_cost_along, run_planners
from terranav.sim.scan_eval import Job
from terranav.types import SensorConfig

EQUAL_TOL = 1e-9  # paired values closer than this count as equal


def evaluate_plan_job(
    job: Job,
    alpha: float = DEFAULT_ALPHA,
    sensor: SensorConfig | None = None,
    limits: PhysicalLimits | None = None,
) -> dict:
    """Plan with every planner and measure each path against the raw world."""
    grid = SimConfig().map_grid
    limits = limits if limits is not None else PhysicalLimits()
    run = run_planners(job, alpha, sensor=sensor)
    out: dict = {"family": job[0], "seed": job[1], "planners": {}}
    for name, p in run.plans.items():
        if p is None:
            out["planners"][name] = {"success": False}
            continue
        x, y = grid.rc_to_xy(p.path_rc[:, 0], p.path_rc[:, 1])
        metrics = evaluate_path(run.scenario, np.column_stack([x, y]), limits)
        gt_cost, gt_lethal = gt_cost_along(p.path_rc, run.trav["ground_truth"], grid)
        out["planners"][name] = {
            "success": True,
            "metrics": asdict(metrics),
            "gt_cost_m": gt_cost,
            "gt_lethal_length_m": gt_lethal,
            "runtime_s": p.runtime_s,
            "expansions": p.expansions,
        }
    return out


def paired_bootstrap_ci(
    d: np.ndarray, n_resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float, float]:
    """(mean, lo, hi): mean of paired differences d with a percentile bootstrap CI."""
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(d.mean()), float(lo), float(hi)


def compare_counts(a: np.ndarray, b: np.ndarray, tol: float = EQUAL_TOL) -> tuple[int, int, int]:
    """(better, equal, worse) of a vs b, lower is better, |a - b| <= tol counts as equal."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    equal = np.abs(diff) <= tol
    return int(np.sum(~equal & (diff < 0))), int(np.sum(equal)), int(np.sum(~equal & (diff > 0)))


def plan_latency_s(job: Job, repeats: int = 3, alpha: float = DEFAULT_ALPHA) -> dict[str, float]:
    """Per planner, median wall time (s) over `repeats` of plan() (A* + smoothing) for one job.

    Perception and GT traversability are built once, outside the timed region.
    """
    grid = SimConfig().map_grid
    run = run_planners(job, alpha)
    sc = run.scenario
    out = {}
    for name, spec in PLANNERS.items():
        times = []
        for _ in range(repeats):
            p = plan(run.trav[spec.source], sc.start_xy, sc.goal_xy, grid, alpha, spec.graded)
            times.append(p.runtime_s if p is not None else float("nan"))
        out[name] = statistics.median(times)
    return out
