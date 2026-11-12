"""Showcase scenario for the presentation visuals: selection rule and one replayable run.

Selection is a fixed rule over DEV seeds (never held-out ones), stated before looking at the
outcome: rank the dev seeds of one family by the paired TerraNav - Baseline-Occ difference of
one metric at the primary alpha and take the median. With an even count the two middle
scenarios tie for "median"; the rule takes the UPPER one (larger difference = the one less
favourable to TerraNav), ties broken by seed.

run_showcase() replays exactly the benchmark job's calls (terranav.sim.benchmark.benchmark_job)
for one scenario and keeps the intermediate products (poses, per-frame points, perceived and GT
traversability) that the 3D renders need.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from terranav.config import SimConfig
from terranav.perception.backproject import backproject
from terranav.perception.features import terrain_features
from terranav.perception.traversability import (
    Traversability,
    TraversabilityConfig,
    traversability,
)
from terranav.planning.planners import DEFAULT_ALPHA, PLANNERS, plan
from terranav.sim.benchmark import BenchmarkConfig, benchmark_job
from terranav.sim.evaluate import PathMetrics, PhysicalLimits, evaluate_path, length_in_disk
from terranav.sim.map_eval import build_map, survey_frames
from terranav.sim.scan_eval import run_parallel
from terranav.sim.scenario import SCAN_STRATEGIES
from terranav.sim.trav_eval import ground_truth
from terranav.types import Scenario

DEV_SEEDS = tuple(range(10))
SHOWCASE_FAMILY = "rough"
SHOWCASE_METRIC = "rough_exposure_m2"
SHOWCASE_RULE = (
    "median (upper of the two middle values for an even count; ties by seed) of the paired "
    "TerraNav - Baseline-Occ rough-exposure difference at alpha = 4 over dev seeds 0..9 of the "
    "rough family"
)


def pick_median(diffs: dict[int, float]) -> tuple[list[tuple[int, float]], int]:
    """(ranking ascending by (difference, seed), chosen seed = ranking[n // 2]).

    n // 2 is the middle for odd n and the upper of the two middle ranks for even n.
    """
    if not diffs:
        raise ValueError("no scenarios to rank")
    ranking = sorted(diffs.items(), key=lambda kv: (kv[1], kv[0]))
    return ranking, ranking[len(ranking) // 2][0]


def dev_records(
    family: str = SHOWCASE_FAMILY,
    seeds: tuple[int, ...] = DEV_SEEDS,
    alpha: float = DEFAULT_ALPHA,
    workers: int = 1,
) -> list[dict]:
    """Benchmark records (benchmark_job, primary alpha only) for dev seeds of one family."""
    if any(s >= 1000 for s in seeds):
        raise ValueError("held-out seeds (>= 1000) must not be used for the showcase")
    bcfg = BenchmarkConfig(families=(family,), seeds=seeds, alphas=(alpha,))
    jobs = [(family, s) for s in seeds]
    per_job = run_parallel(partial(benchmark_job, bcfg=bcfg), jobs, workers)
    return [r for recs in per_job for r in recs]


def paired_metric(records: list[dict], metric: str, alpha: float) -> dict[int, float]:
    """{seed: TerraNav - Baseline-Occ metric} over seeds where both planners succeed."""
    by = {(r["seed"], r["planner"]): r for r in records if r["alpha"] in (None, alpha)}
    out = {}
    for seed in sorted({r["seed"] for r in records}):
        tn, bl = by.get((seed, "terranav")), by.get((seed, "baseline_occ"))
        if tn and bl and tn["success"] and bl["success"]:
            out[seed] = tn["metrics"][metric] - bl["metrics"][metric]
    return out


@dataclass(frozen=True)
class ShowcaseRun:
    """One scenario's pipeline products. Paths are (N, 2) world xy [m] of the planned vertices."""

    scenario: Scenario
    poses: list[np.ndarray]  # (4,4) T_world_cam per frame
    points: list[np.ndarray]  # (N_i, 3) world points per frame
    perceived: Traversability
    ground_truth: Traversability
    paths_xy: dict[str, np.ndarray | None]  # by planner name
    metrics: dict[str, PathMetrics | None]


def run_showcase(family: str, seed: int, alpha: float = DEFAULT_ALPHA) -> ShowcaseRun:
    """Same calls, same order and same generators as benchmark_job, for one alpha."""
    cfg = SimConfig()
    sensor, grid = cfg.sensor, cfg.map_grid
    tcfg, limits = TraversabilityConfig(), PhysicalLimits()
    sc, frames = survey_frames((family, seed, SCAN_STRATEGIES[0]), None, sensor, cfg)
    emap = build_map(sc, frames, sensor, cfg)
    perceived = traversability(emap, terrain_features(emap), tcfg)
    _, gt = ground_truth(sc, grid, tcfg)
    trav = {"perceived": perceived, "ground_truth": gt}
    paths: dict[str, np.ndarray | None] = {}
    metrics: dict[str, PathMetrics | None] = {}
    for name, spec in PLANNERS.items():
        p = plan(trav[spec.source], sc.start_xy, sc.goal_xy, grid, alpha, spec.graded)
        if p is None:
            paths[name] = metrics[name] = None
            continue
        x, y = grid.rc_to_xy(p.path_rc[:, 0], p.path_rc[:, 1])
        paths[name] = np.column_stack([x, y])
        metrics[name] = evaluate_path(sc, paths[name], limits)
    return ShowcaseRun(
        scenario=sc,
        poses=[f.T_world_cam for f in frames],
        points=[backproject(f)[0] for f in frames],
        perceived=perceived,
        ground_truth=gt,
        paths_xy=paths,
        metrics=metrics,
    )


def path_prefix(path_xy: np.ndarray, s_m: float) -> np.ndarray:
    """The polyline's first s_m metres (vertices up to s_m plus the interpolated end point)."""
    path_xy = np.asarray(path_xy, dtype=np.float64)
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if s_m >= cum[-1]:
        return path_xy.copy()
    i = int(np.searchsorted(cum, s_m, side="right")) - 1  # s_m lies on segment i
    t = (s_m - cum[i]) / seg[i] if seg[i] > 0 else 0.0
    end = path_xy[i] + t * (path_xy[i + 1] - path_xy[i])
    return np.vstack([path_xy[: i + 1], end])


def rough_exposure_along(scenario: Scenario, path_xy: np.ndarray, s_m: float) -> float:
    """Rough exposure [m^2] of the path's first s_m metres, by the evaluator's definition
    (sum over rough patches of length inside the patch disk * rms); equals
    evaluate_path(...).rough_exposure_m2 at the full length."""
    prefix = path_prefix(path_xy, s_m)
    return float(
        sum(
            length_in_disk(prefix, f["center_xy"], f["radius"]) * f["rms_m"]
            for f in scenario.params.get("features", ())
            if f["kind"] == "rough"
        )
    )


def load_checked_run(showcase_path: str | Path) -> tuple[dict, ShowcaseRun]:
    """(showcase.json contents, recomputed run). Raises if the recomputed paths' evaluator
    metrics differ from the ones recorded by scripts/pick_showcase.py (benchmark_job)."""
    show = json.loads(Path(showcase_path).read_text())
    run = run_showcase(show["family"], show["seed"], show["alpha"])
    for name, rec in show["metrics"].items():
        m = run.metrics[name]
        same = (m is None) == (not rec["success"]) and (
            m is None
            or all(
                math.isclose(getattr(m, k), rec[k], rel_tol=0.0, abs_tol=1e-12)
                for k in ("length_m", "rough_exposure_m2", "max_rise_m")
            )
        )
        if not same:
            raise ValueError(f"recomputed {name} differs from {showcase_path}")
    return show, run


def showcase_caption(show: dict) -> str:
    """One-line caption: family, seed and the selection rule."""
    return (
        f"{show['family']} family, dev seed {show['seed']}; chosen by rule, not by eye: "
        "median TerraNav - Baseline-Occ rough-exposure difference over dev seeds 0-9"
    )


def rough_intervals(scenario: Scenario, path_xy: np.ndarray) -> list[tuple[float, float]]:
    """Arc-length intervals [s0, s1] (metres) of the path inside any rough patch, merged.

    Same geometry as the evaluator (terranav.sim.evaluate.length_in_disk): each rough patch of
    scenario.params is the disk (center_xy, radius); each segment is clipped to each disk.
    """
    path_xy = np.asarray(path_xy, dtype=np.float64)
    disks = [
        (np.asarray(f["center_xy"], dtype=np.float64), f["radius"])
        for f in scenario.params.get("features", ())
        if f["kind"] == "rough"
    ]
    raw: list[tuple[float, float]] = []
    s0 = 0.0
    for a, b in itertools.pairwise(path_xy):
        d = b - a
        seg = math.sqrt(d @ d)
        for c, radius in disks:
            f = a - c
            qa, qb, qc = d @ d, 2 * (f @ d), f @ f - radius**2
            disc = qb * qb - 4 * qa * qc
            if qa == 0.0 or disc <= 0.0:
                continue
            r = math.sqrt(disc)
            t0, t1 = max(0.0, (-qb - r) / (2 * qa)), min(1.0, (-qb + r) / (2 * qa))
            if t1 > t0:
                raw.append((s0 + t0 * seg, s0 + t1 * seg))
        s0 += seg
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(raw):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def path_slice(path_xy: np.ndarray, s0_m: float, s1_m: float) -> np.ndarray:
    """The polyline between arc lengths s0_m <= s1_m (interpolated end points included)."""
    head = path_prefix(path_xy, s1_m)
    reversed_tail = path_prefix(head[::-1], _length(head) - s0_m)
    return reversed_tail[::-1]


def _length(path_xy: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(np.asarray(path_xy, dtype=np.float64), axis=0).T)))
