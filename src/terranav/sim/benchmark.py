"""Held-out benchmark: configuration, config hash, per-scenario job and runner.

Protocol: docs/benchmark_protocol.md. One job per scenario: survey + perception once, ground-truth
traversability once, Baseline-Occ once, TerraNav and Oracle for every alpha; every path is
measured by the independent evaluator (terranav.sim.evaluate), plus the definition-dependent
GT cost. Timings are recorded per stage but, measured inside parallel workers, they are NOT
latency figures (see the serial `make latency` tables).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from terranav.config import SimConfig
from terranav.perception.features import terrain_features
from terranav.perception.traversability import TraversabilityConfig, traversability
from terranav.planning.planners import DEFAULT_ALPHA, plan
from terranav.sim import scenario as scenario_mod
from terranav.sim.benchmark_report import read_latency_tables, render_summary
from terranav.sim.evaluate import PhysicalLimits, evaluate_path
from terranav.sim.map_eval import build_map, survey_frames
from terranav.sim.pipeline import gt_cost_along
from terranav.sim.scan_eval import run_parallel
from terranav.sim.scenario import FAMILIES, SCAN_STRATEGIES
from terranav.sim.trav_eval import ground_truth
from terranav.types import GridSpec, SensorConfig

__all__ = [
    "BenchmarkConfig",
    "benchmark_job",
    "config_dict",
    "config_hash",
    "render_summary",
    "run_benchmark",
    "scan_parameters",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """What the held-out benchmark runs. seeds 1000..1019 are held out (never used in design)."""

    families: tuple[str, ...] = FAMILIES
    seeds: tuple[int, ...] = tuple(range(1000, 1020))
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    default_alpha: float = DEFAULT_ALPHA
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20261018
    strategy: str = SCAN_STRATEGIES[0]


def scan_parameters() -> dict:
    """The survey's tunable constants (terranav.sim.scenario), for the config hash."""
    names = (
        "DEFAULT_SCAN_K",
        "FLANK_LATERAL_OFFSET_M",
        "FLANK_TOE_IN_DEG",
        "FLANK_FRACTIONS",
        "FLANK_FRACTION_SHIFTS",
        "POSE_CLEARANCE_M",
        "POSE_SEARCH_STEP_M",
        "POSE_SEARCH_AWAY_M",
    )
    return {n: getattr(scenario_mod, n) for n in names}


def config_dict(
    bcfg: BenchmarkConfig | None = None,
    sensor: SensorConfig | None = None,
    tcfg: TraversabilityConfig | None = None,
    limits: PhysicalLimits | None = None,
    scan: dict | None = None,
    map_grid: GridSpec | None = None,
    terrain_grid: GridSpec | None = None,
) -> dict:
    """All configuration that defines a benchmark run, as plain JSON-able data."""
    sim = SimConfig()

    def plain(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, dict):
            return {str(k): plain(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [plain(v) for v in obj]
        return obj

    return plain(
        {
            "benchmark": bcfg or BenchmarkConfig(),
            "sensor": sensor or sim.sensor,
            "traversability": tcfg or TraversabilityConfig(),
            "limits": limits or PhysicalLimits(),
            "scan": scan if scan is not None else scan_parameters(),
            "map_grid": map_grid or sim.map_grid,
            "terrain_grid": terrain_grid or sim.terrain_grid,
        }
    )


def config_hash(**kwargs) -> str:
    """First 12 hex digits of sha256 over the canonical JSON of config_dict(**kwargs)."""
    canonical = json.dumps(config_dict(**kwargs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _finite(obj):
    """JSON-safe copy: non-finite floats -> None (e.g. clearance with no boxes = null)."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_finite(v) for v in obj]
    if isinstance(obj, float | np.floating):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer | np.bool_):
        return obj.item()
    return obj


def benchmark_job(
    job: tuple[str, int], bcfg: BenchmarkConfig, sensor: SensorConfig | None = None
) -> list[dict]:
    """All records (planner x alpha) for one (family, seed) scenario."""
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    tcfg, limits, grid = TraversabilityConfig(), PhysicalLimits(), cfg.map_grid
    family, seed = job[0], int(job[1])

    t0 = time.perf_counter()
    sc, frames = survey_frames((family, seed, bcfg.strategy), None, sensor, cfg)
    t1 = time.perf_counter()
    emap = build_map(sc, frames, sensor, cfg)
    t2 = time.perf_counter()
    perceived = traversability(emap, terrain_features(emap), tcfg)
    t3 = time.perf_counter()
    _, gt = ground_truth(sc, grid, tcfg)
    stage = {"simulation": t1 - t0, "perception": t2 - t1, "features_trav": t3 - t2}

    def record(planner: str, alpha: float | None, trav, graded: bool) -> dict:
        p = plan(trav, sc.start_xy, sc.goal_xy, grid, alpha or 0.0, graded)
        rec = {"family": family, "seed": seed, "planner": planner, "alpha": alpha}
        rec["timings_s"] = {**stage, "plan": p.runtime_s if p else None}
        if p is None:
            return rec | {"success": False, "path_rc": None, "metrics": None, "gt_cost_m": None}
        x, y = grid.rc_to_xy(p.path_rc[:, 0], p.path_rc[:, 1])
        metrics = dataclasses.asdict(evaluate_path(sc, np.column_stack([x, y]), limits))
        gt_cost, gt_lethal = gt_cost_along(p.path_rc, gt, grid)
        return rec | {
            "success": True,
            "path_rc": p.path_rc.tolist(),
            "metrics": metrics,
            "gt_cost_m": gt_cost,  # DEFINITION-DEPENDENT (same definitions as the planners)
            "gt_lethal_length_m": gt_lethal,
            "expansions": p.expansions,
        }

    baseline = record("baseline_occ", None, perceived, graded=False)
    records = [baseline]
    for a in bcfg.alphas:
        tn = record("terranav", a, perceived, graded=True)
        if a == 0.0:
            tn["alpha0_matches_baseline"] = tn["path_rc"] == baseline["path_rc"]
        records += [tn, record("oracle", a, gt, graded=True)]
    return [_finite(r) for r in records]


def run_benchmark(
    bcfg: BenchmarkConfig,
    out_dir: str | Path,
    workers: int,
    sensor: SensorConfig | None = None,
    git: str = "unknown",
    started: str = "",
    latency_dir: str | Path = "out",
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Run every (family, seed) job and write raw.jsonl, meta.json, summary.md and figures."""
    import platform
    import sys

    from terranav.viz.benchmark_plots import plot_headline, plot_tradeoffs

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(f, s) for f in bcfg.families for s in bcfg.seeds]
    per_job = run_parallel(
        partial(benchmark_job, bcfg=bcfg, sensor=sensor), jobs, workers, progress
    )
    records = [r for recs in per_job for r in recs]
    with (out / "raw.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    kwargs = {"bcfg": bcfg} | ({"sensor": sensor} if sensor is not None else {})
    meta = {
        "git": git,
        "config_hash": config_hash(**kwargs),
        "configs": config_dict(**kwargs),
        "started": started,
        "wall_s": time.perf_counter() - t0,
        "machine": {"platform": platform.platform(), "machine": platform.machine()},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "workers": workers,
        "n_scenarios": len(jobs),
        "timings_note": "per-stage timings in raw.jsonl were measured inside parallel workers "
        "(contended CPU): they are not latency figures",
        "json_note": "null in metrics = infinite or not applicable (e.g. clearance with no boxes)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    latency = read_latency_tables(latency_dir, git)
    (out / "summary.md").write_text(render_summary(records, bcfg, meta, latency))
    plot_tradeoffs(records, bcfg, out)
    plot_headline(records, bcfg, out)
    return out
