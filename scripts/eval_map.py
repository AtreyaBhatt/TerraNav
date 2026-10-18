"""Evaluate the 2.5D elevation map against ground truth over many surveys.

Default mode (parallel, seeds 0..N-1, default "flank+start" survey), per family:
  corridor / reachable coverage raw (observed) vs after fill (observed + filled + footprint);
  |z_mean - GT| for observed and filled cells (median, p95, pooled over all cells);
  goal visibility (share of scenarios with >= 80% of goal-disk cells known);
  connectivity (share of scenarios whose start and goal share a 4-connected known component).
--latency mode (serial, seeds 0..4): perception time fuse + fill + footprint per scenario
  (median of 3 repeats), reported as median / p95 per family.
Results are written to out/eval_map.txt / out/eval_map_latency.txt headed by the git commit.
Numbers are measured, not targets.
Run: uv run python scripts/eval_map.py [--seeds 10] [--workers N] [--latency]
"""

from __future__ import annotations

import os

# Before numpy is imported anywhere: one native thread per process (workers inherit this).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime as dt
import platform
import sys
import time
from pathlib import Path

import numpy as np

from terranav.sim.map_eval import GOAL_VISIBLE_FRACTION, evaluate_map_job, perception_latency_s
from terranav.sim.scan_eval import revision_label, run_parallel
from terranav.sim.scenario import DEFAULT_SCAN_K, FAMILIES, SCAN_STRATEGIES

OUT = Path("out/eval_map.txt")
OUT_LATENCY = Path("out/eval_map_latency.txt")
STRATEGY = SCAN_STRATEGIES[0]  # the default survey
LATENCY_SEEDS = 5
LATENCY_REPEATS = 3


def pct(a, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def header(title: str, rev: str, started: str, wall_s: float, extra: list[str]) -> list[str]:
    return [
        f"# TerraNav {title} (measured)",
        f"# git: {rev}",
        f"# started: {started}",
        f"# wall time: {wall_s:.1f} s",
        f"# survey {STRATEGY} (k = {DEFAULT_SCAN_K[STRATEGY]}), default SensorConfig, 0.1 m map grid",
        *extra,
        "",
    ]


def accuracy_table(
    results: list[dict], seeds: int, workers: int, wall_s: float, rev: str, started: str
) -> str:
    lines = header(
        "elevation map evaluation",
        rev,
        started,
        wall_s,
        [
            f"# seeds 0..{seeds - 1}; {len(results)} jobs, {workers} worker(s)",
            "# coverage: mean over scenarios; raw = observed, fill = observed + filled + footprint",
            "# errors |z_mean - GT| in metres, pooled over all cells of the family's scenarios",
            f"# goal vis: share with >= {GOAL_VISIBLE_FRACTION:.0%} of goal-disk cells known",
            "# connected: share whose start and goal share a 4-connected known component",
        ],
    )
    lines.append(
        f"{'family':<15} {'corr raw':>8} {'corr fill':>9} {'reach raw':>9} {'reach fill':>10} "
        f"{'obs med':>8} {'obs p95':>8} {'fill med':>8} {'fill p95':>8} {'goal vis':>8} "
        f"{'connected':>9}"
    )
    for family in FAMILIES:
        rs = [r for r in results if r["family"] == family]
        obs = np.concatenate([r["err_observed"] for r in rs])
        fil = np.concatenate([r["err_filled"] for r in rs])
        vis = np.mean([r["goal_known_fraction"] >= GOAL_VISIBLE_FRACTION for r in rs])
        lines.append(
            f"{family:<15} {np.mean([r['corridor_raw'] for r in rs]):>8.3f} "
            f"{np.mean([r['corridor_filled'] for r in rs]):>9.3f} "
            f"{np.mean([r['reachable_raw'] for r in rs]):>9.3f} "
            f"{np.mean([r['reachable_filled'] for r in rs]):>10.3f} "
            f"{pct(obs, 50):>8.4f} {pct(obs, 95):>8.4f} {pct(fil, 50):>8.4f} {pct(fil, 95):>8.4f} "
            f"{vis:>8.2f} {np.mean([r['connected'] for r in rs]):>9.2f}"
        )
    return "\n".join(lines) + "\n"


def run_latency(rev: str, started: str, t0: float) -> str:
    title = "perception latency"
    extra = [
        f"# fuse + fill_gaps + mark_footprint; serial (1 process), seeds 0..{LATENCY_SEEDS - 1}",
        (
            f"# per scenario: median of {LATENCY_REPEATS} repeats; "
            f"machine: {platform.machine()}, {os.cpu_count()} logical CPUs"
        ),
    ]
    lines = [f"{'family':<15} {'median':>9} {'p95':>9}"]
    for family in FAMILIES:
        t = [
            perception_latency_s((family, s, STRATEGY), LATENCY_REPEATS)
            for s in range(LATENCY_SEEDS)
        ]
        lines.append(f"{family:<15} {pct(t, 50) * 1e3:>7.1f}ms {pct(t, 95) * 1e3:>7.1f}ms")
        print(f"  {family} done", flush=True)
    head = header(title, rev, started, time.perf_counter() - t0, extra)
    return "\n".join(head + lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--latency", action="store_true", help="serial perception timing only")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on an uncommitted tree (hash gets -dirty)"
    )
    args = parser.parse_args()
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    rev = revision_label(args.allow_dirty)

    if args.latency:
        text, out = run_latency(rev, started, t0), OUT_LATENCY
    else:
        jobs = [(f, s, STRATEGY) for f in FAMILIES for s in range(args.seeds)]
        print(f"{len(jobs)} jobs on {args.workers} worker(s)")

        def progress(done: int, total: int) -> None:
            print(f"\r  {done}/{total} jobs done", end="" if done < total else "\n", flush=True)

        results = run_parallel(evaluate_map_job, jobs, args.workers, progress)
        wall_s = time.perf_counter() - t0
        text, out = (
            accuracy_table(results, args.seeds, args.workers, wall_s, rev, started),
            OUT,
        )
    sys.stdout.write(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Saved {out}; wall time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
