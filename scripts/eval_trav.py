"""Evaluate perceived traversability against ground truth over many surveys.

Default mode (parallel, seeds 0..N-1, default "flank+start" survey), per family, pooled over
the corridor cells (within 3 m of the start->goal segment) of all its scenarios:
  false-free   = P(perceived non-lethal | GT lethal)      [safety]
  false-lethal = P(perceived lethal | GT non-lethal)      [conservatism]
  cost MAE on cells non-lethal in both; slope (deg), roughness (m), step (m) MAE on cells with
  finite features in both; traversable connectivity (start-goal through non-lethal cells),
  share of scenarios, perceived vs GT.
--latency mode (serial, seeds 0..4): terrain_features + traversability per scenario (median of
  3 repeats), median / p95 per family.
Writes out/eval_trav.txt / out/eval_trav_latency.txt headed by the git commit.
Numbers are measured, not targets.
Run: uv run python scripts/eval_trav.py [--seeds 10] [--workers N] [--latency]
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

from terranav.sim.scan_eval import git_revision, run_parallel
from terranav.sim.scenario import DEFAULT_SCAN_K, FAMILIES, SCAN_STRATEGIES
from terranav.sim.trav_eval import evaluate_trav_job, trav_latency_s

OUT = Path("out/eval_trav.txt")
OUT_LATENCY = Path("out/eval_trav_latency.txt")
STRATEGY = SCAN_STRATEGIES[0]
LATENCY_SEEDS = 5
LATENCY_REPEATS = 3


def header(title: str, rev: str, extra: list[str]) -> list[str]:
    return [
        f"# TerraNav {title} (measured)",
        f"# git: {rev}",
        f"# date: {dt.datetime.now().isoformat(timespec='seconds')}",
        (
            f"# survey {STRATEGY} (k = {DEFAULT_SCAN_K[STRATEGY]}), default SensorConfig, "
            "default TraversabilityConfig, 0.1 m map grid"
        ),
        *extra,
        "",
    ]


def ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def accuracy_table(results: list[dict], seeds: int, workers: int, wall_s: float, rev: str) -> str:
    lines = header(
        "traversability evaluation",
        rev,
        [
            f"# seeds 0..{seeds - 1}; {len(results)} jobs, {workers} worker(s), wall {wall_s:.1f} s",
            "# rates and MAEs pooled over corridor cells (<= 3 m from the start-goal segment)",
            "# false-free = P(perceived free | GT lethal); false-lethal = P(lethal | GT free)",
            "# conn: share of scenarios with start-goal connected through non-lethal cells",
        ],
    )
    lines.append(
        f"{'family':<15} {'false-free':>10} {'false-leth':>10} {'cost MAE':>9} "
        f"{'slope MAE':>10} {'rough MAE':>10} {'step MAE':>9} {'conn perc':>9} {'conn GT':>8}"
    )
    for family in FAMILIES:
        rs = [r for r in results if r["family"] == family]
        # Pool every count / error sum over the family's scenarios.
        counted = [k for k, v in rs[0].items() if k != "seed" and isinstance(v, int | float)]
        tot = {k: float(sum(r[k] for r in rs)) for k in counted if not isinstance(rs[0][k], bool)}

        lines.append(
            f"{family:<15} {ratio(tot['false_free'], tot['gt_lethal']):>10.3f} "
            f"{ratio(tot['false_lethal'], tot['gt_free']):>10.3f} "
            f"{ratio(tot['cost_abs_sum'], tot['cost_n']):>9.4f} "
            f"{ratio(tot['slope_rad_abs_sum'], tot['slope_rad_n']):>8.2f}deg "
            f"{ratio(tot['roughness_m_abs_sum'], tot['roughness_m_n']):>10.4f} "
            f"{ratio(tot['step_m_abs_sum'], tot['step_m_n']):>9.4f} "
            f"{np.mean([r['connected_perceived'] for r in rs]):>9.2f} "
            f"{np.mean([r['connected_gt'] for r in rs]):>8.2f}"
        )
    return "\n".join(lines) + "\n"


def run_latency(rev: str) -> str:
    lines = header(
        "traversability latency",
        rev,
        [
            f"# terrain_features + traversability; serial, seeds 0..{LATENCY_SEEDS - 1}",
            (
                f"# per scenario: median of {LATENCY_REPEATS} repeats; "
                f"machine: {platform.machine()}, {os.cpu_count()} logical CPUs"
            ),
        ],
    )
    lines.append(f"{'family':<15} {'median':>9} {'p95':>9}")
    for family in FAMILIES:
        t = [trav_latency_s((family, s, STRATEGY), LATENCY_REPEATS) for s in range(LATENCY_SEEDS)]
        lines.append(
            f"{family:<15} {np.percentile(t, 50) * 1e3:>7.1f}ms {np.percentile(t, 95) * 1e3:>7.1f}ms"
        )
        print(f"  {family} done", flush=True)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--latency", action="store_true", help="serial feature/trav timing only")
    args = parser.parse_args()

    rev, dirty = git_revision()
    print(f"git {rev}")
    if dirty:
        print("WARNING: working tree has uncommitted changes; results are not tied to a commit")
        rev += "-dirty"
    t0 = time.perf_counter()
    if args.latency:
        text, out = run_latency(rev), OUT_LATENCY
    else:
        jobs = [(f, s, STRATEGY) for f in FAMILIES for s in range(args.seeds)]
        print(f"{len(jobs)} jobs on {args.workers} worker(s)")

        def progress(done: int, total: int) -> None:
            print(f"\r  {done}/{total} jobs done", end="" if done < total else "\n", flush=True)

        results = run_parallel(evaluate_trav_job, jobs, args.workers, progress)
        wall_s = time.perf_counter() - t0
        text, out = accuracy_table(results, args.seeds, args.workers, wall_s, rev), OUT
    sys.stdout.write(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Saved {out}; wall time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
