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

from terranav.sim.scan_eval import revision_label, run_parallel
from terranav.sim.scenario import DEFAULT_SCAN_K, FAMILIES, SCAN_STRATEGIES
from terranav.sim.trav_eval import evaluate_trav_job, trav_latency_s

OUT = Path("out/eval_trav.txt")
OUT_LATENCY = Path("out/eval_trav_latency.txt")
STRATEGY = SCAN_STRATEGIES[0]
LATENCY_SEEDS = 5
LATENCY_REPEATS = 3


def header(title: str, rev: str, started: str, wall_s: float, extra: list[str]) -> list[str]:
    return [
        f"# TerraNav {title} (measured)",
        f"# git: {rev}",
        f"# started: {started}",
        f"# wall time: {wall_s:.1f} s",
        (
            f"# survey {STRATEGY} (k = {DEFAULT_SCAN_K[STRATEGY]}), default SensorConfig, "
            "default TraversabilityConfig, 0.1 m map grid"
        ),
        *extra,
        "",
    ]


def rate(num: float, den: float, fmt: str = ".3f") -> str:
    """num / den formatted, or "n/a (0)" when the denominator is zero."""
    return format(num / den, fmt) if den else "n/a (0)"


def pooled(rs: list[dict]) -> dict[str, float]:
    """Sum every numeric (non-bool) field over a family's scenarios."""
    keys = [k for k, v in rs[0].items() if isinstance(v, int | float) and not isinstance(v, bool)]
    return {k: float(sum(r[k] for r in rs)) for k in keys if k != "seed"}


def accuracy_table(
    results: list[dict], seeds: int, workers: int, wall_s: float, rev: str, started: str
) -> str:
    lines = header(
        "traversability evaluation",
        rev,
        started,
        wall_s,
        [
            f"# seeds 0..{seeds - 1}; {len(results)} jobs, {workers} worker(s)",
            "# all counts / rates / MAEs pooled over corridor cells (<= 3 m from start-goal)",
            "# core FF  = P(perceived non-lethal | GT lethal before dilation)  [safety]",
            "# band FF  = P(perceived non-lethal | GT lethal after dilation)",
            "# FL       = P(perceived lethal | GT non-lethal), split by the perceived cell's own",
            "#            reason: unknown | no estimate (support) | feature >= crit | dilation only",
            "# conn: share of scenarios with start-goal connected through non-lethal cells",
        ],
    )
    lines.append(
        f"{'family':<15} {'GT core':>8} {'GT leth':>8} {'GT free':>8} {'core FF':>9} "
        f"{'band FF':>9} {'FL':>7} {'FL unkn':>8} {'FL supp':>8} {'FL feat':>8} {'FL dil':>8}"
    )
    for family in FAMILIES:
        t = pooled([r for r in results if r["family"] == family])
        free = t["gt_free"]
        lines.append(
            f"{family:<15} {t['gt_core']:>8.0f} {t['gt_lethal']:>8.0f} {free:>8.0f} "
            f"{rate(t['core_false_free'], t['gt_core']):>9} "
            f"{rate(t['band_false_free'], t['gt_lethal']):>9} {rate(t['false_lethal'], free):>7} "
            f"{rate(t['fl_unknown'], free):>8} {rate(t['fl_no_estimate'], free):>8} "
            f"{rate(t['fl_feature'], free):>8} {rate(t['fl_dilation'], free):>8}"
        )
    lines += [
        "",
        (
            f"{'family':<15} {'cost MAE':>9} {'slope MAE':>10} {'rough MAE':>10} "
            f"{'step MAE':>9} {'conn perc':>9} {'conn GT':>8}"
        ),
    ]
    for family in FAMILIES:
        rs = [r for r in results if r["family"] == family]
        t = pooled(rs)
        lines.append(
            f"{family:<15} {rate(t['cost_abs_sum'], t['cost_n'], '.4f'):>9} "
            f"{rate(t['slope_rad_abs_sum'], t['slope_rad_n'], '.2f'):>7}deg "
            f"{rate(t['roughness_m_abs_sum'], t['roughness_m_n'], '.4f'):>10} "
            f"{rate(t['step_m_abs_sum'], t['step_m_n'], '.4f'):>9} "
            f"{np.mean([r['connected_perceived'] for r in rs]):>9.2f} "
            f"{np.mean([r['connected_gt'] for r in rs]):>8.2f}"
        )
    return "\n".join(lines) + "\n"


def run_latency(rev: str, started: str, t0: float) -> str:
    title = "traversability latency"
    extra = [
        f"# terrain_features + traversability; serial, seeds 0..{LATENCY_SEEDS - 1}",
        (
            f"# per scenario: median of {LATENCY_REPEATS} repeats; "
            f"machine: {platform.machine()}, {os.cpu_count()} logical CPUs"
        ),
    ]
    lines = [f"{'family':<15} {'median':>9} {'p95':>9}"]
    for family in FAMILIES:
        t = [trav_latency_s((family, s, STRATEGY), LATENCY_REPEATS) for s in range(LATENCY_SEEDS)]
        lines.append(
            f"{family:<15} {np.percentile(t, 50) * 1e3:>7.1f}ms {np.percentile(t, 95) * 1e3:>7.1f}ms"
        )
        print(f"  {family} done", flush=True)
    head = header(title, rev, started, time.perf_counter() - t0, extra)
    return "\n".join(head + lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--latency", action="store_true", help="serial feature/trav timing only")
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

        results = run_parallel(evaluate_trav_job, jobs, args.workers, progress)
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
