"""Evaluate TerraNav, Baseline-Occ and Oracle paths against the raw simulated world.

Default mode (parallel, seeds 0..N-1, default "flank+start" survey, alpha = DEFAULT_ALPHA):
  A. per family x planner: success rate; violation rates (collision / rise / slope); median
     length ratio vs Oracle (scenarios where both succeed); mean rough exposure, length in rough,
     max rise, max slope; mean GT traversability cost along the path (DEFINITION-DEPENDENT).
  B. paired comparisons per family (scenarios where both succeed): TerraNav - Baseline-Occ and
     TerraNav - Oracle, mean difference with a 95% paired bootstrap CI (2000 resamples, fixed
     seed) for rough exposure, max rise and length, plus counts better / equal / worse.
  C. every failed plan and every violation.
--latency mode (serial, seeds 0..4): A* + smoothing per planner, median and p95.
Metrics in A-C come from terranav.sim.evaluate (independent of the cost model), except gt_cost.
Writes out/eval_plan.txt / out/eval_plan_latency.txt headed by the git commit, start time and
wall time. Refuses a dirty tree unless --allow-dirty. Numbers are measured, not targets.
Run: uv run python scripts/eval_plan.py [--seeds 10] [--workers N] [--latency] [--allow-dirty]
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

from terranav.planning.planners import DEFAULT_ALPHA, PLANNERS
from terranav.sim.plan_eval import (
    compare_counts,
    evaluate_plan_job,
    paired_bootstrap_ci,
    plan_latency_s,
)
from terranav.sim.scan_eval import revision_label, run_parallel
from terranav.sim.scenario import DEFAULT_SCAN_K, FAMILIES, SCAN_STRATEGIES

OUT = Path("out/eval_plan.txt")
OUT_LATENCY = Path("out/eval_plan_latency.txt")
STRATEGY = SCAN_STRATEGIES[0]
LATENCY_SEEDS = 5
LATENCY_REPEATS = 3
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261018
PAIRS = (("terranav", "baseline_occ"), ("terranav", "oracle"))
PAIRED_METRICS = (
    ("rough_exposure_m2", "rough exp"),
    ("max_rise_m", "max rise"),
    ("length_m", "length"),
)


def header(title: str, rev: str, started: str, wall_s: float, extra: list[str]) -> list[str]:
    return [
        f"# TerraNav {title} (measured)",
        f"# git: {rev}",
        f"# started: {started}",
        f"# wall time: {wall_s:.1f} s",
        f"# survey {STRATEGY} (k = {DEFAULT_SCAN_K[STRATEGY]}), alpha = {DEFAULT_ALPHA:g}, "
        + "default SensorConfig / TraversabilityConfig / PhysicalLimits, 0.1 m map grid",
        *extra,
        "",
    ]


def fmt(v: float, spec: str = ".3f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def planner_table(results: list[dict]) -> list[str]:
    lines = [
        "## A. per family x planner (means over successful plans unless stated)",
        f"{'family':<15} {'planner':<13} {'succ':>5} {'viol':>5} {'coll':>5} {'rise':>5} "
        + f"{'slope':>5} {'len/orc med':>11} {'rough exp':>9} {'in rough':>8} {'max rise':>8} "
        + f"{'max slope':>9} {'gt_cost*':>8}",
    ]
    for family in FAMILIES:
        rs = [r for r in results if r["family"] == family]
        for name in PLANNERS:
            ok = [r["planners"][name] for r in rs if r["planners"][name]["success"]]
            m = [p["metrics"] for p in ok]
            n = len(rs)

            def rate(cond, m=m, n=n) -> float:
                return sum(cond(x) for x in m) / n if n else float("nan")

            ratios = [
                r["planners"][name]["metrics"]["length_m"]
                / r["planners"]["oracle"]["metrics"]["length_m"]
                for r in rs
                if r["planners"][name]["success"] and r["planners"]["oracle"]["success"]
            ]
            lines.append(
                f"{family:<15} {name:<13} {len(ok) / n:>5.2f} "
                f"{rate(lambda x: bool(x['violations'])):>5.2f} "
                f"{rate(lambda x: 'collision' in x['violations']):>5.2f} "
                f"{rate(lambda x: 'rise' in x['violations']):>5.2f} "
                f"{rate(lambda x: 'slope' in x['violations']):>5.2f} "
                f"{fmt(float(np.median(ratios)) if ratios else float('nan')):>11} "
                f"{fmt(mean([x['rough_exposure_m2'] for x in m]), '.4f'):>9} "
                f"{fmt(mean([x['length_in_rough_m'] for x in m]), '.2f'):>8} "
                f"{fmt(mean([x['max_rise_m'] for x in m])):>8} "
                f"{fmt(mean([x['max_slope_deg'] for x in m]), '.1f'):>9} "
                f"{fmt(mean([p['gt_cost_m'] for p in ok])):>8}"
            )
    lines += [
        "rates are over all scenarios of the family (a failed plan counts as not violating);",
        "* gt_cost = integral of GT traversability cost along the path (GT-lethal cells count 1):",
        "  DEFINITION-DEPENDENT, uses the same traversability definitions as the planners.",
        "",
    ]
    return lines


def paired_table(results: list[dict]) -> list[str]:
    lines = [
        "## B. paired differences (A - B), scenarios where both succeed; lower is better",
        f"   95% CI: percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED};",
        "   better / equal / worse = scenarios where A is lower / within 1e-9 / higher.",
        f"{'family':<15} {'A - B':<24} {'n':>3} {'metric':<9} {'mean diff':>10} "
        + f"{'95% CI':>22} {'better/eq/worse':>16}",
    ]
    for family in FAMILIES:
        rs = [r for r in results if r["family"] == family]
        for a, b in PAIRS:
            both = [r for r in rs if r["planners"][a]["success"] and r["planners"][b]["success"]]
            for key, label in PAIRED_METRICS:
                va = np.array([r["planners"][a]["metrics"][key] for r in both])
                vb = np.array([r["planners"][b]["metrics"][key] for r in both])
                mu, lo, hi = paired_bootstrap_ci(va - vb, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
                bet, eq, wor = compare_counts(va, vb)
                lines.append(
                    f"{family:<15} {a + ' - ' + b:<24} {len(both):>3} {label:<9} {fmt(mu, '+.4f'):>10} "
                    f"{'[' + fmt(lo, '+.4f') + ', ' + fmt(hi, '+.4f') + ']':>22} "
                    f"{f'{bet}/{eq}/{wor}':>16}"
                )
    return [*lines, ""]


def failure_list(results: list[dict]) -> list[str]:
    lines = ["## C. failed plans and violations (family, seed, planner, reason)"]
    for r in sorted(results, key=lambda r: (FAMILIES.index(r["family"]), r["seed"])):
        for name in PLANNERS:
            p = r["planners"][name]
            if not p["success"]:
                lines.append(f"  {r['family']} seed {r['seed']} {name}: no path")
            elif p["metrics"]["violations"]:
                m = p["metrics"]
                detail = (
                    f"clearance {m['min_clearance_m']:.3f} m, max rise {m['max_rise_m']:.3f} m, "
                    f"max slope {m['max_slope_deg']:.1f} deg"
                )
                lines.append(
                    f"  {r['family']} seed {r['seed']} {name}: {', '.join(m['violations'])} ({detail})"
                )
    if len(lines) == 1:
        lines.append("  none")
    return [*lines, ""]


def run_latency(rev: str, started: str, t0: float) -> str:
    rows = [f"{'family':<15} {'planner':<13} {'median':>9} {'p95':>9}"]
    for family in FAMILIES:
        per = [plan_latency_s((family, s, STRATEGY), LATENCY_REPEATS) for s in range(LATENCY_SEEDS)]
        for name in PLANNERS:
            t = np.array([p[name] for p in per], dtype=float)
            t = t[np.isfinite(t)]
            med = np.percentile(t, 50) * 1e3 if t.size else float("nan")
            p95 = np.percentile(t, 95) * 1e3 if t.size else float("nan")
            rows.append(f"{family:<15} {name:<13} {fmt(med, '.1f'):>7}ms {fmt(p95, '.1f'):>7}ms")
        print(f"  {family} done", flush=True)
    extra = [
        f"# plan() = A* + smoothing per planner; serial, seeds 0..{LATENCY_SEEDS - 1}",
        f"# per scenario: median of {LATENCY_REPEATS} repeats; machine: {platform.machine()}, "
        + f"{os.cpu_count()} logical CPUs",
    ]
    head = header("planning latency", rev, started, time.perf_counter() - t0, extra)
    return "\n".join(head + rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--latency", action="store_true", help="serial planner timing only")
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

        results = run_parallel(evaluate_plan_job, jobs, args.workers, progress)
        body = planner_table(results) + paired_table(results) + failure_list(results)
        extra = [f"# seeds 0..{args.seeds - 1}; {len(jobs)} scenarios, {args.workers} worker(s)"]
        head = header("planning evaluation", rev, started, time.perf_counter() - t0, extra)
        text, out = "\n".join(head + body) + "\n", OUT
    sys.stdout.write(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Saved {out}; wall time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
