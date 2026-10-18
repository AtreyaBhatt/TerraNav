"""Measure scan coverage for both scan strategies over many scenarios.

For seeds 0..N-1 x all families x {flank, line}: capture every scan pose (frame i uses
np.random.default_rng((seed, i))), backproject, and measure coverage on the 0.1 m map grid of
  corridor  : cells within 3 m of the start->goal segment,
  reachable : cells of the easy-ground component containing start.
Jobs run in worker processes (--workers 1 = serial; results are identical either way).
Prints progress, a table of mean / min coverage per family, region and strategy, "line"
skipped poses and the wall time, and writes the table to out/eval_scan.txt headed by the git
commit; refuses a dirty tree unless --allow-dirty (then the hash gets "-dirty"). Numbers are measured, not targets.
Run: uv run python scripts/eval_scan.py [--seeds 10] [--workers N]
"""

from __future__ import annotations

import os

# Before numpy is imported anywhere: one native thread per process (workers inherit this),
# so N worker processes do not oversubscribe the CPU.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np

from terranav.sim.scan_eval import revision_label, run_jobs
from terranav.sim.scenario import DEFAULT_SCAN_K, FAMILIES, SCAN_STRATEGIES

OUT = Path("out/eval_scan.txt")


def table(results: list[dict], seeds: int, workers: int, wall_s: float, rev: str) -> str:
    lines = [
        "# TerraNav scan coverage evaluation (measured)",
        f"# git: {rev}",
        f"# date: {dt.datetime.now().isoformat(timespec='seconds')}",
        (
            f"# seeds 0..{seeds - 1}, poses per strategy {DEFAULT_SCAN_K}, default SensorConfig, "
            "0.1 m map grid"
        ),
        f"# {len(results)} jobs, {workers} worker(s), wall time {wall_s:.1f} s",
        "# coverage = fraction of region cells with >= 1 backprojected point",
        "",
        f"{'family':<15} {'region':<10}"
        + "".join(f" {st + ' mean':>17} {st + ' min':>16}" for st in SCAN_STRATEGIES),
    ]
    for family in FAMILIES:
        for region in ("corridor", "reachable"):
            row = f"{family:<15} {region:<10}"
            for strategy in SCAN_STRATEGIES:
                vals = [
                    r[region]
                    for r in results
                    if r["family"] == family and r["strategy"] == strategy
                ]
                row += f" {np.mean(vals):>17.3f} {np.min(vals):>16.3f}"
            lines.append(row)
    k_line = DEFAULT_SCAN_K["line"]
    lines += ["", f"line strategy skipped poses (of {k_line} per scenario, {seeds} scenarios):"]
    for family in FAMILIES:
        skips = [r["skipped"] for r in results if r["family"] == family and r["strategy"] == "line"]
        lines.append(
            f"  {family:<15} total {sum(skips):>3} / {k_line * seeds}, "
            f"scenarios with >= 1 skip: {sum(s > 0 for s in skips)}, max per scenario: {max(skips)}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on an uncommitted tree (hash gets -dirty)"
    )
    args = parser.parse_args()
    rev = revision_label(args.allow_dirty)

    jobs = [(f, s, st) for f in FAMILIES for s in range(args.seeds) for st in SCAN_STRATEGIES]
    print(f"{len(jobs)} jobs on {args.workers} worker(s)")

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} jobs done", end="" if done < total else "\n", flush=True)

    t0 = time.perf_counter()
    results = run_jobs(jobs, args.workers, progress=progress)
    wall_s = time.perf_counter() - t0

    text = table(results, args.seeds, args.workers, wall_s, rev)
    print(text, end="")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"Saved {OUT}; wall time {wall_s:.1f} s on {args.workers} worker(s)")


if __name__ == "__main__":
    main()
