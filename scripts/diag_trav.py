"""Diagnose core false-free traversability cells and perceived-only route closures.

Runs diagnose_job (src/terranav/sim/trav_diag.py) for families x seeds 0..N-1 in parallel,
for both settings of TraversabilityConfig.step_gate_for_lethal, and writes a per-family
breakdown to --out (default out/diag_trav.txt) headed by the git commit.
Run: uv run python scripts/diag_trav.py [--seeds 10] [--families ...] [--workers N] [--out ...]
"""

from __future__ import annotations

import os

# Before numpy is imported anywhere: one native thread per process (workers inherit this).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

from terranav.perception.elevation_map import FILLED, FOOTPRINT, OBSERVED
from terranav.perception.traversability import R_ROUGH, R_SLOPE, R_STEP, TraversabilityConfig
from terranav.sim.scan_eval import revision_label, run_parallel
from terranav.sim.scenario import FAMILIES, SCAN_STRATEGIES
from terranav.sim.trav_diag import GATE_SETTINGS, diagnose_job

CRIT = TraversabilityConfig()


def med(a) -> str:
    return f"{np.median(a):.3f}" if len(a) else "-"


def p95(a) -> str:
    return f"{np.percentile(a, 95):.2f}" if len(a) else "-"


def family_rows(results: list[dict], gate: bool) -> list[str]:
    lines = [
        f"## step_gate_for_lethal = {gate}",
        (
            f"{'family':<15} {'coreFF':>6} | {'OBS':>4} {'FILL':>4} {'FOOT':>4} | "
            f"{'GTslp':>5} {'GTrgh':>5} {'GTstp':>5} | {'stp: raw>=crit&gated':>20} "
            f"{'med step':>8} {'med raw':>8} | {'med slope':>9} {'med rough':>9} | "
            f"{'cam med':>7} {'cam p95':>7}"
        ),
    ]
    for family in sorted({r["family"] for r in results}, key=FAMILIES.index):
        d = [r[f"gate_{gate}"] for r in results if r["family"] == family]
        cat = {
            k: np.concatenate([x[k] for x in d]) for k in d[0] if isinstance(d[0][k], np.ndarray)
        }
        reason, src = cat["gt_reason"], cat["source"]
        is_step, is_slope, is_rough = (
            (reason & R_STEP) != 0,
            (reason & R_SLOPE) != 0,
            (reason & R_ROUGH) != 0,
        )
        gated = (
            is_step & (cat["step_raw_m"] >= CRIT.step_crit_m) & (cat["step_m"] < CRIT.step_crit_m)
        )
        lines.append(
            f"{family:<15} {len(src):>6} | {np.sum(src == OBSERVED):>4} {np.sum(src == FILLED):>4} "
            f"{np.sum(src == FOOTPRINT):>4} | {is_slope.sum():>5} {is_rough.sum():>5} "
            f"{is_step.sum():>5} | {f'{gated.sum()} of {is_step.sum()}':>20} "
            f"{med(cat['step_m'][is_step]):>8} {med(cat['step_raw_m'][is_step]):>8} | "
            f"{med(np.degrees(cat['slope_rad'][is_slope])):>9} "
            f"{med(cat['roughness_m'][is_rough]):>9} | "
            f"{med(cat['dist_cam_m']):>7} {p95(cat['dist_cam_m']):>7}"
        )
    lines += ["", "perceived disconnected but GT connected (route blockers = core categories whose"]
    lines += ["removal alone reconnects start and goal):"]
    closed = [
        r
        for r in results
        if r[f"gate_{gate}"]["connected_gt"] and not r[f"gate_{gate}"]["connected"]
    ]
    for r in closed:
        lines.append(
            f"  {r['family']} seed {r['seed']}: {', '.join(r[f'gate_{gate}']['blockers'])}"
        )
    if not closed:
        lines.append("  none")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--out", type=Path, default=Path("out/diag_trav.txt"))
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on an uncommitted tree (hash gets -dirty)"
    )
    args = parser.parse_args()

    rev = revision_label(args.allow_dirty)
    jobs = [(f, s, SCAN_STRATEGIES[0]) for f in args.families for s in range(args.seeds)]
    t0 = time.perf_counter()
    results = run_parallel(diagnose_job, jobs, args.workers)
    lines = [
        "# TerraNav traversability diagnosis (measured)",
        f"# git: {rev}",
        f"# date: {dt.datetime.now().isoformat(timespec='seconds')}",
        (
            f"# families {args.families}, seeds 0..{args.seeds - 1}, {SCAN_STRATEGIES[0]} survey, "
            f"{len(jobs)} scenarios, wall {time.perf_counter() - t0:.1f} s"
        ),
        "# core FF cells: corridor cells lethal in GT before dilation but perceived non-lethal.",
        "# OBS/FILL/FOOT: perceived source; GTslp/GTrgh/GTstp: GT reason (a cell may have several);",
        "# stp raw>=crit&gated: GT-step cells whose perceived raw jump >= step_crit but failed the",
        "# 3-sigma gate; med step/raw/slope(deg)/rough: perceived values on cells with that GT",
        (
            f"# reason (crit: step {CRIT.step_crit_m} m, slope {CRIT.slope_crit_deg} deg, rough "
            f"{CRIT.rough_crit_m} m); cam: distance to the nearest camera (m)."
        ),
        "",
    ]
    for gate in GATE_SETTINGS:
        lines += family_rows(results, gate)
    text = "\n".join(lines)
    sys.stdout.write(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
