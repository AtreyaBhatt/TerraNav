"""Diagnose traversability safety errors over TraversabilityConfig variants.

Runs diagnose_job (src/terranav/sim/trav_diag.py) for families x seeds 0..N-1 in parallel, for
step-lethality upper-confidence-bound factors --ucb-k (default 0 1 2). Writes to --out
(default out/diag_trav.txt), headed by the git commit, the start time and the wall time:
  A. per variant and family: corridor GT core / free counts, core false-free (count and rate),
     false-lethal rate split by the perceived cell's own cause, route closures;
  B. for the first variant: margins of the core false-free cells above the critical values
     (GT value - crit) and the perceived values of the same feature.
Refuses a dirty tree unless --allow-dirty.
Run: uv run python scripts/diag_trav.py [--seeds 10] [--families ...] [--ucb-k 0 1 2] [--out ...]
"""

from __future__ import annotations

import os

# Before numpy is imported anywhere: one native thread per process (workers inherit this).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime as dt
import math
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from terranav.perception.elevation_map import FILLED, FOOTPRINT, OBSERVED
from terranav.perception.traversability import R_ROUGH, R_SLOPE, R_STEP, TraversabilityConfig
from terranav.sim.scan_eval import revision_label, run_parallel
from terranav.sim.scenario import FAMILIES, SCAN_STRATEGIES
from terranav.sim.trav_diag import diagnose_job
from terranav.sim.trav_eval import FL_CAUSES

CRIT = TraversabilityConfig()
# Margin bins (GT value - crit), in cm for step / roughness and degrees for slope.
LENGTH_BINS_CM = ((0.0, "<1cm"), (1.0, "1-2cm"), (2.0, "2-5cm"), (5.0, ">5cm"))
SLOPE_BINS_DEG = ((0.0, "<1deg"), (1.0, "1-2deg"), (2.0, "2-5deg"), (5.0, ">5deg"))
# (GT reason bit, name, record key, crit in record units, record -> display scale, bins)
FEATURES = (
    (R_STEP, "step", "step_m", CRIT.step_crit_m, 100.0, LENGTH_BINS_CM),
    (R_ROUGH, "rough", "roughness_m", CRIT.rough_crit_m, 100.0, LENGTH_BINS_CM),
    (
        R_SLOPE,
        "slope",
        "slope_rad",
        math.radians(CRIT.slope_crit_deg),
        math.degrees(1.0),
        SLOPE_BINS_DEG,
    ),
)


def rate(num: float, den: float) -> str:
    return f"{num / den:.3f}" if den else "n/a (0)"


def variant_table(results: list[dict], name: str, families: list[str]) -> list[str]:
    head = f"{'family':<15} {'GT core':>8} {'core FF':>8} {'FF rate':>8} {'GT free':>8} {'FL':>7} "
    head += " ".join(f"{'FL ' + c[:5]:>8}" for c in FL_CAUSES) + f" {'closed':>7}"
    lines = [f"## A. variant {name}", head]
    keys = ("gt_core", "core_false_free", "gt_free", "false_lethal")
    for family in families:
        rs = [r[name] for r in results if r["family"] == family]
        t = {k: sum(r["counts"][k] for r in rs) for k in keys}
        fl = {c: sum(r["counts"][f"fl_{c}"] for r in rs) for c in FL_CAUSES}
        closed = sum(r["connected_gt"] and not r["connected"] for r in rs)
        row = f"{family:<15} {t['gt_core']:>8} {t['core_false_free']:>8} "
        row += f"{rate(t['core_false_free'], t['gt_core']):>8} {t['gt_free']:>8} "
        row += f"{rate(t['false_lethal'], t['gt_free']):>7} "
        row += " ".join(f"{rate(fl[c], t['gt_free']):>8}" for c in FL_CAUSES) + f" {closed:>7}"
        lines.append(row)
    lines.append("route closures (perceived disconnected, GT connected; blockers):")
    closures = [r for r in results if r[name]["connected_gt"] and not r[name]["connected"]]
    lines += [
        f"  {r['family']} seed {r['seed']}: {', '.join(r[name]['blockers'])}" for r in closures
    ]
    return [*lines, "  none" if not closures else "", ""]


def margin_table(results: list[dict], name: str, families: list[str]) -> list[str]:
    lines = [
        f"## B. core false-free margins, variant {name}",
        "GT margin = GT value - crit (a cell counts under every feature that makes it GT-lethal);",
        "perceived = the same feature on the perceived map (median, p90), crit in brackets;",
        "sources = perceived source of those cells, OBSERVED/FILLED/FOOTPRINT.",
        f"{'family':<15} {'feature':<6} {'cells':>6} | {'GT margin histogram':<40} | "
        + f"{'perc med':>8} {'perc p90':>8} {'(crit)':>7} | sources",
    ]
    for family in families:
        recs = [r[name]["records"] for r in results if r["family"] == family]
        cat = {k: np.concatenate([x[k] for x in recs]) for k in recs[0]}
        for bit, feat, key, crit, scale, bins in FEATURES:
            sel = (cat["gt_reason"] & bit) != 0
            if not sel.any():
                continue
            margin = (cat[f"gt_{key}"][sel] - crit) * scale
            counts = np.histogram(margin, bins=[lo for lo, _ in bins] + [np.inf])[0]
            hist = "  ".join(f"{label} {n}" for (_, label), n in zip(bins, counts, strict=True))
            perc = cat[key][sel] * scale
            src = cat["source"][sel]
            sources = (
                f"{np.sum(src == OBSERVED)}/{np.sum(src == FILLED)}/{np.sum(src == FOOTPRINT)}"
            )
            lines.append(
                f"{family:<15} {feat:<6} {int(sel.sum()):>6} | {hist:<40} | "
                f"{np.nanmedian(perc):>8.2f} {np.nanpercentile(perc, 90):>8.2f} "
                f"{f'({crit * scale:.3g})':>7} | {sources}"
            )
    lines.append("units: step / rough in cm, slope in degrees.")
    return [*lines, ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--ucb-k", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--out", type=Path, default=Path("out/diag_trav.txt"))
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on an uncommitted tree (hash gets -dirty)"
    )
    args = parser.parse_args()

    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    rev = revision_label(args.allow_dirty)
    variants = tuple((f"k={k:g}", replace(CRIT, lethal_ucb_k=k)) for k in args.ucb_k)
    jobs = [(f, s, SCAN_STRATEGIES[0]) for f in args.families for s in range(args.seeds)]
    print(f"{len(jobs)} scenarios x {len(variants)} variants on {args.workers} worker(s)")
    results = run_parallel(partial(diagnose_job, variants=variants), jobs, args.workers)
    lines = [
        "# TerraNav traversability diagnosis (measured)",
        f"# git: {rev}",
        f"# started: {started}",
        f"# wall time: {time.perf_counter() - t0:.1f} s",
        f"# families {args.families}, seeds 0..{args.seeds - 1}, {SCAN_STRATEGIES[0]} survey, "
        + f"{len(jobs)} scenarios; variants TraversabilityConfig(lethal_ucb_k=k), k in {args.ucb_k}",
        "# counts/rates over corridor cells; core FF = GT lethal before dilation, perceived free;",
        "# FL causes: the perceived cell's own reason (unknown | no estimate | feature | dilation).",
        "",
    ]
    for name, _ in variants:
        lines += variant_table(results, name, args.families)
    lines += margin_table(results, variants[0][0], args.families)
    text = "\n".join(lines)
    sys.stdout.write(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
