"""Pick the showcase scenario by a fixed rule over DEV seeds (no cherry-picking).

Rule (terranav.sim.showcase.SHOWCASE_RULE): among dev seeds 0..9 of the rough family, the
scenario whose paired TerraNav - Baseline-Occ rough-exposure difference at alpha = 4 is the
median (upper middle for an even count). Prints the full ranking and writes
docs/img/showcase.json. Runs the benchmark job on the 10 dev scenarios (alpha = 4 only).
Run: uv run python scripts/pick_showcase.py [--workers N]
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path

from terranav.planning.planners import DEFAULT_ALPHA
from terranav.sim.scan_eval import revision_label
from terranav.sim.showcase import (
    DEV_SEEDS,
    SHOWCASE_FAMILY,
    SHOWCASE_METRIC,
    SHOWCASE_RULE,
    dev_records,
    paired_metric,
    pick_median,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--out", type=Path, default=Path("docs/img/showcase.json"))
    args = parser.parse_args()
    rev = revision_label(allow_dirty=True)  # labels the choice; the rule does not depend on it

    records = dev_records(SHOWCASE_FAMILY, DEV_SEEDS, DEFAULT_ALPHA, min(args.workers, 10))
    diffs = paired_metric(records, SHOWCASE_METRIC, DEFAULT_ALPHA)
    ranking, seed = pick_median(diffs)
    print(f"rule: {SHOWCASE_RULE}")
    print("rank  seed  TerraNav - Baseline-Occ rough exposure [m^2]")
    for i, (s, d) in enumerate(ranking, start=1):
        print(f"{i:>4}  {s:>4}  {d:+.4f}{'   <- chosen' if s == seed else ''}")
    failed = sorted(set(DEV_SEEDS) - set(diffs))
    if failed:
        print(f"excluded (a planner failed): seeds {failed}")

    chosen = {
        r["planner"]: r
        for r in records
        if r["seed"] == seed and r["alpha"] in (None, DEFAULT_ALPHA)
    }
    out = {
        "family": SHOWCASE_FAMILY,
        "seed": seed,
        "alpha": DEFAULT_ALPHA,
        "rule": SHOWCASE_RULE,
        "git": rev,
        "ranking": [{"seed": s, "diff_rough_exposure_m2": d} for s, d in ranking],
        "excluded_seeds": failed,
        "metrics": {
            p: {
                "success": r["success"],
                "length_m": r["metrics"]["length_m"] if r["success"] else None,
                "rough_exposure_m2": r["metrics"]["rough_exposure_m2"] if r["success"] else None,
                "max_rise_m": r["metrics"]["max_rise_m"] if r["success"] else None,
                "max_slope_deg": r["metrics"]["max_slope_deg"] if r["success"] else None,
                "violations": r["metrics"]["violations"] if r["success"] else None,
            }
            for p, r in chosen.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"chosen: {SHOWCASE_FAMILY} seed {seed}; saved {args.out}")


if __name__ == "__main__":
    main()
