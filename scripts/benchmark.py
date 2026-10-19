"""Run the pre-registered held-out benchmark (docs/benchmark_protocol.md).

Seeds 1000..1019 per family (BenchmarkConfig), every alpha in the sweep, parallel workers.
Writes results/<git-short>_<config-hash>/: raw.jsonl, meta.json, summary.md, tradeoff_<family>.png,
headline.png. Refuses a dirty tree (no override: results must be tied to a commit) and refuses
to overwrite an existing results directory.
Run: uv run python scripts/benchmark.py [--workers N]
"""

from __future__ import annotations

import os

# Before numpy is imported anywhere: one native thread per process (workers inherit this).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime as dt
import sys
from pathlib import Path

from terranav.sim.benchmark import BenchmarkConfig, config_hash, run_benchmark
from terranav.sim.scan_eval import revision_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    started = dt.datetime.now().isoformat(timespec="seconds")

    rev = revision_label(allow_dirty=False)  # exits on a dirty tree
    bcfg = BenchmarkConfig()
    out = args.results_root / f"{rev[:7]}_{config_hash(bcfg=bcfg)}"
    if out.exists():
        sys.exit(f"error: {out} already exists; a pre-registered run is not repeated in place")
    n = len(bcfg.families) * len(bcfg.seeds)
    print(f"{n} scenarios x {len(bcfg.alphas)} alphas on {args.workers} worker(s) -> {out}")

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} scenarios done", end="" if done < total else "\n", flush=True)

    run_benchmark(bcfg, out, args.workers, git=rev, started=started, progress=progress)
    print(f"Saved {out}/ (summary.md, raw.jsonl, meta.json, figures)")


if __name__ == "__main__":
    main()
