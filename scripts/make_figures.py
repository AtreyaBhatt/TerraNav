"""Regenerate the presentation figures from a committed benchmark run (raw.jsonl only).

Writes <out>/headline.png and <out>/tradeoff_<family>.png (rough, slopes, steps, mixed); every
headline mean / CI is checked against the run's summary.md first. Nothing in the run directory
is modified and nothing is rerun.
Run: uv run python scripts/make_figures.py results/0cf7be5_96a4e7ffad40 [--out docs/img]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terranav.viz.figures import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    paths = make_figures(args.run_dir, args.out)
    print(f"headline values match {args.run_dir / 'summary.md'}")
    for p in paths:
        print(f"Saved {p} ({p.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
