"""Hero still of the showcase scenario (docs/img/showcase.json): docs/img/hero.png, 1920x1080.

Terrain coloured by GT traversability cost, the three paths (TerraNav, Baseline-Occ, Oracle
dashed), start/goal, a legend with each path's length and rough exposure from the evaluator.
The run is recomputed (same calls as the benchmark job) and checked against showcase.json.
Run: uv run python scripts/render_hero.py [--showcase docs/img/showcase.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terranav.config import SimConfig
from terranav.sim.showcase import load_checked_run, showcase_caption
from terranav.viz.showcase3d import render_hero, unknown_blocked_cells


def gap_note(run, grid) -> str | None:
    """The legend's observation, stated only if this run supports it: Oracle's path crosses
    cells that TerraNav's map blocks because of nearby unknown cells, and TerraNav's path is
    longer than Oracle's."""
    blocked = unknown_blocked_cells(run, "oracle", grid)
    tn, orc = run.metrics["terranav"], run.metrics["oracle"]
    print(f"Oracle path cells blocked on TerraNav's map by unknown cells: {blocked}")
    if blocked == 0 or tn is None or orc is None or tn.length_m <= orc.length_m:
        return None
    return (
        "Oracle uses the gap between patches; TerraNav, which cannot see it\n"
        f"({blocked} cells on Oracle's route are unknown-blocked on its map), detours wider"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showcase", type=Path, default=Path("docs/img/showcase.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/img/hero.png"))
    args = parser.parse_args()
    show, run = load_checked_run(args.showcase)
    grid = SimConfig().map_grid
    out = render_hero(
        run, grid, showcase_caption(show), args.out, show["alpha"], gap_note(run, grid)
    )
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
