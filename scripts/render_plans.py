"""Plan one scenario with TerraNav, Baseline-Occ and Oracle, and render the paths.

Output: out/plans_<family>_s<seed>.png. Left: perceived cost with the two perceived-map
planners; right: ground-truth cost with all three. Prints runtime and A* expansions per
planner. Frame i of the survey uses np.random.default_rng((seed, i)).
Run: uv run python scripts/render_plans.py --family mixed --seed 0 [--alpha 4.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terranav.config import SimConfig
from terranav.planning.planners import DEFAULT_ALPHA
from terranav.sim.pipeline import run_planners
from terranav.sim.scenario import FAMILIES, SCAN_STRATEGIES
from terranav.viz.plans2d import path_length_m, render_plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args()

    grid = SimConfig().map_grid
    run = run_planners((args.family, args.seed, SCAN_STRATEGIES[0]), alpha=args.alpha)
    for name, p in run.plans.items():
        if p is None:
            print(f"{args.family} s{args.seed} {name:<13} no path")
            continue
        print(
            f"{args.family} s{args.seed} {name:<13} length {path_length_m(p, grid):6.2f} m  "
            f"cost {p.cost:6.3f}  vertices {len(p.path_rc):3d}  expansions {p.expansions:6d}  "
            f"runtime {p.runtime_s * 1e3:7.1f} ms"
        )
    out = render_plans(
        run,
        grid,
        args.out_dir / f"plans_{args.family}_s{args.seed}.png",
        title=f"{args.family} seed {args.seed}, alpha = {args.alpha:g} (planning cost model)",
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
