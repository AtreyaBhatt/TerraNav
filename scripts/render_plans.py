"""Plan one scenario with TerraNav, Baseline-Occ and Oracle, and render the paths.

Output: out/plans_<family>_s<seed>.png. Left: perceived cost with the two perceived-map
planners; right: ground-truth cost with all three. Prints runtime and A* expansions per
planner and the independent PathMetrics (terranav.sim.evaluate); violations (collision point,
max-rise point) are marked with red crosses on the GT panel. Frame i of the survey uses
np.random.default_rng((seed, i)).
Run: uv run python scripts/render_plans.py --family mixed --seed 0 [--alpha 4.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from terranav.config import SimConfig
from terranav.planning.planners import DEFAULT_ALPHA
from terranav.sim.evaluate import PhysicalLimits, evaluate_path
from terranav.sim.pipeline import gt_cost_along, run_planners
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
    limits = PhysicalLimits()
    marks: dict[str, list] = {}
    for name, p in run.plans.items():
        if p is None:
            print(f"{args.family} s{args.seed} {name:<13} no path")
            continue
        print(
            f"{args.family} s{args.seed} {name:<13} length {path_length_m(p, grid):6.2f} m  "
            f"cost {p.cost:6.3f}  vertices {len(p.path_rc):3d}  expansions {p.expansions:6d}  "
            f"runtime {p.runtime_s * 1e3:7.1f} ms"
        )
        x, y = grid.rc_to_xy(p.path_rc[:, 0], p.path_rc[:, 1])
        m = evaluate_path(run.scenario, np.column_stack([x, y]), limits)
        gt_cost, gt_lethal = gt_cost_along(p.path_rc, run.trav["ground_truth"], grid)
        print(
            f"    PathMetrics: clearance {m.min_clearance_m:.3f} m, max rise {m.max_rise_m:.3f} m, "
            f"max slope {m.max_slope_deg:.1f} deg (pitch {m.max_pitch_deg:.1f}, roll "
            f"{m.max_roll_deg:.1f}), mean slope {m.mean_slope_deg:.1f} deg, rough exposure "
            f"{m.rough_exposure_m2:.4f} m^2 over {m.length_in_rough_m:.2f} m, on features "
            f"{ {k: round(float(v), 2) for k, v in m.length_on_features_m.items()} }, violations "
            f"{list(m.violations) or 'none'}"
        )
        print(f"    gt_cost (definition-dependent) {gt_cost:.3f}, on GT-lethal {gt_lethal:.2f} m")
        marks[name] = []
        if m.collision:
            marks[name].append((m.collision_xy, "collision"))
        if "rise" in m.violations:
            marks[name].append((m.max_rise_xy, f"rise {m.max_rise_m:.2f} m"))
    out = render_plans(
        run,
        grid,
        args.out_dir / f"plans_{args.family}_s{args.seed}.png",
        title=f"{args.family} seed {args.seed}, alpha = {args.alpha:g} (planning cost model)",
        violations=marks,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
