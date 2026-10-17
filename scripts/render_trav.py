"""Render perceived vs ground-truth traversability for one scenario (default survey).

Output: out/trav_<family>_s<seed>.png (slope, roughness, step, perceived cost, GT cost,
disagreement). Frame i uses np.random.default_rng((seed, i)).
Run: uv run python scripts/render_trav.py --family mixed --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terranav.config import SimConfig
from terranav.perception.features import terrain_features
from terranav.perception.traversability import TraversabilityConfig, traversability
from terranav.sim.coverage import corridor_mask
from terranav.sim.ground_truth import ground_truth_map
from terranav.sim.map_eval import build_map, survey_frames
from terranav.sim.scenario import FAMILIES, SCAN_STRATEGIES
from terranav.sim.trav_eval import traversable_connected
from terranav.viz.trav2d import render_trav_panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=SCAN_STRATEGIES, default=SCAN_STRATEGIES[0])
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args()

    cfg, tcfg = SimConfig(), TraversabilityConfig()
    grid = cfg.map_grid
    sc, frames = survey_frames((args.family, args.seed, args.strategy), None, cfg.sensor, cfg)
    emap = build_map(sc, frames, cfg.sensor, cfg)
    feats = terrain_features(emap)
    trav = traversability(emap, feats, tcfg)
    gt_map = ground_truth_map(sc, grid)
    gt = traversability(gt_map, terrain_features(gt_map), tcfg)
    conn = traversable_connected(trav, grid, sc.start_xy, sc.goal_xy)
    conn_gt = traversable_connected(gt, grid, sc.start_xy, sc.goal_xy)
    out = render_trav_panels(
        sc,
        grid,
        feats,
        trav,
        gt,
        corridor_mask(sc, grid),
        tcfg,
        args.out_dir / f"trav_{args.family}_s{args.seed}.png",
        title=(
            f"{args.family} seed {args.seed}, {args.strategy} survey; start-goal traversable: "
            f"perceived {conn}, GT {conn_gt}"
        ),
    )
    print(f"Saved {out}; start-goal traversable: perceived {conn}, GT {conn_gt}")


if __name__ == "__main__":
    main()
