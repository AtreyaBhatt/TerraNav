"""Capture a placeholder survey pass, render it in 3D, and plot top-down coverage.

For the chosen (family, seed, strategy): scan poses (scan_poses), one capture per pose with
rng = np.random.default_rng((seed, frame_index)), backprojection to world, then
  out/scene3d_<family>_s<seed>.png (+ _top) : offscreen PyVista render
  out/coverage_top_<family>_s<seed>.png      : frames-per-cell on the 0.1 m map grid
and the measured corridor / reachable coverage printed to stdout.
Run: uv run python scripts/render_scene.py --family mixed --seed 0 [--strategy flank+start] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from terranav.config import SimConfig
from terranav.perception.backproject import backproject
from terranav.sim.coverage import corridor_mask, coverage, observation_counts, reachable_mask
from terranav.sim.scenario import (
    DEFAULT_SCAN_K,
    FAMILIES,
    SCAN_STRATEGIES,
    generate_scenario,
    scan_poses,
)
from terranav.sim.sensor import capture
from terranav.viz.coverage2d import render_coverage_top
from terranav.viz.scene3d import render_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=SCAN_STRATEGIES, default=SCAN_STRATEGIES[0])
    parser.add_argument(
        "--frames", type=int, default=None, help="scan poses k (default per strategy)"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    args = parser.parse_args()

    cfg = SimConfig()
    sensor, map_grid = cfg.sensor, cfg.map_grid
    sc = generate_scenario(args.family, args.seed, cfg)
    poses = scan_poses(sc, k=args.frames, strategy=args.strategy, sensor=sensor)
    k = DEFAULT_SCAN_K[args.strategy] if args.frames is None else args.frames
    if len(poses) < k:
        print(f"{k - len(poses)} of {k} poses skipped ({args.strategy})")
    points_list = []
    for i, T in enumerate(poses):
        frame, _ = capture(sc.world, T, sensor, np.random.default_rng((args.seed, i)))
        points, _ = backproject(frame)
        points_list.append(points)
        print(f"frame {i}: camera at {np.round(T[:3, 3], 2)}, {len(points)} valid points")

    corridor = corridor_mask(sc, map_grid)
    reachable = reachable_mask(sc, map_grid)
    print(
        f"coverage ({args.strategy}): corridor {coverage(points_list, map_grid, corridor):.3f}, "
        f"reachable {coverage(points_list, map_grid, reachable):.3f}"
    )
    tag = f"{args.family}_s{args.seed}"
    oblique, top = render_scene(
        sc,
        poses,
        points_list,
        args.out_dir / f"scene3d_{tag}.png",
        sensor.intrinsics(),
        show=args.show,
    )
    cov = render_coverage_top(
        sc,
        poses,
        observation_counts(points_list, map_grid),
        map_grid,
        corridor,
        args.out_dir / f"coverage_top_{tag}.png",
        title=f"{args.family} seed {args.seed}, {args.strategy}: frames per 0.1 m cell",
    )
    print(f"Saved {oblique}, {top}, {cov}")


if __name__ == "__main__":
    main()
