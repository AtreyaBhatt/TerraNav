"""Capture a placeholder survey pass and render the fused point cloud in 3D.

For the chosen (family, seed): k scan poses along start->goal (scan_poses), one capture per
pose with rng = np.random.default_rng((seed, frame_index)), backprojection to world, then
an offscreen PyVista render to out/scene3d.png (oblique) and out/scene3d_top.png (top-down).
Run: uv run python scripts/render_scene.py --family mixed --seed 0 [--frames 6] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from terranav.perception.backproject import backproject
from terranav.sim.scenario import FAMILIES, generate_scenario, scan_poses
from terranav.sim.sensor import capture
from terranav.types import SensorConfig
from terranav.viz.scene3d import render_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=6, help="number of scan poses (k)")
    parser.add_argument("--out", type=Path, default=Path("out/scene3d.png"))
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    args = parser.parse_args()

    cfg = SensorConfig()
    sc = generate_scenario(args.family, args.seed)
    poses = scan_poses(sc, k=args.frames, sensor=cfg)
    points_list = []
    for i, T in enumerate(poses):
        frame, _ = capture(sc.world, T, cfg, np.random.default_rng((args.seed, i)))
        points, _ = backproject(frame)
        points_list.append(points)
        print(f"frame {i}: camera at {np.round(T[:3, 3], 2)}, {len(points)} valid points")
    oblique, top = render_scene(sc, poses, points_list, args.out, cfg.intrinsics(), show=args.show)
    print(f"Saved {oblique} and {top}")


if __name__ == "__main__":
    main()
