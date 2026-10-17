"""Build the elevation map for one scenario (default survey) and render four diagnostic panels.

Frame i uses np.random.default_rng((seed, i)). Output: out/elevation_<family>_s<seed>.png.
Run: uv run python scripts/render_map.py --family mixed --seed 0 [--strategy flank+start]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terranav.config import SimConfig
from terranav.sim.coverage import corridor_mask
from terranav.sim.ground_truth import ground_truth_elevation
from terranav.sim.map_eval import build_map, known_connected, survey_frames
from terranav.sim.scenario import FAMILIES, SCAN_STRATEGIES
from terranav.viz.map2d import render_map_panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=SCAN_STRATEGIES, default=SCAN_STRATEGIES[0])
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args()

    cfg = SimConfig()
    sc, frames = survey_frames((args.family, args.seed, args.strategy), None, cfg.sensor, cfg)
    emap = build_map(sc, frames, cfg.sensor, cfg)
    connected = known_connected(emap, sc.start_xy, sc.goal_xy)
    out = render_map_panels(
        sc,
        emap,
        ground_truth_elevation(sc, cfg.map_grid),
        corridor_mask(sc, cfg.map_grid),
        args.out_dir / f"elevation_{args.family}_s{args.seed}.png",
        title=(
            f"{args.family} seed {args.seed}, {args.strategy} survey ({len(frames)} frames), "
            f"start-goal known-connected: {connected}"
        ),
    )
    print(f"Saved {out}; start-goal connected through known cells: {connected}")


if __name__ == "__main__":
    main()
