"""Benchmark: wall time of one sensor capture (ray-cast + sensor model).

Scene: `mixed` seed 0, camera at the start pose, default SensorConfig. Frame i uses
np.random.default_rng((0, i)). One warm-up run, then the mean of --runs runs.
Run: uv run python scripts/bench_capture.py [--runs 5]
"""

from __future__ import annotations

import argparse
import math
import platform
import time

import numpy as np

from terranav.frames import T_world_cam
from terranav.sim.scenario import generate_scenario
from terranav.sim.sensor import capture
from terranav.types import SensorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    cfg = SensorConfig()
    sc = generate_scenario("mixed", 0)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    T = T_world_cam(
        np.array([sx, sy, cfg.mount_height_m]), math.atan2(gy - sy, gx - sx), cfg.pitch_down_rad
    )
    capture(sc.world, T, cfg, np.random.default_rng((0, 0)))  # warm-up
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        capture(sc.world, T, cfg, np.random.default_rng((0, i)))
        times.append(time.perf_counter() - t0)
    print(
        f"capture {cfg.width}x{cfg.height} on {platform.processor() or platform.machine()}: "
        f"mean {np.mean(times):.3f} s, min {min(times):.3f}, max {max(times):.3f} "
        f"({args.runs} runs)"
    )


if __name__ == "__main__":
    main()
