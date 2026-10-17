"""Benchmark: wall time of one cast_frame at the default sensor resolution.

Scene: `mixed` seed 0, camera at the start pose (mount height, yaw toward goal, default
pitch), max_depth_z_m from SensorConfig. One warm-up run, then the mean of --runs runs.
Run: uv run python scripts/bench_raycast.py [--runs 5]
"""

from __future__ import annotations

import argparse
import math
import platform
import time

import numpy as np

from terranav.frames import T_world_cam
from terranav.sim.raycast import cast_frame
from terranav.sim.scenario import generate_scenario
from terranav.types import SensorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    cfg = SensorConfig()
    intr = cfg.intrinsics()
    sc = generate_scenario("mixed", 0)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    T = T_world_cam(
        np.array([sx, sy, cfg.mount_height_m]), math.atan2(gy - sy, gx - sx), cfg.pitch_down_rad
    )
    cast_frame(sc.world, T, intr, cfg.max_depth_z_m)  # warm-up
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        cast_frame(sc.world, T, intr, cfg.max_depth_z_m)
        times.append(time.perf_counter() - t0)
    print(
        f"cast_frame {intr.width}x{intr.height} on {platform.processor() or platform.machine()}: "
        f"mean {np.mean(times):.3f} s, min {min(times):.3f}, max {max(times):.3f} "
        f"({args.runs} runs)"
    )


if __name__ == "__main__":
    main()
