"""Eyeball check: ideal depth_z and hit_kind for one `mixed` scenario from the start pose.

The camera sits mount_height_m above the terrain at start, yawed toward the goal, pitched
down by SensorConfig.pitch_down_rad. Saved to out/depth_preview.png; NaN (miss) is magenta.
Run: uv run python scripts/preview_depth.py [seed]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from terranav.frames import T_world_cam
from terranav.sim.raycast import HIT_BOX, HIT_NONE, HIT_TERRAIN, cast_frame, range_to_depth
from terranav.sim.scenario import generate_scenario
from terranav.types import SensorConfig

OUT = Path("out/depth_preview.png")


def main(seed: int = 0) -> None:
    sensor = SensorConfig()
    intr = sensor.intrinsics()
    sc = generate_scenario("mixed", seed)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    # Start disk is at base height 0 by construction (scenario keep-out).
    T = T_world_cam(
        np.array([sx, sy, sensor.mount_height_m]),
        math.atan2(gy - sy, gx - sx),
        sensor.pitch_down_rad,
    )
    res = cast_frame(sc.world, T, intr, sensor.max_depth_z_m)
    depth_z = range_to_depth(res.range_t, intr)

    fig, (ax_d, ax_k) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("magenta")
    im = ax_d.imshow(depth_z, cmap=cmap, vmin=0.0, vmax=sensor.max_depth_z_m)
    fig.colorbar(im, ax=ax_d, label="depth_z [m]  (magenta = NaN / miss)")
    ax_d.set_title(f"depth_z, mixed seed {seed} (ideal, no noise)")

    kinds = ListedColormap(["magenta", "tab:green", "tab:red"])
    imk = ax_k.imshow(res.hit_kind, cmap=kinds, vmin=-0.5, vmax=2.5, interpolation="nearest")
    cb = fig.colorbar(imk, ax=ax_k, ticks=[HIT_NONE, HIT_TERRAIN, HIT_BOX])
    cb.ax.set_yticklabels(["none", "terrain", "box"])
    ax_k.set_title("hit_kind")
    for ax in (ax_d, ax_k):
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
