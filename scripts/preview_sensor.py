"""Eyeball check: ideal depth, noisy depth and invalid_reason for one `mixed` scenario.

Camera at the start pose (mount_height_m above the base-height start disk, yawed toward the
goal, pitched down by SensorConfig.pitch_down_rad). Frame RNG follows the documented rule
np.random.default_rng((scenario_seed, frame_index)). Saved to out/sensor_preview.png.
Run: uv run python scripts/preview_sensor.py [seed]
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
from matplotlib.patches import Patch

from terranav.frames import T_world_cam
from terranav.sim.scenario import generate_scenario
from terranav.sim.sensor import BEYOND_MAX, DROPOUT, GRAZING, MISS, capture
from terranav.types import SensorConfig

OUT = Path("out/sensor_preview.png")

# Categories for display. A pixel can carry GRAZING and BEYOND_MAX together; it is shown
# with the first matching bit in this order.
CATEGORIES = (
    ("valid", None, "#d9d9d9"),
    ("miss", MISS, "#7f3b08"),
    ("grazing", GRAZING, "#e08214"),
    ("beyond max", BEYOND_MAX, "#2d004b"),
    ("dropout", DROPOUT, "#8073ac"),
)


def categorise(reason: np.ndarray) -> np.ndarray:
    cat = np.zeros(reason.shape, dtype=np.int8)
    for i in range(len(CATEGORIES) - 1, 0, -1):  # earlier entries win, so assign them last
        cat[(reason & CATEGORIES[i][1]) != 0] = i
    return cat


def main(seed: int = 0, frame_index: int = 0) -> None:
    cfg = SensorConfig()
    sc = generate_scenario("mixed", seed)
    (sx, sy), (gx, gy) = sc.start_xy, sc.goal_xy
    T = T_world_cam(
        np.array([sx, sy, cfg.mount_height_m]), math.atan2(gy - sy, gx - sx), cfg.pitch_down_rad
    )
    frame, dbg = capture(sc.world, T, cfg, np.random.default_rng((seed, frame_index)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("magenta")
    for ax, img, title in (
        (axes[0], dbg.depth_z_ideal, "ideal depth_z"),
        (axes[1], frame.depth_z, "noisy depth_z (DepthFrame)"),
    ):
        im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=cfg.max_depth_z_m)
        ax.set_title(f"{title}, magenta = NaN")
    fig.colorbar(im, ax=axes[:2], shrink=0.85, label="depth_z [m]")

    colours = ListedColormap([c for _, _, c in CATEGORIES])
    axes[2].imshow(
        categorise(dbg.invalid_reason),
        cmap=colours,
        vmin=-0.5,
        vmax=len(CATEGORIES) - 0.5,
        interpolation="nearest",
    )
    counts = [
        int(np.sum(dbg.invalid_reason == 0)),
        *[int(np.sum((dbg.invalid_reason & bit) != 0)) for _, bit, _ in CATEGORIES[1:]],
    ]
    handles = [
        Patch(color=c, label=f"{name} ({n})")
        for (name, _, c), n in zip(CATEGORIES, counts, strict=True)
    ]
    axes[2].legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    axes[2].set_title("invalid_reason (pixel counts per bit)")
    for ax in axes:
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
    fig.suptitle(f"mixed seed {seed}, frame {frame_index}, start pose")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(
        f"Saved {OUT}; reason counts {dict(zip([c[0] for c in CATEGORIES], counts, strict=True))}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
