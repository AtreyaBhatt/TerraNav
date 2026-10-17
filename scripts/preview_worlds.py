"""Eyeball check: one scenario per family, saved to out/worlds_preview.png.

All panels share one colour scale so heights are comparable across families.
Run: uv run python scripts/preview_worlds.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from terranav.config import SimConfig
from terranav.sim.scenario import FAMILIES, generate_scenario

OUT = Path("out/worlds_preview.png")


def main(seed: int = 0) -> None:
    cfg = SimConfig()
    g = cfg.terrain_grid
    extent = [g.origin[0], g.origin[0] + g.size_x, g.origin[1], g.origin[1] + g.size_y]
    scenarios = [generate_scenario(family, seed, cfg) for family in FAMILIES]
    vmin = min(sc.world.heightfield.min() for sc in scenarios)
    vmax = max(sc.world.heightfield.max() for sc in scenarios)

    fig, axes = plt.subplots(
        1, len(FAMILIES), figsize=(4 * len(FAMILIES), 4.4), layout="constrained"
    )
    for ax, sc in zip(axes, scenarios, strict=True):
        # Row index = iy = world y, so origin="lower" puts +y up and +x right.
        im = ax.imshow(
            sc.world.heightfield,
            origin="lower",
            extent=extent,
            cmap="terrain",
            vmin=vmin,
            vmax=vmax,
        )
        for b in sc.world.boxes:
            w, h = b.max_xyz[0] - b.min_xyz[0], b.max_xyz[1] - b.min_xyz[1]
            ax.add_patch(Rectangle(b.min_xyz[:2], w, h, fill=False, ec="red", lw=1.2))
        ax.plot(*zip(sc.start_xy, sc.goal_xy, strict=True), ":", color="k", lw=1)
        ax.plot(*sc.start_xy, "o", color="lime", mec="k", ms=8, label="start")
        ax.plot(*sc.goal_xy, "*", color="gold", mec="k", ms=12, label="goal")
        ax.set_title(f"{sc.family} (seed {seed})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="terrain height z [m]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
