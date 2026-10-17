"""Top-down coverage plot: map cells coloured by how many frames observed them."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from terranav.sim.world import cell_centers_xy
from terranav.types import GridSpec, Scenario

ARROW_M = 0.8  # drawn length of each pose's heading arrow


def render_coverage_top(
    scenario: Scenario,
    poses: Sequence[np.ndarray],
    counts: np.ndarray,
    map_grid: GridSpec,
    corridor: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Save a top-down image of per-cell frame counts (0 = grey) on the map grid.

    counts: (ny, nx) int frames-per-cell (sim.coverage.observation_counts); corridor: (ny, nx)
    bool region whose outline is drawn. Poses are drawn as arrows along their optical axis's
    horizontal component, numbered by frame index. World x right, y up (origin="lower").
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = max(len(poses), int(counts.max()), 1)
    cmap = ListedColormap(["#c8c8c8", *plt.get_cmap("viridis")(np.linspace(0.0, 1.0, k))])
    norm = BoundaryNorm(np.arange(-0.5, k + 1.5), cmap.N)
    x0, y0 = map_grid.origin
    extent = [x0, x0 + map_grid.size_x, y0, y0 + map_grid.size_y]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")
    im = ax.imshow(
        counts, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest"
    )
    fig.colorbar(im, ax=ax, ticks=range(k + 1), label="frames observing the cell (0 = never)")
    x, y = cell_centers_xy(map_grid)
    ax.contour(
        x, y, corridor.astype(float), levels=[0.5], colors="k", linestyles="--", linewidths=1
    )
    for i, T in enumerate(poses):
        px, py = T[0, 3], T[1, 3]
        d = T[:2, 2] / np.linalg.norm(T[:2, 2])  # optical axis projected on the ground
        ax.annotate(
            "",
            xy=(px + ARROW_M * d[0], py + ARROW_M * d[1]),
            xytext=(px, py),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.5},
        )
        ax.plot(px, py, "o", color="red", ms=4)
        ax.text(px, py, f" {i}", color="red", fontsize=9, va="bottom")
    ax.plot(*scenario.start_xy, "o", color="lime", mec="k", ms=10, label="start")
    ax.plot(*scenario.goal_xy, "*", color="gold", mec="k", ms=15, label="goal")
    ax.plot([], [], "k--", lw=1, label="corridor band")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
