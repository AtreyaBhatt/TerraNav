"""Four-panel top-down view of an ElevationMap: height, source, point count, error vs GT."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch

from terranav.perception.elevation_map import FILLED, FOOTPRINT, OBSERVED, UNKNOWN, ElevationMap
from terranav.sim.world import cell_centers_xy
from terranav.types import Scenario

SOURCES = (
    (UNKNOWN, "unknown", "#c8c8c8"),
    (OBSERVED, "observed", "#1f77b4"),
    (FILLED, "filled", "#ff7f0e"),
    (FOOTPRINT, "footprint", "#2ca02c"),
)
ERROR_VMAX_M = 0.1  # error colour scale is clipped here so typical errors stay readable


def render_map_panels(
    scenario: Scenario,
    emap: ElevationMap,
    gt: np.ndarray,
    corridor: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Save z_mean, source, count and |z_mean - GT| panels (world x right, y up)."""
    grid = emap.grid
    x0, y0 = grid.origin
    extent = [x0, x0 + grid.size_x, y0, y0 + grid.size_y]
    x, y = cell_centers_xy(grid)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), layout="constrained")
    common = {"origin": "lower", "extent": extent, "interpolation": "nearest"}

    height_cmap = plt.get_cmap("viridis").copy()
    height_cmap.set_bad("#c8c8c8")
    im = axes[0].imshow(emap.z_mean, cmap=height_cmap, **common)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="z_mean [m] (grey = unknown)")
    axes[0].set_title("fused height z_mean")

    src_cmap = ListedColormap([c for _, _, c in SOURCES])
    axes[1].imshow(emap.source, cmap=src_cmap, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), **common)
    counts = {name: int(np.sum(emap.source == code)) for code, name, _ in SOURCES}
    axes[1].legend(
        handles=[Patch(color=c, label=f"{n} ({counts[n]})") for _, n, c in SOURCES],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8,
    )
    axes[1].set_title("source (cells)")

    count = np.where(emap.count > 0, emap.count, np.nan)
    count_cmap = plt.get_cmap("magma").copy()
    count_cmap.set_bad("#c8c8c8")
    vmax = max(2.0, float(np.nanmax(count))) if np.isfinite(count).any() else 2.0
    im = axes[2].imshow(count, cmap=count_cmap, norm=LogNorm(1, vmax), **common)
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="points per cell (log; grey = 0)")
    axes[2].set_title("point count")

    err_cmap = plt.get_cmap("Reds").copy()
    err_cmap.set_bad("#c8c8c8")
    im = axes[3].imshow(
        np.abs(emap.z_mean - gt), cmap=err_cmap, vmin=0, vmax=ERROR_VMAX_M, **common
    )
    fig.colorbar(im, ax=axes[3], shrink=0.8, extend="max", label="|z_mean - GT| [m]")
    axes[3].set_title("height error vs ground truth")

    for ax in axes:
        ax.contour(
            x, y, corridor.astype(float), levels=[0.5], colors="k", linestyles="--", linewidths=0.8
        )
        ax.plot(*scenario.start_xy, "o", color="lime", mec="k", ms=8)
        ax.plot(*scenario.goal_xy, "*", color="gold", mec="k", ms=13)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.suptitle(f"{title}   (o start, * goal, -- corridor band)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
