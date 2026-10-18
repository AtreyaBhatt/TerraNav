"""Two-panel plan comparison: perceived cost with perceived-map plans, GT cost with all plans."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from terranav.planning.astar import PlanResult
from terranav.sim.pipeline import PipelineRun
from terranav.types import GridSpec

STYLES = {
    "terranav": {"color": "tab:green", "lw": 2.2, "ls": "-"},
    "baseline_occ": {"color": "tab:blue", "lw": 2.0, "ls": "--"},
    "oracle": {"color": "tab:purple", "lw": 1.8, "ls": ":"},
}


def path_length_m(p: PlanResult, grid: GridSpec) -> float:
    """Geometric length of the polyline through the cell centres, metres."""
    return grid.resolution * sum(math.hypot(*(b - a)) for a, b in pairwise(p.path_rc))


def render_plans(run: PipelineRun, grid: GridSpec, out_path: str | Path, title: str = "") -> Path:
    x0, y0 = grid.origin
    extent = [x0, x0 + grid.size_x, y0, y0 + grid.size_y]
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("black")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    panels = (
        (axes[0], "perceived", ("terranav", "baseline_occ"), "perceived cost (black = lethal)"),
        (axes[1], "ground_truth", ("terranav", "baseline_occ", "oracle"), "ground-truth cost"),
    )
    for ax, source, names, label in panels:
        im = ax.imshow(
            run.trav[source].cost,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        for name in names:
            p = run.plans[name]
            if p is None:
                ax.plot([], [], label=f"{name}: no path", **STYLES[name])
                continue
            x, y = grid.rc_to_xy(p.path_rc[:, 0], p.path_rc[:, 1])
            ax.plot(x, y, label=f"{name}: {path_length_m(p, grid):.2f} m", **STYLES[name])
        ax.plot(*run.scenario.start_xy, "o", color="lime", mec="k", ms=10, label="start")
        ax.plot(*run.scenario.goal_xy, "*", color="gold", mec="k", ms=15, label="goal")
        ax.set_title(label)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="cost [0, 1)")
    fig.suptitle(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
