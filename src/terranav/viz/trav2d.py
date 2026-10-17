"""Six-panel top-down view of perceived vs ground-truth traversability."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from terranav.perception.features import TerrainFeatures
from terranav.perception.traversability import Traversability, TraversabilityConfig
from terranav.types import GridSpec, Scenario

UNKNOWN_GREY = "#c8c8c8"


def _cmap(name: str, bad: str) -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad)
    return cmap


def render_trav_panels(
    scenario: Scenario,
    grid: GridSpec,
    feats: TerrainFeatures,
    trav: Traversability,
    gt_trav: Traversability,
    corridor: np.ndarray,
    cfg: TraversabilityConfig,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Slope, roughness, step (NaN grey; colour scales end at the lethal thresholds), perceived
    and GT cost (lethal black), and disagreement (false-free red, false-lethal blue)."""
    x0, y0 = grid.origin
    extent = [x0, x0 + grid.size_x, y0, y0 + grid.size_y]
    common = {"origin": "lower", "extent": extent, "interpolation": "nearest"}
    x, y = grid.cell_centers_xy()
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5), layout="constrained")
    ax = axes.ravel()

    panels = (
        (np.degrees(feats.slope_rad), cfg.slope_crit_deg, "slope [deg]"),
        (feats.roughness_m, cfg.rough_crit_m, "roughness [m]"),
        (feats.step_m, cfg.step_crit_m, "step [m]"),
    )
    for a, (img, vmax, label) in zip(ax[:3], panels, strict=True):
        im = a.imshow(img, cmap=_cmap("viridis", UNKNOWN_GREY), vmin=0, vmax=vmax, **common)
        fig.colorbar(im, ax=a, shrink=0.8, extend="max", label=f"{label} (scale ends at lethal)")
        a.set_title(f"perceived {label.split()[0]} (grey = no estimate)")
    for a, t, name in ((ax[3], trav, "perceived"), (ax[4], gt_trav, "ground-truth")):
        im = a.imshow(t.cost, cmap=_cmap("YlOrRd", "black"), vmin=0, vmax=1, **common)
        fig.colorbar(im, ax=a, shrink=0.8, label="cost [0, 1) (black = lethal)")
        a.set_title(f"{name} cost")

    diff = np.zeros(trav.lethal.shape, dtype=np.int8)
    diff[gt_trav.lethal & ~trav.lethal] = 1  # false-free (unsafe)
    diff[~gt_trav.lethal & trav.lethal] = 2  # false-lethal (conservative)
    colours = ("#f0f0f0", "#d62728", "#1f77b4")
    ax[5].imshow(diff, cmap=ListedColormap(colours), vmin=-0.5, vmax=2.5, **common)
    counts = [int(np.sum(diff == i)) for i in range(3)]
    labels = ("agree", "false-free (GT lethal)", "false-lethal (GT free)")
    ax[5].legend(
        handles=[
            Patch(color=c, label=f"{n} ({k})")
            for c, n, k in zip(colours, labels, counts, strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8,
    )
    ax[5].set_title("perceived vs ground truth (all cells)")

    for a in ax:
        a.contour(
            x, y, corridor.astype(float), levels=[0.5], colors="k", linestyles="--", linewidths=0.8
        )
        a.plot(*scenario.start_xy, "o", color="lime", mec="k", ms=8)
        a.plot(*scenario.goal_xy, "*", color="gold", mec="k", ms=13)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
    fig.suptitle(f"{title}   (o start, * goal, -- corridor band)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
