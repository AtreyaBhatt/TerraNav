"""Ground-truth elevation on the map grid (evaluation and the future Oracle planner only)."""

from __future__ import annotations

import numpy as np

from terranav.sim.world import cell_centers_xy, coarsen_factor
from terranav.types import GridSpec, Scenario


def ground_truth_elevation(scenario: Scenario, map_grid: GridSpec) -> np.ndarray:
    """(ny, nx) float64 true surface height (m) per map cell.

    Per cell: the larger of (a) the mean terrain height over the f x f terrain cells inside it
    and (b) the top of any box whose x-y footprint (closed) contains the cell centre.
    """
    world = scenario.world
    f = coarsen_factor(world.grid, map_grid)
    hf = world.heightfield
    gt = hf.reshape(map_grid.ny, f, map_grid.nx, f).mean(axis=(1, 3))
    x, y = cell_centers_xy(map_grid)
    for b in world.boxes:
        covered = (x >= b.min_xyz[0]) & (x <= b.max_xyz[0])
        covered &= (y >= b.min_xyz[1]) & (y <= b.max_xyz[1])
        gt = np.where(covered, np.maximum(gt, b.max_xyz[2]), gt)
    return gt
