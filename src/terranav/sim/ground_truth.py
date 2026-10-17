"""Ground truth on the map grid: elevation, features, traversability.

For evaluation and the future Oracle planner only; perception never imports this module.
"""

from __future__ import annotations

import numpy as np

from terranav.perception.elevation_map import OBSERVED, SIGMA_FLOOR_M, ElevationMap
from terranav.perception.features import TerrainFeatures, terrain_features
from terranav.perception.traversability import (
    Traversability,
    TraversabilityConfig,
    traversability,
)
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


def ground_truth_map(scenario: Scenario, map_grid: GridSpec) -> ElevationMap:
    """ElevationMap with every cell known at its ground-truth height.

    z_mean = z_min = z_max = GT; z_var = noise_var = 0; z_mean_var = SIGMA_FLOOR_M^2 (the
    smallest variance perception ever assigns, so weights stay finite and every GT jump is
    significant).
    """
    gt = ground_truth_elevation(scenario, map_grid)
    shape = gt.shape
    return ElevationMap(
        grid=map_grid,
        count=np.ones(shape, dtype=np.int32),
        z_min=gt,
        z_max=gt,
        z_mean=gt,
        z_mean_var=np.full(shape, SIGMA_FLOOR_M**2),
        z_var=np.zeros(shape),
        noise_var=np.zeros(shape),
        source=np.full(shape, OBSERVED, dtype=np.uint8),
    )


def ground_truth_features(scenario: Scenario, map_grid: GridSpec) -> TerrainFeatures:
    """Perception's terrain_features applied to the ground-truth map."""
    return terrain_features(ground_truth_map(scenario, map_grid))


def ground_truth_traversability(
    scenario: Scenario, map_grid: GridSpec, cfg: TraversabilityConfig
) -> Traversability:
    """Perception's traversability applied to ground-truth features (same rules, no noise)."""
    emap = ground_truth_map(scenario, map_grid)
    return traversability(emap, terrain_features(emap), cfg)
