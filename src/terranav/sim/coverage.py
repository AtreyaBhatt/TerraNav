"""Scan coverage measurement on the map grid (evaluation only; uses ground truth regions).

Coverage of a region = fraction of its map cells that contain at least one backprojected
point (from any frame). Points are binned by their world (x, y) only; points outside the
map are ignored.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from terranav.sim.scenario import easy_mask
from terranav.sim.world import cell_centers_xy
from terranav.types import GridSpec, Scenario

CORRIDOR_HALF_WIDTH_M = 3.0


def observation_counts(points_list: Sequence[np.ndarray], map_grid: GridSpec) -> np.ndarray:
    """(ny, nx) int: number of frames with >= 1 point in each map cell."""
    counts = np.zeros((map_grid.ny, map_grid.nx), dtype=np.int64)
    for points in points_list:
        if len(points) == 0:
            continue
        row, col = map_grid.xy_to_rc(points[:, 0], points[:, 1])
        ok = (row >= 0) & (row < map_grid.ny) & (col >= 0) & (col < map_grid.nx)
        seen = np.zeros_like(counts, dtype=bool)
        seen[row[ok], col[ok]] = True
        counts += seen
    return counts


def coverage(
    points_list: Sequence[np.ndarray], map_grid: GridSpec, region_mask: np.ndarray
) -> float:
    """Fraction of region_mask cells (map grid) holding >= 1 point from any frame."""
    region = np.asarray(region_mask, dtype=bool)
    if region.shape != (map_grid.ny, map_grid.nx):
        raise ValueError(f"region shape {region.shape} != map grid {(map_grid.ny, map_grid.nx)}")
    if not region.any():
        raise ValueError("region_mask is empty")
    return float(np.mean(observation_counts(points_list, map_grid)[region] > 0))


def corridor_mask(
    scenario: Scenario, map_grid: GridSpec, half_width_m: float = CORRIDOR_HALF_WIDTH_M
) -> np.ndarray:
    """Map cells whose centre lies within half_width_m of the start->goal SEGMENT."""
    x, y = cell_centers_xy(map_grid)
    s, g = np.asarray(scenario.start_xy), np.asarray(scenario.goal_xy)
    d = g - s
    t = np.clip(((x - s[0]) * d[0] + (y - s[1]) * d[1]) / (d @ d), 0.0, 1.0)
    return np.hypot(x - (s[0] + t * d[0]), y - (s[1] + t * d[1])) <= half_width_m


def reachable_mask(scenario: Scenario, map_grid: GridSpec) -> np.ndarray:
    """Map cells fully inside the easy-ground component (4-connected) containing start.

    Computed on the terrain grid, then reduced to the map grid: a map cell is in the region
    iff ALL terrain cells inside it are in the component (map resolution must be an integer
    multiple of the terrain resolution, with the same origin and extent).
    """
    tgrid = scenario.world.grid
    f = round(map_grid.resolution / tgrid.resolution)
    if (
        abs(f * tgrid.resolution - map_grid.resolution) > 1e-9
        or (map_grid.ny * f, map_grid.nx * f) != (tgrid.ny, tgrid.nx)
        or map_grid.origin != tgrid.origin
    ):
        raise ValueError("map grid must be an integer coarsening of the terrain grid")
    four_connected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labels, _ = ndimage.label(easy_mask(scenario), structure=four_connected)
    r, c = tgrid.xy_to_rc(*scenario.start_xy)
    start_label = labels[int(r), int(c)]
    if start_label == 0:
        raise ValueError("start cell is not on easy ground")
    comp = labels == start_label
    return comp.reshape(map_grid.ny, f, map_grid.nx, f).all(axis=(1, 3))
