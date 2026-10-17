"""Test helper: build ElevationMaps directly from per-cell arrays (no sensor involved)."""

from __future__ import annotations

import numpy as np

from terranav.perception.elevation_map import OBSERVED, UNKNOWN, ElevationMap
from terranav.types import GridSpec

MAP = GridSpec(resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))


def cell_xy(grid: GridSpec = MAP) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre (x, y), each (ny, nx), computed by hand from the grid layout."""
    x = grid.origin[0] + (np.arange(grid.nx) + 0.5) * grid.resolution
    y = grid.origin[1] + (np.arange(grid.ny) + 0.5) * grid.resolution
    return np.meshgrid(x, y, indexing="xy")


def emap_from(
    z: np.ndarray,
    known: np.ndarray | None = None,
    z_mean_var: float | np.ndarray = 1e-8,
    noise_var: float | np.ndarray = 0.0,
    grid: GridSpec = MAP,
) -> ElevationMap:
    """ElevationMap whose known cells have z_mean = z_min = z_max = z (one point each)."""
    shape = (grid.ny, grid.nx)
    known = np.ones(shape, dtype=bool) if known is None else known
    z = np.where(known, z, np.nan)

    def on_known(v):
        return np.where(known, np.broadcast_to(v, shape), np.nan)

    return ElevationMap(
        grid=grid,
        count=known.astype(np.int32),
        z_min=z,
        z_max=z,
        z_mean=z,
        z_mean_var=on_known(z_mean_var),
        z_var=on_known(0.0),
        noise_var=on_known(noise_var),
        source=np.where(known, OBSERVED, UNKNOWN),
    )
