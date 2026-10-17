"""2.5D elevation map: per-cell height statistics fused from backprojected depth frames.

No ground segmentation: the world has no overhangs (heightfield + solid boxes), so one
height distribution per map cell carries what the cost function needs; boxes show up as
large steps (docs/decisions.md). World frame, metres; arrays are (ny, nx) on the map grid,
indexed [row=iy, col=ix].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from terranav.perception.backproject import backproject
from terranav.sensor_model import noise_sigma
from terranav.types import DepthFrame, GridSpec, SensorConfig

# source codes (uint8)
UNKNOWN, OBSERVED, FILLED, FOOTPRINT = 0, 1, 2, 3

# Lower bound on per-point sigma (m) so inverse-variance weights stay finite when the noise
# model is switched off (s0 = s1 = 0). With equal sigmas the weighted mean is the plain mean.
SIGMA_FLOOR_M = 1e-4


@dataclass(frozen=True, slots=True, eq=False)
class ElevationMap:
    """Per-cell height statistics on `grid`; all arrays (ny, nx) and read-only.

    count: int32 number of fused points.
    z_min, z_max: extreme point heights (m).
    z_mean: inverse-variance weighted mean height (m), weights 1 / sigma(depth_z)^2.
    z_mean_var: variance of that weighted mean (m^2), 1 / sum(1 / sigma_i^2).
    z_var: unweighted population variance (ddof = 0) of the cell's point heights (m^2);
        0 for single-point cells. Measures within-cell spread (roughness, steps).
    noise_var: mean of the points' modelled sensor variances sigma^2 (m^2).
    source: uint8 UNKNOWN / OBSERVED / FILLED / FOOTPRINT.
    Height fields are NaN where nothing is known (count == 0 and not filled or footprint).
    """

    grid: GridSpec
    count: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_mean_var: np.ndarray
    z_var: np.ndarray
    noise_var: np.ndarray
    source: np.ndarray

    def __post_init__(self) -> None:
        shape = (self.grid.ny, self.grid.nx)
        dtypes = {"count": np.int32, "source": np.uint8}
        names = ("count", "z_min", "z_max", "z_mean", "z_mean_var", "z_var", "noise_var", "source")
        for name in names:
            arr = np.array(getattr(self, name), dtype=dtypes.get(name, np.float64), copy=True)
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != grid {shape}")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def fuse_points(points: np.ndarray, sigma: np.ndarray, map_grid: GridSpec) -> ElevationMap:
    """Bin world points (N, 3) with per-point height std sigma (N,) into an ElevationMap.

    Binning is floor((x - origin_x) / res) etc. (GridSpec.xy_to_rc), so the map covers the
    half-open box [x0, x0 + size_x) x [y0, y0 + size_y): a point exactly on a cell boundary
    goes to the higher-index cell, and points outside are dropped. Vectorised; the result
    does not depend on point order (beyond float summation order for z_mean / z_var).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    ny, nx = map_grid.ny, map_grid.nx
    row, col = map_grid.xy_to_rc(points[:, 0], points[:, 1])
    keep = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
    idx = (row[keep] * nx + col[keep]).astype(np.intp)
    z, s = points[keep, 2], sigma[keep]
    n_cells = ny * nx

    count = np.bincount(idx, minlength=n_cells)
    seen = count > 0
    n = np.where(seen, count, 1)  # avoid 0-division; unseen cells are set to NaN below
    w = 1.0 / np.square(np.maximum(s, SIGMA_FLOOR_M))
    sum_w = np.where(seen, np.bincount(idx, w, n_cells), 1.0)
    z_mean = np.bincount(idx, w * z, n_cells) / sum_w
    plain_mean = np.bincount(idx, z, n_cells) / n
    z_var = np.bincount(idx, np.square(z - plain_mean[idx]), n_cells) / n  # two-pass, ddof 0
    noise_var = np.bincount(idx, np.square(s), n_cells) / n
    z_min = np.full(n_cells, np.inf)
    z_max = np.full(n_cells, -np.inf)
    np.minimum.at(z_min, idx, z)
    np.maximum.at(z_max, idx, z)

    def grid(a: np.ndarray) -> np.ndarray:
        return np.where(seen, a, np.nan).reshape(ny, nx)

    return ElevationMap(
        grid=map_grid,
        count=count.reshape(ny, nx),
        z_min=grid(z_min),
        z_max=grid(z_max),
        z_mean=grid(z_mean),
        z_mean_var=grid(1.0 / sum_w),
        z_var=grid(z_var),
        noise_var=grid(noise_var),
        source=np.where(seen, OBSERVED, UNKNOWN).reshape(ny, nx),
    )


def fuse(
    frames: Sequence[DepthFrame], sensor_cfg: SensorConfig, map_grid: GridSpec
) -> ElevationMap:
    """Backproject and fuse depth frames into one ElevationMap on map_grid.

    Each point's sigma comes from its measured depth_z through the sensor's datasheet noise
    model sigma = s0 + s1 * depth_z^2 (SensorConfig). Perception knows that model, NOT the
    noise realisation or the ideal depth: the simulator's ground truth is never used here.
    """
    all_points, all_sigma = [np.zeros((0, 3))], [np.zeros(0)]
    for frame in frames:
        points, vu = backproject(frame)
        depth_z = frame.depth_z[vu[:, 0], vu[:, 1]].astype(np.float64)
        all_points.append(points)
        all_sigma.append(noise_sigma(depth_z, sensor_cfg))
    return fuse_points(np.vstack(all_points), np.concatenate(all_sigma), map_grid)
