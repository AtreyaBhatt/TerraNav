"""Heightfield primitives and box placement.

All primitives are pure: they take a heightfield (ny, nx) float64 in metres on a GridSpec,
indexed [row=iy, col=ix], and return a NEW array (or a Box). None of them draw random
numbers; randomness is pre-sampled by the caller (parameters, noise arrays).
World frame: x forward, y left, z up (docs/frames.md).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from terranav.types import Box, GridSpec

# Fraction of a rough patch's radius over which its amplitude tapers linearly to zero,
# so the patch does not introduce an artificial step at its rim.
ROUGH_TAPER_FRACTION = 0.2


def cell_centers_xy(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) of every cell centre, metres; each array has shape (ny, nx)."""
    row, col = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
    return grid.rc_to_xy(row, col)


def bilinear_sample(field: np.ndarray, grid: GridSpec, x, y) -> np.ndarray:
    """Bilinear sample of a (ny, nx) grid field at world (x, y), metres. Returns shape of x.

    Cell-centre convention: fractional col = (x - origin_x)/res - 0.5, row likewise from y;
    scipy map_coordinates takes (row, col). The outer half-cell of the map clamps to the
    edge values (mode="nearest"); callers decide what "outside the map" means.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    col = (x - grid.origin[0]) / grid.resolution - 0.5
    row = (y - grid.origin[1]) / grid.resolution - 0.5
    coords = np.stack([np.ravel(row), np.ravel(col)])
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest").reshape(np.shape(x))


def _check_hf(hf: np.ndarray, grid: GridSpec) -> np.ndarray:
    hf = np.asarray(hf, dtype=np.float64)
    if hf.shape != (grid.ny, grid.nx):
        raise ValueError(f"heightfield shape {hf.shape} != (ny, nx) {(grid.ny, grid.nx)}")
    return hf


def _rect_local_distance(
    grid: GridSpec, center_xy: tuple[float, float], size_xy: tuple[float, float], yaw_rad: float
) -> np.ndarray:
    """Signed distance (m) from each cell centre to the nearest side of a rotated rectangle.

    The rectangle is size_xy = (length, width) metres, centred at center_xy, its length axis
    at yaw_rad from world +x. In rectangle-local coordinates (a, b):
        d = min(length/2 - |a|, width/2 - |b|)
    d > 0 inside, d < 0 outside. Returns (ny, nx) float64.
    """
    length, width = size_xy
    x, y = cell_centers_xy(grid)
    dx, dy = x - center_xy[0], y - center_xy[1]
    a = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    b = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    return np.minimum(length / 2 - np.abs(a), width / 2 - np.abs(b))


def add_ramp(
    hf: np.ndarray,
    grid: GridSpec,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_rad: float,
    slope_deg: float,
    h_max_m: float,
) -> np.ndarray:
    """Add a frustum (truncated pyramid) over a rotated rectangle.

    With d the distance to the nearest footprint side (see _rect_local_distance):
        h = clip(tan(slope_deg) * d, 0, h_max_m)
    so every face rises at slope_deg from the footprint edge, and the top is flat at h_max_m
    if the footprint is large enough. Outside the footprint d < 0 and the terrain is unchanged.
    """
    hf = _check_hf(hf, grid)
    d = _rect_local_distance(grid, center_xy, size_xy, yaw_rad)
    return hf + np.clip(math.tan(math.radians(slope_deg)) * d, 0.0, h_max_m)


def add_rough_patch(
    hf: np.ndarray,
    grid: GridSpec,
    center_xy: tuple[float, float],
    radius: float,
    rms_m: float,
    corr_len_m: float,
    noise: np.ndarray,
) -> np.ndarray:
    """Add spatially correlated roughness inside a disk.

    `noise` is a pre-sampled (ny, nx) array (e.g. standard normal). It is smoothed with
    ndimage.gaussian_filter(sigma = corr_len_m / resolution), rescaled to RMS rms_m (metres),
    and added inside the disk with a linear taper over the outer ROUGH_TAPER_FRACTION of the
    radius. Cells whose centres are at distance >= radius are unchanged.
    """
    hf = _check_hf(hf, grid)
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape != hf.shape:
        raise ValueError(f"noise shape {noise.shape} != heightfield shape {hf.shape}")
    smooth = ndimage.gaussian_filter(noise, sigma=corr_len_m / grid.resolution)
    std = smooth.std()
    if std > 0:
        smooth = smooth * (rms_m / std)
    x, y = cell_centers_xy(grid)
    r = np.hypot(x - center_xy[0], y - center_xy[1])
    weight = np.clip((radius - r) / (ROUGH_TAPER_FRACTION * radius), 0.0, 1.0)
    return hf + weight * smooth


def add_terrace(
    hf: np.ndarray,
    grid: GridSpec,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_rad: float,
    step_height_m: float,
    n_steps: int,
    step_depth_m: float,
    direction: int,
) -> np.ndarray:
    """Add a stepped mesa (direction=+1) or pit (direction=-1) over a rotated rectangle.

    With d the distance to the nearest footprint side (see _rect_local_distance), inside the
    footprint (d >= 0):
        h = direction * step_height_m * min(floor(d / step_depth_m) + 1, n_steps)
    i.e. concentric rings step_depth_m wide, each one step_height_m higher (or lower) than
    the ring outside it. The centre reaches direction * n_steps * step_height_m when
    min(size_xy) / 2 >= (n_steps - 1) * step_depth_m. Outside the footprint: unchanged.
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    hf = _check_hf(hf, grid)
    d = _rect_local_distance(grid, center_xy, size_xy, yaw_rad)
    level = np.minimum(np.floor(d / step_depth_m) + 1, n_steps)
    return hf + np.where(d >= 0, direction * step_height_m * level, 0.0)


def _footprint_heights(
    hf: np.ndarray, grid: GridSpec, xmin: float, xmax: float, ymin: float, ymax: float
) -> np.ndarray:
    """Heights of cells whose centres lie in the footprint (or the cell under its centre)."""
    c0 = max(math.ceil((xmin - grid.origin[0]) / grid.resolution - 0.5), 0)
    c1 = min(math.floor((xmax - grid.origin[0]) / grid.resolution - 0.5), grid.nx - 1)
    r0 = max(math.ceil((ymin - grid.origin[1]) / grid.resolution - 0.5), 0)
    r1 = min(math.floor((ymax - grid.origin[1]) / grid.resolution - 0.5), grid.ny - 1)
    if c1 < c0 or r1 < r0:
        r, c = grid.xy_to_rc((xmin + xmax) / 2, (ymin + ymax) / 2)
        return hf[int(r), int(c)].reshape(1)
    return hf[r0 : r1 + 1, c0 : c1 + 1]


def box_on_terrain(
    grid: GridSpec,
    hf: np.ndarray,
    min_xy: tuple[float, float],
    max_xy: tuple[float, float],
    height_m: float,
) -> Box:
    """Axis-aligned box over footprint [min_xy, max_xy] (world metres) resting on the terrain.

    min_z = lowest terrain under the footprint; max_z = highest terrain under it + height_m.
    """
    hf = _check_hf(hf, grid)
    under = _footprint_heights(hf, grid, min_xy[0], max_xy[0], min_xy[1], max_xy[1])
    z0, z1 = float(under.min()), float(under.max()) + float(height_m)
    return Box((float(min_xy[0]), float(min_xy[1]), z0), (float(max_xy[0]), float(max_xy[1]), z1))
