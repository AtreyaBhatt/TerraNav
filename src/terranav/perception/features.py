"""Noise-aware terrain features per map cell: slope, roughness, step height.

All features are NaN where the cell is unknown or its plane fit has insufficient support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from terranav.perception.elevation_map import UNKNOWN, ElevationMap
from terranav.perception.plane_fit import PLANE_FIT_RADIUS_M, disk_offsets, plane_fit

STEP_RADIUS_M = 0.2  # r_step: a cell reports the largest significant jump within this radius
STEP_SIGMA_GATE = 3.0  # a jump counts only if > 3 sigma of the two cell means combined
# Neighbour directions (drow, dcol) covering every 8-neighbour pair exactly once.
PAIR_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True, slots=True, eq=False)
class TerrainFeatures:
    """Per-cell features on the map grid, (ny, nx), read-only; NaN where not computable.

    slope_rad: arctan(hypot(a, b)) of the local plane fit.
    roughness_m: sqrt(max(0, residual^2 - fit_noise_var)): plane-fit residual with the
        modelled measurement noise of the fitted cell means removed.
    step_m: largest significant height jump between 8-neighbouring known cells with an
        endpoint within STEP_RADIUS_M of the cell (0 if none).
    support: known cells in the plane-fit disk.
    """

    slope_rad: np.ndarray
    roughness_m: np.ndarray
    step_m: np.ndarray
    support: np.ndarray

    def __post_init__(self) -> None:
        for name in ("slope_rad", "roughness_m", "step_m", "support"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def pair_jumps(z: np.ndarray, var: np.ndarray, gate: float = STEP_SIGMA_GATE) -> np.ndarray:
    """(ny, nx): per cell, the largest significant |z_i - z_j| over its 8-neighbour pairs.

    A pair counts only if both cells are known (finite z) and
    |z_i - z_j| > gate * sqrt(var_i + var_j). Cells with no such pair get 0.
    On a clean plane of slope s the largest adjacent jump is the diagonal one,
    tan(s) * 0.1 m * sqrt(2) on the 0.1 m grid (~0.066 m at 25 deg), below the step threshold.
    """
    ny, nx = z.shape
    best = np.zeros((ny, nx))
    for dr, dc in PAIR_DIRECTIONS:
        r0, r1 = slice(0, ny - dr), slice(dr, ny)
        c0 = slice(max(0, -dc), nx - max(0, dc))
        c1 = slice(max(0, dc), nx - max(0, -dc))
        jump = np.abs(z[r1, c1] - z[r0, c0])
        sig = jump > gate * np.sqrt(var[r1, c1] + var[r0, c0])  # False where NaN
        j = np.where(sig, jump, 0.0)
        best[r0, c0] = np.maximum(best[r0, c0], j)
        best[r1, c1] = np.maximum(best[r1, c1], j)
    return best


def terrain_features(
    emap: ElevationMap,
    fit_radius_m: float = PLANE_FIT_RADIUS_M,
    step_radius_m: float = STEP_RADIUS_M,
) -> TerrainFeatures:
    """Compute slope, roughness and step for every known, sufficiently supported cell."""
    fit = plane_fit(emap, fit_radius_m)
    valid = (emap.source != UNKNOWN) & np.isfinite(fit.a)
    slope = np.arctan(np.hypot(fit.a, fit.b))
    rough = np.sqrt(np.maximum(0.0, fit.residual_m**2 - fit.fit_noise_var))

    known = emap.source != UNKNOWN
    z = np.where(known, emap.z_mean, np.nan)
    var = np.where(known, emap.z_mean_var, np.nan)
    disk, _, _ = disk_offsets(math.floor(step_radius_m / emap.grid.resolution + 1e-9))
    step = ndimage.maximum_filter(pair_jumps(z, var), footprint=disk, mode="constant", cval=0.0)

    def masked(a: np.ndarray) -> np.ndarray:
        return np.where(valid, a, np.nan)

    return TerrainFeatures(
        slope_rad=masked(slope),
        roughness_m=masked(rough),
        step_m=masked(step),
        support=fit.support,
    )
