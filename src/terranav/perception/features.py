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

STEP_SIGMA_GATE = 3.0  # a jump counts only if > 3 sigma of the two cell means combined
# ILLUSTRATIVE: steps compare cells up to this far apart (centre to centre, Euclidean).
STEP_PAIR_DIST_M = 0.2
# ILLUSTRATIVE: a significant pair jump of at least this much is an "edge"; roughness is zeroed
# for every cell whose plane-fit window contains an edge endpoint (see terrain_features).
ROUGH_EDGE_STEP_M = 0.10


def pair_offsets(grid_res_m: float, max_dist_m: float = STEP_PAIR_DIST_M) -> list[tuple[int, int]]:
    """(drow, dcol) offsets with 0 < |d| * res <= max_dist_m, one of each +-pair.

    On the 0.1 m grid with 0.2 m: (0,1), (1,-1), (1,0), (1,1), (0,2), (2,0).
    """
    r = int(max_dist_m / grid_res_m + 1e-9)
    return [
        (dr, dc)
        for dr in range(r + 1)
        for dc in range(-r, r + 1)
        if (dr > 0 or dc > 0) and (dr * dr + dc * dc) * grid_res_m**2 <= max_dist_m**2 + 1e-12
    ]


@dataclass(frozen=True, slots=True, eq=False)
class TerrainFeatures:
    """Per-cell features on the map grid, (ny, nx), read-only; NaN where not computable.

    slope_rad: arctan(hypot(a, b)) of the local plane fit.
    roughness_m: sqrt(max(0, residual^2 - fit_noise_var)): plane-fit residual with the
        modelled measurement noise of the fitted cell means removed; 0 near edges (see
        terrain_features).
    step_m: largest SIGNIFICANT height jump over the cell pairs (<= STEP_PAIR_DIST_M apart) the
        cell is an endpoint of (0 if none); no spreading to other cells, the robot's extent is
        handled only by traversability's dilation.
    step_raw_m: largest jump over the same pairs without the significance gate.
    support: known cells in the plane-fit disk.
    """

    slope_rad: np.ndarray
    roughness_m: np.ndarray
    step_m: np.ndarray
    step_raw_m: np.ndarray
    support: np.ndarray

    def __post_init__(self) -> None:
        for name in ("slope_rad", "roughness_m", "step_m", "step_raw_m", "support"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def pair_jumps(
    z: np.ndarray,
    var: np.ndarray,
    offsets: list[tuple[int, int]],
    gate: float = STEP_SIGMA_GATE,
    ucb_k: float = 0.0,
) -> np.ndarray:
    """(ny, nx): per cell, the largest |z_i - z_j| + ucb_k * sigma_ij over the pairs
    (i, i + offset) it belongs to, with sigma_ij = sqrt(var_i + var_j).

    A pair counts only if both cells are known (finite z) and |z_i - z_j| > gate * sigma_ij;
    gate = 0 gives the raw (ungated) maximum. ucb_k > 0 turns the jump into an upper confidence
    bound (taken per pair, before the max). Cells with no such pair get 0. Both endpoints of a
    pair receive its value; nothing else.

    Why pairs up to 0.2 m (not only adjacent cells): a vertical edge that falls inside a cell
    makes that cell's mean a mix of the two levels, splitting one jump into two adjacent
    half-jumps (a 0.24 m box edge reads as 2 x 0.12 m, each below step_crit). Cells two apart
    straddle the mixed cell and see the whole jump.
    Why not z_max / z_min: those are noise extremes of many points, whose spread grows with
    range (sigma = s0 + s1 * z^2); means with the 3-sigma gate stay calibrated.
    Clean slopes stay below step_crit: over 0.2 m a 30 deg slope rises 0.2 * tan(30 deg) = 0.115 m.
    """
    ny, nx = z.shape
    best = np.zeros((ny, nx))
    for dr, dc in offsets:
        r0, r1 = slice(0, ny - dr), slice(dr, ny)
        c0 = slice(max(0, -dc), nx - max(0, dc))
        c1 = slice(max(0, dc), nx - max(0, -dc))
        jump = np.abs(z[r1, c1] - z[r0, c0])
        sigma = np.sqrt(var[r1, c1] + var[r0, c0])
        sig = jump > gate * sigma  # False where NaN
        j = np.where(sig, jump + ucb_k * sigma, 0.0)
        best[r0, c0] = np.maximum(best[r0, c0], j)
        best[r1, c1] = np.maximum(best[r1, c1], j)
    return best


def terrain_features(
    emap: ElevationMap, fit_radius_m: float = PLANE_FIT_RADIUS_M
) -> TerrainFeatures:
    """Compute slope, roughness and step for every known, sufficiently supported cell.

    Edge-aware roughness: roughness measures surface texture; height edges are the step
    feature's job. A plane fitted across an edge reports the edge as residual, so a
    sub-critical step (say 0.15 m) would otherwise surround itself with a band of high
    "roughness" one fit radius wide. (Diagnosis at 6c4fd3f: all 64 core false-free cells of the
    steps family were roughness-only, with GT within 1 cm of rough_crit, along terrace edges.)
    So roughness is set to 0 for every cell whose plane-fit window (radius fit_radius_m)
    contains an endpoint of a SIGNIFICANT pair jump >= ROUGH_EDGE_STEP_M. Perception and ground
    truth use this same function.
    Limitation: texture within ~fit_radius_m of a >= ROUGH_EDGE_STEP_M edge is ignored (an
    edge inside rough ground hides that ground's roughness). The scenario generator does not
    overlap features, so rough patches and edges are separated in this benchmark.
    """
    fit = plane_fit(emap, fit_radius_m)
    valid = (emap.source != UNKNOWN) & np.isfinite(fit.a)
    slope = np.arctan(np.hypot(fit.a, fit.b))
    rough = np.sqrt(np.maximum(0.0, fit.residual_m**2 - fit.fit_noise_var))

    known = emap.source != UNKNOWN
    z = np.where(known, emap.z_mean, np.nan)
    var = np.where(known, emap.z_mean_var, np.nan)
    offsets = pair_offsets(emap.grid.resolution)
    step = pair_jumps(z, var, offsets)
    disk, _, _ = disk_offsets(math.floor(fit_radius_m / emap.grid.resolution + 1e-9))
    near_edge = ndimage.binary_dilation(step >= ROUGH_EDGE_STEP_M, structure=disk)
    rough = np.where(near_edge, 0.0, rough)

    def masked(a: np.ndarray) -> np.ndarray:
        return np.where(valid, a, np.nan)

    return TerrainFeatures(
        slope_rad=masked(slope),
        roughness_m=masked(rough),
        step_m=masked(step),
        step_raw_m=masked(pair_jumps(z, var, offsets, gate=0.0)),
        support=fit.support,
    )
