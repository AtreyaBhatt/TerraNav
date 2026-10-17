"""Small-gap filling and robot-footprint marking for an ElevationMap.

Unknown cells are treated as blocked by planning, so narrow unobserved gaps (between scan
lines, behind small occluders) would wrongly cut the map. fill_gaps closes only gaps up to
max_gap_m wide, and only across locally flat ground; larger or stepped unknown regions stay
unknown.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy import ndimage

from terranav.perception.elevation_map import FILLED, FOOTPRINT, ElevationMap

# ILLUSTRATIVE: widest unobserved gap (m) that is bridged by interpolation.
MAX_GAP_M = 0.3
# ILLUSTRATIVE: no fill where observed heights in the window span more than this (m), e.g.
# around box rims whose tops were never seen. M2-B's step threshold is meant to reuse it.
STEP_GUARD_M = 0.10
# Robot footprint radius (m) marked as known around the start pose.
FOOTPRINT_RADIUS_M = 0.4


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return xx**2 + yy**2 <= r**2


def fill_gaps(
    emap: ElevationMap, max_gap_m: float = MAX_GAP_M, step_guard_m: float = STEP_GUARD_M
) -> ElevationMap:
    """Fill unobserved cells inside gaps up to max_gap_m wide (metres).

    r = ceil(max_gap_m / 2 / res) cells.
    fillable = binary_closing(observed, disk(r)) & ~observed, computed on a map padded by r
        cells replicating its edge, so the map border neither creates nor erodes gaps.
    Window = the (2r+1)^2 square around the cell; lo / hi = min z_min / max z_max over the
    observed cells in it. Guard: a cell is NOT filled if hi - lo > step_guard_m.
    z_mean: normalized convolution, sum(G * observed * z_mean) / sum(G * observed), with a
        Gaussian G of sigma r/3 cells truncated at 3 sigma, i.e. exactly the window.
    z_min / z_max of a filled cell = lo / hi. Filled cells keep count 0 (z_var, noise_var
    stay NaN) and get source FILLED.
    """
    res = emap.grid.resolution
    r = math.ceil(max_gap_m / 2 / res)
    observed = emap.count > 0
    if r < 1 or not observed.any():
        return emap
    padded = np.pad(observed, r, mode="edge")
    closed = ndimage.binary_closing(padded, structure=_disk(r))[r:-r, r:-r]
    window = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    lo = ndimage.minimum_filter(np.where(observed, emap.z_min, np.inf), footprint=window)
    hi = ndimage.maximum_filter(np.where(observed, emap.z_max, -np.inf), footprint=window)
    # Every closed cell has an observed cell within disk(r), so lo / hi are finite there.
    fillable = closed & ~observed
    fillable[fillable] = (hi - lo)[fillable] <= step_guard_m
    if not fillable.any():
        return emap

    sigma = r / 3.0
    z_obs = np.where(observed, emap.z_mean, 0.0)
    num = ndimage.gaussian_filter(z_obs, sigma, mode="constant", truncate=3.0)
    den = ndimage.gaussian_filter(observed.astype(float), sigma, mode="constant", truncate=3.0)
    return replace(
        emap,
        z_mean=np.where(fillable, num / np.where(fillable, den, 1.0), emap.z_mean),
        z_min=np.where(fillable, lo, emap.z_min),
        z_max=np.where(fillable, hi, emap.z_max),
        source=np.where(fillable, FILLED, emap.source),
    )


def mark_footprint(
    emap: ElevationMap,
    center_xy: tuple[float, float],
    radius_m: float,
    heights: np.ndarray,
) -> ElevationMap:
    """Mark cells whose centre is within radius_m of center_xy as FOOTPRINT.

    heights: (ny, nx) terrain heights (m) supplied by the CALLER for the cells the robot
    stands on — proprioception (its own pose and ground contact), not the depth sensor. Only
    the footprint cells are read. Apply after fill_gaps so these cells are never overwritten.
    z_mean = z_min = z_max = heights; z_var = noise_var = 0; count unchanged.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if heights.shape != emap.count.shape:
        raise ValueError(f"heights shape {heights.shape} != map {emap.count.shape}")
    x, y = emap.grid.cell_centers_xy()
    foot = (x - center_xy[0]) ** 2 + (y - center_xy[1]) ** 2 <= radius_m**2
    return replace(
        emap,
        z_mean=np.where(foot, heights, emap.z_mean),
        z_min=np.where(foot, heights, emap.z_min),
        z_max=np.where(foot, heights, emap.z_max),
        z_var=np.where(foot, 0.0, emap.z_var),
        noise_var=np.where(foot, 0.0, emap.noise_var),
        source=np.where(foot, FOOTPRINT, emap.source),
    )
