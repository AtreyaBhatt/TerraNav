"""Traversability: lethal mask and [0, 1) cost from terrain features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from terranav.perception.elevation_map import UNKNOWN, ElevationMap
from terranav.perception.features import TerrainFeatures
from terranav.perception.plane_fit import disk_offsets


@dataclass(frozen=True)
class TraversabilityConfig:
    """ALL VALUES ILLUSTRATIVE: plausible for a small wheeled robot, not from a vehicle model."""

    slope_crit_deg: float = 30.0  # slope at/above this is lethal
    rough_crit_m: float = 0.05  # roughness (residual RMS) at/above this is lethal
    step_crit_m: float = 0.20  # step height at/above this is lethal
    w_slope: float = 0.4  # cost weights; they sum to 1, so cost < 1 off lethal cells
    w_rough: float = 0.3
    w_step: float = 0.3
    robot_radius_m: float = 0.3  # lethal cells are grown by this disk (robot footprint)


@dataclass(frozen=True, slots=True, eq=False)
class Traversability:
    """lethal: (ny, nx) bool; cost: (ny, nx) float in [0, 1), NaN where lethal. Read-only."""

    lethal: np.ndarray
    cost: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in (("lethal", bool), ("cost", np.float64)):
            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def traversability(
    emap: ElevationMap, features: TerrainFeatures, cfg: TraversabilityConfig
) -> Traversability:
    """Lethal mask and cost.

    lethal = unknown | any feature NaN (insufficient support) | slope >= slope_crit |
             roughness >= rough_crit | step >= step_crit, then dilated by a robot_radius_m disk
             (cells outside the map do not count as lethal; the planner stays on the map).
    cost   = w_slope * slope/slope_crit + w_rough * rough/rough_crit + w_step * step/step_crit,
             clipped to [0, 1); NaN where lethal.
    """
    slope_crit = math.radians(cfg.slope_crit_deg)
    slope, rough, step = features.slope_rad, features.roughness_m, features.step_m
    nan = np.isnan(slope) | np.isnan(rough) | np.isnan(step)
    with np.errstate(invalid="ignore"):  # NaN comparisons are False; NaN cells are lethal
        too_much = (slope >= slope_crit) | (rough >= cfg.rough_crit_m) | (step >= cfg.step_crit_m)
    core = (emap.source == UNKNOWN) | nan | too_much
    disk, _, _ = disk_offsets(math.floor(cfg.robot_radius_m / emap.grid.resolution + 1e-9))
    lethal = ndimage.binary_dilation(core, structure=disk)
    cost = (
        cfg.w_slope * slope / slope_crit
        + cfg.w_rough * rough / cfg.rough_crit_m
        + cfg.w_step * step / cfg.step_crit_m
    )
    cost = np.clip(cost, 0.0, np.nextafter(1.0, 0.0))
    return Traversability(lethal=lethal, cost=np.where(lethal, np.nan, cost))
