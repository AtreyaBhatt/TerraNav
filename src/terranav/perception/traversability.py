"""Traversability: lethal mask and [0, 1) cost from terrain features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from terranav.perception.elevation_map import UNKNOWN, ElevationMap
from terranav.perception.features import STEP_SIGMA_GATE, TerrainFeatures, pair_jumps, pair_offsets
from terranav.perception.plane_fit import disk_offsets
from terranav.types import GridSpec

# core_reason bits (uint8): why a cell is lethal BEFORE robot-radius dilation.
R_UNKNOWN = 1  # map cell unknown
R_NO_ESTIMATE = 2  # known, but a feature is NaN (insufficient plane-fit support)
R_SLOPE = 4  # slope >= slope_crit
R_ROUGH = 8  # roughness >= rough_crit
R_STEP = 16  # step >= step_crit (gated or raw, see step_gate_for_lethal)
R_FEATURE = R_SLOPE | R_ROUGH | R_STEP


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
    # True: a step is lethal only if the jump is >= step_crit AND 3-sigma significant.
    # False: any raw jump >= step_crit is lethal. The cost term always uses the gated step.
    step_gate_for_lethal: bool = True
    # k > 0: a step is lethal when jump + k * sigma_jump >= step_crit, with sigma_jump =
    # sqrt(z_mean_var_i + z_mean_var_j) (the same sigma as the significance gate). 0 = point value.
    lethal_ucb_k: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class Traversability:
    """Read-only (ny, nx) arrays.

    lethal: bool, core lethal cells dilated by the robot radius.
    core_reason: uint8 bitmask of R_* reasons per cell before dilation (0 = not core lethal).
    cost: float in [0, 1), NaN where lethal.
    """

    lethal: np.ndarray
    core_reason: np.ndarray
    cost: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in (("lethal", bool), ("core_reason", np.uint8), ("cost", np.float64)):
            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def dilate_lethal(core: np.ndarray, grid: GridSpec, robot_radius_m: float) -> np.ndarray:
    """Grow core-lethal cells by a robot_radius_m disk (cells outside the map do not count)."""
    disk, _, _ = disk_offsets(math.floor(robot_radius_m / grid.resolution + 1e-9))
    return ndimage.binary_dilation(core, structure=disk)


def traversability(
    emap: ElevationMap, features: TerrainFeatures, cfg: TraversabilityConfig
) -> Traversability:
    """Lethal mask, core reasons and cost.

    core lethal = unknown | any feature NaN (insufficient support) | slope >= slope_crit |
                  roughness >= rough_crit | step >= step_crit, where step is the 3-sigma gated
                  step_m (step_gate_for_lethal) or the raw step_raw_m; with lethal_ucb_k = k > 0
                  the per-pair bound jump + k * sigma_jump replaces the jump (pair_jumps);
    lethal      = core lethal dilated by a robot_radius_m disk (dilate_lethal);
    cost        = w_slope*slope/slope_crit + w_rough*rough/rough_crit + w_step*step_m/step_crit
                  (gated step), clipped to [0, 1); NaN where lethal.
    """
    slope_crit = math.radians(cfg.slope_crit_deg)
    slope, rough, step = features.slope_rad, features.roughness_m, features.step_m
    step_lethal = step if cfg.step_gate_for_lethal else features.step_raw_m
    if cfg.lethal_ucb_k > 0:
        known = emap.source != UNKNOWN
        bound = pair_jumps(
            np.where(known, emap.z_mean, np.nan),
            np.where(known, emap.z_mean_var, np.nan),
            pair_offsets(emap.grid.resolution),
            gate=STEP_SIGMA_GATE if cfg.step_gate_for_lethal else 0.0,
            ucb_k=cfg.lethal_ucb_k,
        )
        step_lethal = np.where(np.isnan(step_lethal), np.nan, bound)
    reason = np.zeros(slope.shape, dtype=np.uint8)
    unknown = emap.source == UNKNOWN
    reason[unknown] |= R_UNKNOWN
    no_estimate = ~unknown & (np.isnan(slope) | np.isnan(rough) | np.isnan(step))
    reason[no_estimate] |= R_NO_ESTIMATE
    with np.errstate(invalid="ignore"):  # NaN comparisons are False; NaN cells are core anyway
        reason[slope >= slope_crit] |= R_SLOPE
        reason[rough >= cfg.rough_crit_m] |= R_ROUGH
        reason[step_lethal >= cfg.step_crit_m] |= R_STEP
    lethal = dilate_lethal(reason != 0, emap.grid, cfg.robot_radius_m)
    cost = (
        cfg.w_slope * slope / slope_crit
        + cfg.w_rough * rough / cfg.rough_crit_m
        + cfg.w_step * step / cfg.step_crit_m
    )
    cost = np.clip(cost, 0.0, np.nextafter(1.0, 0.0))
    return Traversability(lethal=lethal, core_reason=reason, cost=np.where(lethal, np.nan, cost))
