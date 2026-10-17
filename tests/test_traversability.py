"""Traversability rules (hand-derived) and ground-truth traversability."""

from __future__ import annotations

import math

import numpy as np
import pytest
from map_helpers import MAP, cell_xy, emap_from

from terranav.perception.features import TerrainFeatures
from terranav.perception.traversability import TraversabilityConfig, traversability
from terranav.sim.ground_truth import ground_truth_traversability
from terranav.types import Box, GridSpec, Scenario, World

X, Y = cell_xy()
CFG = TraversabilityConfig()
SHAPE = X.shape
TERRAIN = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))


def feats(slope_rad=0.0, rough_m=0.0, step_m=0.0) -> TerrainFeatures:
    def full(v):
        return np.broadcast_to(np.asarray(v, dtype=float), SHAPE).copy()

    return TerrainFeatures(
        slope_rad=full(slope_rad), roughness_m=full(rough_m), step_m=full(step_m), support=full(29)
    )


FLAT = emap_from(np.zeros(SHAPE))
# Cells within robot_radius = 0.3 m (3 cells) of (row 60, col 60): 29 cells (disk radius 3).
DISK = (X - 6.05) ** 2 + (Y - 6.05) ** 2 <= 0.3**2 + 1e-9


def one_cell(value: float) -> np.ndarray:
    a = np.zeros(SHAPE)
    a[60, 60] = value
    return a


def test_config_values():
    assert CFG.slope_crit_deg == 30.0 and CFG.rough_crit_m == 0.05 and CFG.step_crit_m == 0.20
    assert (CFG.w_slope, CFG.w_rough, CFG.w_step) == (0.4, 0.3, 0.3)
    assert CFG.robot_radius_m == 0.3


@pytest.mark.parametrize(
    "f",
    [
        feats(slope_rad=one_cell(math.radians(30.0))),
        feats(rough_m=one_cell(0.05)),
        feats(step_m=one_cell(0.20)),
        feats(slope_rad=one_cell(np.nan)),  # insufficient support
    ],
    ids=["slope", "rough", "step", "nan"],
)
def test_each_lethal_rule_dilated_by_robot_radius(f):
    t = traversability(FLAT, f, CFG)
    assert DISK.sum() == 29
    np.testing.assert_array_equal(t.lethal, DISK)
    assert np.all(np.isnan(t.cost[DISK])) and np.all(np.isfinite(t.cost[~DISK]))


def test_unknown_cell_is_lethal():
    known = np.ones(SHAPE, dtype=bool)
    known[60, 60] = False
    t = traversability(emap_from(np.zeros(SHAPE), known=known), feats(), CFG)
    np.testing.assert_array_equal(t.lethal, DISK)


def test_just_below_critical_is_not_lethal():
    f = feats(math.radians(29.999), 0.04999, 0.19999)
    t = traversability(FLAT, f, CFG)
    assert not t.lethal.any()
    assert np.all((t.cost >= 0.0) & (t.cost < 1.0))


@pytest.mark.parametrize("which", ["slope", "rough", "step"])
def test_cost_monotonic_in_each_feature(which):
    crit = {"slope": math.radians(30.0), "rough": 0.05, "step": 0.20}[which]
    ramp = np.broadcast_to(np.linspace(0.0, 0.99 * crit, SHAPE[1]), SHAPE)
    kw = {"slope": "slope_rad", "rough": "rough_m", "step": "step_m"}[which]
    t = traversability(FLAT, feats(**{kw: ramp}), CFG)
    row = t.cost[60]
    assert np.all(np.diff(row) > 0)
    weight = {"slope": 0.4, "rough": 0.3, "step": 0.3}[which]
    assert row[-1] == pytest.approx(weight * 0.99)  # cost = w * feature / crit


def test_ground_truth_traversability_box():
    # Flat ground with a 0.5 m box over x in [5, 6], y in [5, 7]: map cells cols 50..59,
    # rows 50..69. The 0.5 m edge steps are lethal (>= 0.2 m) for cells within r_step of an
    # edge pair, then dilated by 0.3 m: the whole 1 m-wide box is lethal. Far away the ground
    # is flat: slope = roughness = step = 0 -> cost 0.
    box = Box((5.0, 5.0, 0.0), (6.0, 7.0, 0.5))
    world = World(np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (box,), "test", 0)
    sc = Scenario(world, (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    t = ground_truth_traversability(sc, MAP, CFG)
    assert np.all(t.lethal[50:70, 50:60])
    far = (np.abs(X - 5.5) > 1.5) | (np.abs(Y - 6.0) > 2.0)
    far &= (X > 0.5) & (X < 11.5) & (Y > 0.5) & (Y < 11.5)
    assert not t.lethal[far].any()
    np.testing.assert_allclose(t.cost[far], 0.0, atol=1e-9)
