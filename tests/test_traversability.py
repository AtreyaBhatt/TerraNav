"""Traversability rules (hand-derived) and ground-truth traversability."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from map_helpers import MAP, cell_xy, emap_from

from terranav.perception.features import TerrainFeatures, terrain_features
from terranav.perception.traversability import (
    R_NO_ESTIMATE,
    R_ROUGH,
    R_SLOPE,
    R_STEP,
    R_UNKNOWN,
    TraversabilityConfig,
    traversability,
)
from terranav.sim.ground_truth import ground_truth_traversability
from terranav.types import Box, GridSpec, Scenario, World

X, Y = cell_xy()
CFG = TraversabilityConfig()
SHAPE = X.shape
TERRAIN = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))


def feats(slope_rad=0.0, rough_m=0.0, step_m=0.0, step_raw_m=None) -> TerrainFeatures:
    def full(v):
        return np.broadcast_to(np.asarray(v, dtype=float), SHAPE).copy()

    return TerrainFeatures(
        slope_rad=full(slope_rad),
        roughness_m=full(rough_m),
        step_m=full(step_m),
        step_raw_m=full(step_m if step_raw_m is None else step_raw_m),
        step_span_m=full(0.0),
        support=full(29),
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
    ("f", "reason"),
    [
        (feats(slope_rad=one_cell(math.radians(30.0))), R_SLOPE),
        (feats(rough_m=one_cell(0.05)), R_ROUGH),
        (feats(step_m=one_cell(0.20)), R_STEP),
        (feats(slope_rad=one_cell(np.nan)), R_NO_ESTIMATE),  # insufficient support
    ],
    ids=["slope", "rough", "step", "nan"],
)
def test_each_lethal_rule_dilated_by_robot_radius(f, reason):
    t = traversability(FLAT, f, CFG)
    assert DISK.sum() == 29
    np.testing.assert_array_equal(t.lethal, DISK)
    assert t.core_reason[60, 60] == reason and np.count_nonzero(t.core_reason) == 1
    assert np.all(np.isnan(t.cost[DISK])) and np.all(np.isfinite(t.cost[~DISK]))


def test_unknown_cell_is_lethal():
    known = np.ones(SHAPE, dtype=bool)
    known[60, 60] = False
    t = traversability(emap_from(np.zeros(SHAPE), known=known), feats(), CFG)
    np.testing.assert_array_equal(t.lethal, DISK)
    assert t.core_reason[60, 60] == R_UNKNOWN


def test_step_gate_for_lethal_on_far_range_step():
    # A 0.2 m step at x = 6 m (cols 59 | 60) seen at long range: every cell mean has
    # z_mean_var = 0.07^2, so the 3-sigma gate is 3 * sqrt(2) * 0.07 = 0.297 m > 0.2 m. The
    # gated step is 0; the raw jump is 0.2 m = step_crit, seen from cols 58..61.
    z = np.where(X >= 6.0, 0.2, 0.0)
    emap = emap_from(z, z_mean_var=0.07**2)
    f = terrain_features(emap)
    rows = slice(3, -3)
    assert np.all(f.step_m[rows, 59:61] == 0.0)
    np.testing.assert_allclose(f.step_raw_m[rows, 59:61], 0.2, atol=1e-12)
    gated = traversability(emap, f, CFG)
    raw = traversability(emap, f, replace(CFG, step_gate_for_lethal=False))
    assert not (gated.core_reason & R_STEP).any()
    assert np.all(raw.core_reason[rows, 58:62] & R_STEP)  # pairs up to 2 cells across the edge
    assert not (raw.core_reason[rows, :58] & R_STEP).any()
    assert np.all(raw.lethal[rows, 55:65])  # step cells dilated by 3 cells reach cols 55..64
    # The cost term always uses the gated step: identical wherever both are non-lethal.
    both = ~gated.lethal & ~raw.lethal
    np.testing.assert_array_equal(gated.cost[both], raw.cost[both])


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
    # rows 50..69. The 0.5 m jumps across the box outline make its edge cells (box cols 50
    # and 59, ground cols 49 and 60) step-lethal; dilated by 0.3 m (3 cells) they cover the
    # outline +- 3 cells. Far away the ground is flat: slope = roughness = step = 0 -> cost 0.
    box = Box((5.0, 5.0, 0.0), (6.0, 7.0, 0.5))
    world = World(np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (box,), "test", 0)
    sc = Scenario(world, (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    t = ground_truth_traversability(sc, MAP, CFG)
    for col in (49, 50, 59, 60):
        assert np.all(t.core_reason[52:68, col] & R_STEP)
    assert np.all(t.lethal[50:70, 47:54]) and np.all(t.lethal[50:70, 56:63])
    far = (np.abs(X - 5.5) > 1.5) | (np.abs(Y - 6.0) > 2.0)
    far &= (X > 0.5) & (X < 11.5) & (Y > 0.5) & (Y < 11.5)
    assert not t.lethal[far].any()
    np.testing.assert_allclose(t.cost[far], 0.0, atol=1e-9)


def ucb_step_map(height: float, sigma: float):
    """Flat ground with a step of `height` at x = 6 m (cols 59 | 60); every cell mean has
    standard deviation sigma / sqrt(2), so each pair jump has sigma_jump = sigma."""
    z = np.where(X >= 6.0, height, 0.0)
    emap = emap_from(z, z_mean_var=sigma**2 / 2)
    return emap, terrain_features(emap)


@pytest.mark.parametrize(
    ("height", "k", "lethal"),
    [
        (0.195, 0.0, False),  # point value 0.195 < 0.20
        (0.195, 2.0, True),  # 0.195 + 2 * 0.005 = 0.205 >= 0.20
        (0.150, 2.0, False),  # 0.150 + 0.010 = 0.160 < 0.20
    ],
)
def test_lethal_ucb_on_step(height, k, lethal):
    # sigma_jump = 0.005 m: the jump is far above the 3-sigma gate (0.015 m) in every case.
    emap, f = ucb_step_map(height, 0.005)
    t = traversability(emap, f, replace(CFG, lethal_ucb_k=k))
    rows = slice(3, -3)
    assert bool(np.all(t.core_reason[rows, 58:62] & R_STEP)) is lethal
    assert not (t.core_reason[rows, :58] & R_STEP).any()


def test_lethal_ucb_k0_unchanged():
    # k = 0 reproduces the default exactly (lethal mask, reasons and cost).
    emap, f = ucb_step_map(0.195, 0.005)
    a = traversability(emap, f, CFG)
    b = traversability(emap, f, replace(CFG, lethal_ucb_k=0.0))
    np.testing.assert_array_equal(a.lethal, b.lethal)
    np.testing.assert_array_equal(a.core_reason, b.core_reason)
    np.testing.assert_array_equal(a.cost, b.cost)


def test_ground_truth_terrace_edge_has_no_rough_band():
    # One-step 0.15 m mesa (4 x 4 m, yaw 0) on flat ground: GT step 0.15 m along its outline
    # (sub-critical), and with edge-aware roughness no rough-lethal (>= 0.05 m) cell around it.
    from terranav.sim.ground_truth import ground_truth_features
    from terranav.sim.world import add_terrace

    hf = add_terrace(
        np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (6.0, 6.0), (4.0, 4.0), 0.0, 0.15, 1, 2.0, 1
    )
    sc = Scenario(World(hf, TERRAIN, (), "test", 0), (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    f = ground_truth_features(sc, MAP)
    inner = (slice(5, -5), slice(5, -5))
    assert np.nanmax(f.step_m[inner]) == pytest.approx(0.15, abs=1e-9)
    assert np.nanmax(f.roughness_m[inner]) < CFG.rough_crit_m
    t = ground_truth_traversability(sc, MAP, CFG)
    assert not t.lethal[inner].any()
