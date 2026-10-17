"""Coverage measurement tests. Expected values are derived by hand on small grids."""

from __future__ import annotations

import numpy as np
import pytest

from terranav.sim.coverage import corridor_mask, coverage, observation_counts, reachable_mask
from terranav.types import Box, GridSpec, Scenario, World

TERRAIN = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
MAP = GridSpec(resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
SMALL = GridSpec(resolution=1.0, size_x=4.0, size_y=3.0, origin=(0.0, 0.0))  # (ny, nx) = (3, 4)


def scenario(boxes=(), start=(1.0, 6.0), goal=(11.0, 6.0)) -> Scenario:
    world = World(np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, boxes, "test", 0)
    return Scenario(world, start, goal, "test", 0, {})


def test_observation_counts_per_frame_not_per_point():
    # Frame 0: two points in cell (row 0, col 0), one in (2, 3). Frame 1: one in (0, 0) and
    # one outside the map (ignored). Counts are frames per cell, not points per cell.
    f0 = np.array([[0.2, 0.3, 0.0], [0.8, 0.9, 5.0], [3.5, 2.5, 0.0]])
    f1 = np.array([[0.5, 0.5, 0.0], [9.0, 1.0, 0.0]])
    counts = observation_counts([f0, f1, np.zeros((0, 3))], SMALL)
    expected = np.zeros((3, 4), dtype=int)
    expected[0, 0] = 2
    expected[2, 3] = 1
    np.testing.assert_array_equal(counts, expected)


def test_coverage_fraction():
    region = np.zeros((3, 4), dtype=bool)
    region[0, :] = True  # 4 cells; points hit (0, 0) and (0, 2) -> 2 / 4
    pts = np.array([[0.5, 0.5, 0.0], [2.5, 0.5, 0.0], [1.5, 2.5, 0.0]])  # last one off-region
    assert coverage([pts], SMALL, region) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        coverage([pts], SMALL, np.zeros((3, 4), dtype=bool))


def test_corridor_mask_uses_segment_distance():
    # Segment (1, 6) -> (11, 6), half-width 3: cells with centre |y - 6| <= 3 for x in [1, 11],
    # and within 3 m of an endpoint beyond it.
    m = corridor_mask(scenario(), MAP)
    r, c = MAP.xy_to_rc(
        np.array([6.05, 6.05, 6.05, 0.05, 11.95]), np.array([8.95, 9.05, 2.95, 6.05, 3.05])
    )
    assert m[r[0], c[0]]  # 2.95 m above the line
    assert not m[r[1], c[1]]  # 3.05 m above the line
    assert not m[r[2], c[2]]  # 3.05 m below the line
    assert m[r[3], c[3]]  # 0.95 m behind the start
    assert not m[r[4], c[4]]  # beyond the goal: distance to (11, 6) = hypot(0.95, 2.95) > 3


def test_reachable_mask_flat_and_with_box():
    assert reachable_mask(scenario(), MAP).all()
    # A box at x in [5, 6], y in [5, 7]; easy ground excludes it dilated by 0.3 m, so map cells
    # with centres inside x in (4.7, 6.3), y in (4.7, 7.3) are unreachable; others remain.
    box = Box((5.0, 5.0, 0.0), (6.0, 7.0, 1.0))
    m = reachable_mask(scenario((box,)), MAP)
    r, c = MAP.xy_to_rc(np.array([5.55, 4.55, 6.45, 5.55]), np.array([6.05, 6.05, 6.05, 7.45]))
    assert not m[r[0], c[0]]
    assert m[r[1], c[1]] and m[r[2], c[2]] and m[r[3], c[3]]
