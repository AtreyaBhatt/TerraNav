"""Planner wiring: graded (TerraNav / Oracle) vs ungraded (Baseline-Occ)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terranav.perception.traversability import Traversability
from terranav.planning.planners import DEFAULT_ALPHA, PLANNERS, plan
from terranav.types import GridSpec

GRID = GridSpec(resolution=0.1, size_x=1.1, size_y=1.1, origin=(0.0, 0.0))  # 11 x 11 cells


def band_trav() -> Traversability:
    """Row 5 costs 0.5 except a zero-cost gap at column 0; nothing lethal."""
    cost = np.zeros((11, 11))
    cost[5, 1:] = 0.5
    return Traversability(
        lethal=np.zeros((11, 11), dtype=bool), core_reason=np.zeros((11, 11), np.uint8), cost=cost
    )


def xy(r: int, c: int) -> tuple[float, float]:
    return ((c + 0.5) * 0.1, (r + 0.5) * 0.1)


def test_graded_detours_and_ungraded_goes_straight():
    t = band_trav()
    graded = plan(t, xy(0, 5), xy(10, 5), GRID, alpha=10.0, graded=True)
    ungraded = plan(t, xy(0, 5), xy(10, 5), GRID, alpha=10.0, graded=False)
    # Graded: the detour through the gap, two straight legs of 5 * sqrt(2) cells each.
    assert [tuple(p) for p in graded.path_rc] == [(0, 5), (5, 0), (10, 5)]
    assert graded.cost == pytest.approx(10 * 0.1 * math.sqrt(2), abs=1e-12)
    # Ungraded: cost is ignored (alpha has no effect), the straight 1.0 m line wins.
    assert [tuple(p) for p in ungraded.path_rc] == [(0, 5), (10, 5)]
    assert ungraded.cost == pytest.approx(1.0, abs=1e-12)


def test_planner_registry():
    assert DEFAULT_ALPHA == 4.0
    assert PLANNERS["terranav"].source == "perceived" and PLANNERS["terranav"].graded
    assert PLANNERS["baseline_occ"].source == "perceived" and not PLANNERS["baseline_occ"].graded
    assert PLANNERS["oracle"].source == "ground_truth" and PLANNERS["oracle"].graded


def test_lethal_start_returns_none():
    t = band_trav()
    lethal = np.zeros((11, 11), dtype=bool)
    lethal[0, 5] = True
    t = Traversability(
        lethal=lethal, core_reason=t.core_reason, cost=np.where(lethal, np.nan, t.cost)
    )
    assert plan(t, xy(0, 5), xy(10, 5), GRID, alpha=1.0, graded=True) is None
