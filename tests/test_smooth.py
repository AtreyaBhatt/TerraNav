"""Cost-preserving shortcutting tests (checked independently by dense sampling)."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from terranav.planning.astar import astar
from terranav.planning.smooth import polyline_cost, shortcut

RES = 0.1


def sampled_cells(waypoints: np.ndarray, step: float = 0.002) -> set[tuple[int, int]]:
    """Cells visited by the polyline between cell centres, by dense sampling (cell units).

    Independent of the module's exact traversal; it can miss a cell touched only at a corner,
    so it is used as a necessary check (the module's own check is stricter, see smooth.py).
    """
    cells = set()
    for (r0, c0), (r1, c1) in pairwise(waypoints):
        n = max(2, int(math.hypot(r1 - r0, c1 - c0) / step))
        t = np.linspace(0.0, 1.0, n)
        rr = np.floor(r0 + 0.5 + t * (r1 - r0)).astype(int)
        cc = np.floor(c0 + 0.5 + t * (c1 - c0)).astype(int)
        cells |= set(zip(rr.tolist(), cc.tolist(), strict=True))
    return cells


def sampled_cost(waypoints, cost, alpha, step: float = 0.0005) -> float:
    """Line integral of (1 + alpha * c) over the polyline, by midpoint sampling (metres)."""
    total = 0.0
    for (r0, c0), (r1, c1) in pairwise(waypoints):
        length = math.hypot(r1 - r0, c1 - c0)
        n = max(1, int(length / step))
        t = (np.arange(n) + 0.5) / n
        rr = np.floor(r0 + 0.5 + t * (r1 - r0)).astype(int)
        cc = np.floor(c0 + 0.5 + t * (c1 - c0)).astype(int)
        total += float(np.sum(1.0 + alpha * cost[rr, cc])) * (length / n) * RES
    return total


def test_free_grid_collapses_to_one_segment():
    lethal, cost = np.zeros((10, 10), dtype=bool), np.zeros((10, 10))
    r = astar(lethal, cost, (0, 0), (3, 7), alpha=4.0, resolution_m=RES)
    w = shortcut(r.path_rc, lethal, cost, alpha=4.0, resolution_m=RES)
    assert [tuple(p) for p in w] == [(0, 0), (3, 7)]
    assert polyline_cost(w, cost, 4.0, RES) == pytest.approx(RES * math.hypot(3, 7), abs=1e-12)


def test_shortcut_respects_costly_band():
    # Row 5 costs 0.5 except a zero-cost gap at column 0; with alpha = 10 the A* path detours
    # through the gap. Shortcutting must keep the detour: no smoothed segment may cross a band
    # cell (cutting through costs more than the sub-path it would replace).
    lethal, cost = np.zeros((11, 11), dtype=bool), np.zeros((11, 11))
    cost[5, 1:] = 0.5
    r = astar(lethal, cost, (0, 5), (10, 5), alpha=10.0, resolution_m=RES)
    w = shortcut(r.path_rc, lethal, cost, alpha=10.0, resolution_m=RES)
    band = {(5, c) for c in range(1, 11)}
    assert not (sampled_cells(w) & band)
    assert polyline_cost(w, cost, 10.0, RES) <= r.cost + 1e-12
    # Here the detour is two straight legs: start -> gap -> goal.
    assert [tuple(p) for p in w] == [(0, 5), (5, 0), (10, 5)]


def test_polyline_cost_matches_line_integral():
    rng = np.random.default_rng(1)
    cost = rng.random((20, 20))
    w = np.array([[0, 0], [7, 13], [19, 2], [4, 18]])
    assert polyline_cost(w, cost, 3.0, RES) == pytest.approx(sampled_cost(w, cost, 3.0), rel=2e-3)


@settings(deadline=None, max_examples=60)
@given(seed=st.integers(0, 2**31 - 1), p_lethal=st.floats(0.0, 0.35), alpha=st.floats(0.0, 8.0))
def test_shortcut_never_crosses_lethal_nor_increases_cost(seed, p_lethal, alpha):
    rng = np.random.default_rng(seed)
    lethal = rng.random((20, 20)) < p_lethal
    cost = np.where(lethal, np.nan, rng.random((20, 20)) * 0.999)
    lethal[0, 0] = lethal[19, 19] = False
    cost[0, 0] = cost[19, 19] = 0.0
    r = astar(lethal, cost, (0, 0), (19, 19), alpha=alpha, resolution_m=RES)
    if r is None:
        return
    w = shortcut(r.path_rc, lethal, cost, alpha=alpha, resolution_m=RES)
    assert tuple(w[0]) == (0, 0) and tuple(w[-1]) == (19, 19)
    assert not any(lethal[c] for c in sampled_cells(w))
    smoothed = polyline_cost(w, cost, alpha, RES)
    assert smoothed <= r.cost + 1e-9
    assert smoothed == pytest.approx(sampled_cost(w, np.nan_to_num(cost), alpha), rel=5e-3)
