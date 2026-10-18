"""A* planner tests: hand-derived costs and an independent Dijkstra reference."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from terranav.planning.astar import PlanResult, astar

RES = 0.1
SQ2 = math.sqrt(2.0)


def free(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((n_rows, n_cols), dtype=bool), np.zeros((n_rows, n_cols))


def edge_cost(c_u: float, c_v: float, length_m: float, alpha: float) -> float:
    return length_m * (1.0 + alpha * (c_u + c_v) / 2.0)


# --- free grid ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ((0, 5), 5 * RES),  # straight: 5 orthogonal steps
        ((5, 5), 5 * RES * SQ2),  # diagonal: 5 diagonal steps
        ((2, 7), 5 * RES + 2 * RES * SQ2),  # octile: 2 diagonal + 5 orthogonal
    ],
)
def test_free_grid_optimal_cost(goal, expected):
    lethal, cost = free(10, 10)
    r = astar(lethal, cost, (0, 0), goal, alpha=4.0, resolution_m=RES)
    assert isinstance(r, PlanResult)
    assert r.cost == pytest.approx(expected, abs=1e-12)
    assert tuple(r.path_rc[0]) == (0, 0) and tuple(r.path_rc[-1]) == goal
    steps = np.diff(r.path_rc, axis=0)
    assert np.all(np.abs(steps).max(axis=1) == 1)  # 8-connected moves only
    assert r.expansions > 0 and r.runtime_s >= 0.0
    assert r.path_rc.dtype.kind == "i" and r.path_rc.shape[1] == 2


# --- obstacles ------------------------------------------------------------------------------


def test_wall_with_single_gap():
    lethal, cost = free(10, 10)
    lethal[:, 5] = True
    lethal[7, 5] = False  # the only gap
    r = astar(lethal, cost, (2, 0), (2, 9), alpha=0.0, resolution_m=RES)
    cells = {tuple(p) for p in r.path_rc}
    assert (7, 5) in cells
    assert not lethal[r.path_rc[:, 0], r.path_rc[:, 1]].any()
    # Hand: the gap can only be crossed orthogonally, (7,4) -> (7,5) -> (7,6): entering it
    # diagonally would cut the corner of a lethal wall cell. (2,0) -> (7,4): 4 diagonal + 1
    # orthogonal; (7,4) -> (7,6): 2 orthogonal; (7,6) -> (2,9): 3 diagonal + 2 orthogonal.
    assert [7, 4] in r.path_rc.tolist() and [7, 6] in r.path_rc.tolist()
    assert r.cost == pytest.approx(RES * (7 * SQ2 + 5), abs=1e-12)


def band_grid() -> tuple[np.ndarray, np.ndarray]:
    """11 x 11 grid; row 5 has cost 0.5 except a zero-cost gap at column 0."""
    lethal, cost = free(11, 11)
    cost[5, 1:] = 0.5
    return lethal, cost


def test_band_alpha_zero_crosses():
    # alpha = 0: cost is pure length; straight down column 5 is 10 orthogonal steps = 1.0 m.
    lethal, cost = band_grid()
    r = astar(lethal, cost, (0, 5), (10, 5), alpha=0.0, resolution_m=RES)
    assert r.cost == pytest.approx(10 * RES, abs=1e-12)
    assert (5, 0) not in {tuple(p) for p in r.path_rc}


def test_band_large_alpha_detours_through_gap():
    # alpha = 10. Crossing the band at column 5: two edges touch the band cell, each costing
    # 0.1 * (1 + 10 * 0.25) = 0.35 instead of 0.1 -> 1.0 + 0.5 = 1.5. Through the gap (5, 0):
    # 5 diagonal steps there and 5 back, all at zero cost: 10 * 0.1 * sqrt(2) = 1.4142. Any other
    # band crossing is longer than 1.4142 already before its band surcharge.
    lethal, cost = band_grid()
    r = astar(lethal, cost, (0, 5), (10, 5), alpha=10.0, resolution_m=RES)
    assert (5, 0) in {tuple(p) for p in r.path_rc}
    assert r.cost == pytest.approx(10 * RES * SQ2, abs=1e-12)
    straight = 8 * RES + 2 * edge_cost(0.0, 0.5, RES, 10.0)
    assert straight == pytest.approx(1.5) and r.cost < straight


def test_no_corner_cutting():
    # (0, 1) lethal: the diagonal (0,0) -> (1,1) would brush its corner, so it is not allowed;
    # the path goes through (1, 0): 2 orthogonal steps = 0.2 instead of 0.1414.
    lethal, cost = free(3, 3)
    lethal[0, 1] = True
    r = astar(lethal, cost, (0, 0), (1, 1), alpha=0.0, resolution_m=RES)
    assert [tuple(p) for p in r.path_rc] == [(0, 0), (1, 0), (1, 1)]
    assert r.cost == pytest.approx(2 * RES, abs=1e-12)
    # Both orthogonal neighbours lethal: the start is sealed in, no path.
    lethal[1, 0] = True
    assert astar(lethal, cost, (0, 0), (1, 1), alpha=0.0, resolution_m=RES) is None


def test_unreachable_and_lethal_endpoints():
    lethal, cost = free(6, 6)
    lethal[:, 3] = True
    assert astar(lethal, cost, (0, 0), (0, 5), alpha=1.0, resolution_m=RES) is None
    lethal, cost = free(6, 6)
    lethal[0, 0] = True
    assert astar(lethal, cost, (0, 0), (5, 5), alpha=1.0, resolution_m=RES) is None
    assert astar(lethal, cost, (5, 5), (0, 0), alpha=1.0, resolution_m=RES) is None


def test_deterministic():
    rng = np.random.default_rng(0)
    lethal = rng.random((30, 30)) < 0.2
    cost = np.where(lethal, np.nan, rng.random((30, 30)))
    lethal[0, 0] = lethal[29, 29] = False
    cost[0, 0] = cost[29, 29] = 0.0
    a = astar(lethal, cost, (0, 0), (29, 29), alpha=3.0, resolution_m=RES)
    b = astar(lethal, cost, (0, 0), (29, 29), alpha=3.0, resolution_m=RES)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a.path_rc, b.path_rc)
    assert a.cost == b.cost and a.expansions == b.expansions


# --- optimality against Dijkstra -------------------------------------------------------------

MOVES = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def reference_cost(lethal, cost, start, goal, alpha) -> float:
    """Dijkstra (scipy) on the same graph: 8-connected, no corner cutting, same edge costs."""
    ny, nx = lethal.shape
    rows, cols, w = [], [], []
    for r in range(ny):
        for c in range(nx):
            if lethal[r, c]:
                continue
            for dr, dc in MOVES:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < ny and 0 <= cc < nx) or lethal[rr, cc]:
                    continue
                if dr and dc and (lethal[r + dr, c] or lethal[r, c + dc]):
                    continue
                length = RES * (SQ2 if dr and dc else 1.0)
                rows.append(r * nx + c)
                cols.append(rr * nx + cc)
                # +1e-300 keeps zero-cost edges from being dropped as "no edge" by csr_matrix
                w.append(edge_cost(cost[r, c], cost[rr, cc], length, alpha) + 1e-300)
    graph = csr_matrix((w, (rows, cols)), shape=(ny * nx, ny * nx))
    d = dijkstra(graph, indices=start[0] * nx + start[1])
    return float(d[goal[0] * nx + goal[1]])


@settings(deadline=None, max_examples=60)
@given(
    seed=st.integers(0, 2**31 - 1),
    p_lethal=st.floats(0.0, 0.4),
    alpha=st.floats(0.0, 8.0),
)
def test_astar_matches_dijkstra(seed, p_lethal, alpha):
    rng = np.random.default_rng(seed)
    lethal = rng.random((12, 12)) < p_lethal
    cost = np.where(lethal, np.nan, rng.random((12, 12)) * 0.999)
    free_cells = np.argwhere(~lethal)
    if len(free_cells) < 2:
        return
    s, g = (
        tuple(int(v) for v in free_cells[i]) for i in rng.choice(len(free_cells), 2, replace=False)
    )
    ref = reference_cost(lethal, cost, s, g, alpha)
    r = astar(lethal, cost, s, g, alpha=alpha, resolution_m=RES)
    if math.isinf(ref):
        assert r is None
    else:
        assert r is not None and abs(r.cost - ref) < 1e-9
