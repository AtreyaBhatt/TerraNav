"""Grid A* over a traversability map (8-connected, terrain-weighted edges).

Cells are (row, col) on the map grid (row = y, col = x). Edge cost between neighbouring cells
u and v:  step_len_m * (1 + alpha * (c_u + c_v) / 2), with step_len_m = res or res * sqrt(2)
and c in [0, 1) the traversability cost. Lethal cells are never entered, and a diagonal move
is not allowed when either orthogonal cell it passes between is lethal (no corner cutting).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)
# 8-connected moves (drow, dcol, length in cells).
MOVES = tuple(
    (dr, dc, SQRT2 if dr and dc else 1.0)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


@dataclass(frozen=True, slots=True, eq=False)
class PlanResult:
    """path_rc: (N, 2) int64 cells (or polyline vertices after smoothing), start to goal.
    cost: path cost under the edge-cost model (m-equivalent). expansions: nodes expanded.
    runtime_s: wall time of the planning call."""

    path_rc: np.ndarray
    cost: float
    expansions: int
    runtime_s: float

    def __post_init__(self) -> None:
        path = np.array(self.path_rc, dtype=np.int64, copy=True).reshape(-1, 2)
        path.setflags(write=False)
        object.__setattr__(self, "path_rc", path)


def octile_m(r0: int, c0: int, r1: int, c1: int, resolution_m: float) -> float:
    """Octile distance between cells, metres: the shortest 8-connected length on a free grid."""
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    return (max(dr, dc) - min(dr, dc) + SQRT2 * min(dr, dc)) * resolution_m


def astar(
    lethal: np.ndarray,
    cost: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    alpha: float,
    resolution_m: float,
) -> PlanResult | None:
    """Least-cost 8-connected path from start_rc to goal_rc, or None.

    lethal: (ny, nx) bool; cost: (ny, nx) float in [0, 1) (ignored where lethal); alpha >= 0.
    Heuristic: octile distance in metres, admissible and consistent because every edge costs at
    least its length (multiplier 1 + alpha * mean cost >= 1). Ties are broken by f, then h,
    then insertion order, so results are deterministic. Returns None if start or goal is
    outside the grid or lethal, or the goal is unreachable.
    """
    t0 = time.perf_counter()
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    lethal = np.asarray(lethal, dtype=bool)
    ny, nx = lethal.shape
    (sr, sc), (gr, gc) = (tuple(int(v) for v in start_rc), tuple(int(v) for v in goal_rc))
    for r, c in ((sr, sc), (gr, gc)):
        if not (0 <= r < ny and 0 <= c < nx) or lethal[r, c]:
            return None
    c_cell = np.where(lethal, 0.0, np.asarray(cost, dtype=np.float64))
    if np.isnan(c_cell).any():
        raise ValueError("cost must be finite on non-lethal cells")

    g = np.full((ny, nx), np.inf)
    parent = np.full((ny, nx), -1, dtype=np.int64)
    closed = np.zeros((ny, nx), dtype=bool)
    g[sr, sc] = 0.0
    h0 = octile_m(sr, sc, gr, gc, resolution_m)
    heap = [(h0, h0, 0, sr, sc)]
    counter = 1
    expansions = 0
    while heap:
        _, _, _, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue  # stale entry
        closed[r, c] = True
        expansions += 1
        if (r, c) == (gr, gc):
            break
        g_rc, c_rc = g[r, c], c_cell[r, c]
        for dr, dc, length in MOVES:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < ny and 0 <= cc < nx) or lethal[rr, cc] or closed[rr, cc]:
                continue
            if dr and dc and (lethal[r + dr, c] or lethal[r, c + dc]):
                continue  # no corner cutting
            step = length * resolution_m * (1.0 + alpha * (c_rc + c_cell[rr, cc]) / 2.0)
            new_g = g_rc + step
            if new_g < g[rr, cc]:
                g[rr, cc] = new_g
                parent[rr, cc] = r * nx + c
                h = octile_m(rr, cc, gr, gc, resolution_m)
                heapq.heappush(heap, (new_g + h, h, counter, rr, cc))
                counter += 1
    if not closed[gr, gc]:
        return None

    path = [(gr, gc)]
    idx = parent[gr, gc]
    while idx >= 0:
        path.append(divmod(int(idx), nx))
        idx = parent[path[-1]]
    path.reverse()
    return PlanResult(
        path_rc=np.array(path),
        cost=float(g[gr, gc]),
        expansions=expansions,
        runtime_s=time.perf_counter() - t0,
    )
