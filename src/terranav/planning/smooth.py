"""Cost-preserving shortcutting of grid paths.

Plain line-of-sight smoothing (replace any sub-path by a straight segment that avoids lethal
cells) would undo terrain-aware routing: a detour around a costly-but-passable band is still
"in sight" of a straight line through the band, so the smoother would cut right back through
it. Here a shortcut is accepted only if, under the SAME cost model the planner optimised, it
costs no more than the sub-path it replaces.

Segment model: a segment runs from cell centre to cell centre; its cost is the line integral of
(1 + alpha * c) along it, with c constant per cell (length inside each cell from an exact grid
traversal). For a single orthogonal or diagonal grid move this equals the planner's edge cost
step_len * (1 + alpha * (c_u + c_v) / 2), so A* costs and smoothed costs are comparable.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np


def segment_cells(a_rc, b_rc) -> list[tuple[int, int, float]]:
    """Cells crossed by the segment between the centres of cells a and b (supercover).

    Returns (row, col, length_in_cells) in traversal order. Where the segment passes exactly
    through a cell corner, both side cells are included with length 0 (they are touched, so
    they count for the lethal check, but carry no cost). Exact integer arithmetic for the
    crossing order: the k-th row boundary is crossed at t = (2k + 1) / (2 |drow|).
    """
    r, c = int(a_rc[0]), int(a_rc[1])
    r1, c1 = int(b_rc[0]), int(b_rc[1])
    dr, dc = r1 - r, c1 - c
    if dr == 0 and dc == 0:
        return [(r, c, 0.0)]
    length = math.hypot(dr, dc)
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    ar, ac = abs(dr), abs(dc)
    kr = kc = 0  # boundaries crossed so far in each axis
    t = 0.0
    out = []
    while True:
        # next crossing parameters as fractions (2k + 1) / (2 * |d|); compare exactly
        tr_num, tr_den = (2 * kr + 1, 2 * ar) if ar and kr < ar else (1, 0)
        tc_num, tc_den = (2 * kc + 1, 2 * ac) if ac and kc < ac else (1, 0)
        if tr_den == 0 and tc_den == 0:
            out.append((r, c, (1.0 - t) * length))
            return out
        if tc_den == 0 or (tr_den and tr_num * tc_den < tc_num * tr_den):
            t_next, move = tr_num / tr_den, "r"
        elif tr_den == 0 or tc_num * tr_den < tr_num * tc_den:
            t_next, move = tc_num / tc_den, "c"
        else:
            t_next, move = tr_num / tr_den, "rc"  # exact corner
        out.append((r, c, (t_next - t) * length))
        if move == "rc":
            out += [(r + sr, c, 0.0), (r, c + sc, 0.0)]
            r, c, kr, kc = r + sr, c + sc, kr + 1, kc + 1
        elif move == "r":
            r, kr = r + sr, kr + 1
        else:
            c, kc = c + sc, kc + 1
        t = t_next


def segment_cost(a_rc, b_rc, cost: np.ndarray, alpha: float, resolution_m: float) -> float:
    """Line integral of (1 + alpha * c) from centre a to centre b, metres."""
    return (
        sum(n * (1.0 + alpha * cost[r, c]) for r, c, n in segment_cells(a_rc, b_rc) if n > 0)
        * resolution_m
    )


def polyline_cost(
    waypoints: np.ndarray, cost: np.ndarray, alpha: float, resolution_m: float
) -> float:
    """Sum of segment_cost over consecutive waypoints (cells), metres."""
    return float(sum(segment_cost(a, b, cost, alpha, resolution_m) for a, b in pairwise(waypoints)))


def shortcut(
    path_rc: np.ndarray, lethal: np.ndarray, cost: np.ndarray, alpha: float, resolution_m: float
) -> np.ndarray:
    """Greedy cost-preserving shortcutting of a cell path; returns polyline vertices (M, 2).

    From the current vertex i, the segment to path cell j is extended (j = i+2, i+3, ...) while
    it touches no lethal cell (supercover, corners included) and costs no more than the path from
    i to j; at the first failure the last accepted j becomes the next vertex. Each accepted
    segment costs <= the sub-path it replaces, so the result never costs more than the input.
    """
    path = np.asarray(path_rc, dtype=np.int64)
    if len(path) <= 2:
        return path.copy()
    c_cell = np.where(lethal, 0.0, cost)
    # cumulative cost along the input path (each step is one grid move)
    steps = [segment_cost(a, b, c_cell, alpha, resolution_m) for a, b in pairwise(path)]
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    out = [0]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        for j in range(i + 2, len(path)):
            cells = segment_cells(path[i], path[j])
            if any(lethal[r, c] for r, c, _ in cells):
                break
            seg = sum(n * (1.0 + alpha * c_cell[r, c]) for r, c, n in cells if n > 0) * resolution_m
            if seg > cum[j] - cum[i] + 1e-12:
                break
            best = j
        out.append(best)
        i = best
    return path[out].copy()
