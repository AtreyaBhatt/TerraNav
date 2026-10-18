"""Planner front-end: traversability map -> smoothed path.

The three compared planners differ only in their inputs (see PLANNERS); building the perceived
and ground-truth traversability maps is the simulation pipeline's job (terranav.sim), not
planning's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from terranav.perception.traversability import Traversability
from terranav.planning.astar import PlanResult, astar
from terranav.planning.smooth import polyline_cost, shortcut
from terranav.types import GridSpec

DEFAULT_ALPHA = 4.0  # ILLUSTRATIVE terrain-cost weight in the edge cost; swept in M4


@dataclass(frozen=True)
class PlannerSpec:
    """source: which traversability map ("perceived" or "ground_truth"); graded: use the cost."""

    source: str
    graded: bool


PLANNERS = {
    "terranav": PlannerSpec(source="perceived", graded=True),
    "baseline_occ": PlannerSpec(source="perceived", graded=False),  # occupancy only
    "oracle": PlannerSpec(source="ground_truth", graded=True),
}


def plan(
    trav: Traversability,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    grid: GridSpec,
    alpha: float,
    graded: bool,
) -> PlanResult | None:
    """A* on `trav` (graded: cost weighted by alpha; ungraded: every non-lethal cell costs 0,
    i.e. shortest collision-free path, alpha ignored), then cost-preserving shortcutting.

    Returns the smoothed polyline (vertices as cells), its cost under the planning cost model,
    A* expansions and the total runtime (A* + smoothing), or None if no path exists.
    """
    t0 = time.perf_counter()
    lethal = trav.lethal
    cost = np.where(lethal, np.nan, trav.cost if graded else 0.0)
    a = alpha if graded else 0.0
    start = tuple(int(v) for v in grid.xy_to_rc(*start_xy))
    goal = tuple(int(v) for v in grid.xy_to_rc(*goal_xy))
    raw = astar(lethal, cost, start, goal, a, grid.resolution)
    if raw is None:
        return None
    waypoints = shortcut(raw.path_rc, lethal, cost, a, grid.resolution)
    return PlanResult(
        path_rc=waypoints,
        cost=polyline_cost(waypoints, np.where(lethal, 0.0, cost), a, grid.resolution),
        expansions=raw.expansions,
        runtime_s=time.perf_counter() - t0,
    )
