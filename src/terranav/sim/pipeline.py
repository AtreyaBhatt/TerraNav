"""End-to-end pipeline for one scenario: survey -> perceived and GT traversability -> plans.

Builds the inputs each planner in terranav.planning.planners.PLANNERS needs ("perceived" or
"ground_truth" traversability) and runs them. Planning itself never touches the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from terranav.config import SimConfig
from terranav.perception.traversability import Traversability, TraversabilityConfig
from terranav.planning.astar import PlanResult
from terranav.planning.planners import DEFAULT_ALPHA, PLANNERS, plan
from terranav.sim.scan_eval import Job
from terranav.sim.trav_eval import ground_truth, perceive
from terranav.types import Scenario, SensorConfig


@dataclass(frozen=True)
class PipelineRun:
    scenario: Scenario
    trav: dict[str, Traversability]  # "perceived", "ground_truth"
    plans: dict[str, PlanResult | None]  # by planner name


def run_planners(
    job: Job,
    alpha: float = DEFAULT_ALPHA,
    tcfg: TraversabilityConfig | None = None,
    sensor: SensorConfig | None = None,
) -> PipelineRun:
    """Survey and perceive the scenario of `job`, build GT traversability, run every planner."""
    cfg = SimConfig()
    sensor = sensor if sensor is not None else cfg.sensor
    tcfg = tcfg if tcfg is not None else TraversabilityConfig()
    sc, _, _, _, (perceived,) = perceive(job, sensor, (tcfg,))
    _, gt = ground_truth(sc, cfg.map_grid, tcfg)
    trav = {"perceived": perceived, "ground_truth": gt}
    plans = {
        name: plan(trav[spec.source], sc.start_xy, sc.goal_xy, cfg.map_grid, alpha, spec.graded)
        for name, spec in PLANNERS.items()
    }
    return PipelineRun(scenario=sc, trav=trav, plans=plans)
