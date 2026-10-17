"""Scan evaluation: a parallel run must reproduce the serial run exactly."""

from __future__ import annotations

from terranav.sim.scan_eval import run_jobs
from terranav.types import SensorConfig

# Low resolution keeps the captures cheap; the property under test is independence of the
# job distribution, not coverage values.
SENSOR = SensorConfig(width=48, height=36)
JOBS = [("mixed", 0, "flank"), ("steps", 1, "line")]


def test_parallel_matches_serial():
    serial = run_jobs(JOBS, workers=1, sensor=SENSOR)
    parallel = run_jobs(JOBS, workers=2, sensor=SENSOR)
    assert serial == parallel  # exact float equality, job order preserved
    assert [(r["family"], r["seed"], r["strategy"]) for r in serial] == JOBS
    for r in serial:
        assert 0.0 < r["corridor"] <= 1.0 and 0.0 < r["reachable"] <= 1.0


def test_map_eval_job_smoke():
    from terranav.sim.map_eval import evaluate_map_job

    r = evaluate_map_job(("mixed", 0, "flank+start"), sensor=SENSOR)
    assert 0.0 < r["corridor_raw"] <= r["corridor_filled"] <= 1.0
    assert 0.0 < r["reachable_raw"] <= r["reachable_filled"] <= 1.0
    assert len(r["err_observed"]) > 0
    assert 0.0 <= r["goal_known_fraction"] <= 1.0
    assert isinstance(r["connected"], bool)


def test_known_connected_hand_built():
    # 1 m map, 3 x 5 cells; start (0.5, 1.5) -> cell (1, 0), goal (4.5, 1.5) -> cell (1, 4).
    #   row 2:  . . . . .        K = known, . = unknown
    #   row 1:  K K . K K   <- a gap at (1, 2) separates start from goal
    #   row 0:  K . K . K   <- diagonal contacts only: not 4-connected
    import numpy as np

    from terranav.perception.elevation_map import FILLED, FOOTPRINT, OBSERVED, ElevationMap
    from terranav.sim.map_eval import known_connected
    from terranav.types import GridSpec

    grid = GridSpec(resolution=1.0, size_x=5.0, size_y=3.0, origin=(0.0, 0.0))
    src = np.zeros((3, 5), dtype=np.uint8)
    src[1, [0, 1, 3, 4]] = [FOOTPRINT, OBSERVED, FILLED, OBSERVED]
    src[0, [0, 2, 4]] = OBSERVED
    nan = np.full((3, 5), np.nan)
    count = (src == OBSERVED).astype(np.int32)

    def emap(source):
        return ElevationMap(grid, count, nan, nan, nan, nan, nan, nan, source)

    assert not known_connected(emap(src), (0.5, 1.5), (4.5, 1.5))
    src[1, 2] = FILLED  # close the gap
    assert known_connected(emap(src), (0.5, 1.5), (4.5, 1.5))
    assert not known_connected(emap(src), (0.5, 1.5), (2.5, 2.5))  # goal cell unknown


def test_trav_eval_job_smoke():
    from terranav.sim.trav_eval import evaluate_trav_job

    r = evaluate_trav_job(("mixed", 0, "flank+start"), sensor=SENSOR)
    assert 0 <= r["false_free"] <= r["gt_lethal"] and 0 <= r["false_lethal"] <= r["gt_free"]
    assert r["gt_lethal"] + r["gt_free"] > 0 and r["cost_n"] <= r["gt_free"]
    assert isinstance(r["connected_perceived"], bool) and isinstance(r["connected_gt"], bool)
    assert r["connected_gt"]  # GT must allow a route: scenarios guarantee easy reachability
