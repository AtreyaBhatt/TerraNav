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
