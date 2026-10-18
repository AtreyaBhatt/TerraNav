"""Planning evaluation: paired statistics (hand-checked) and a small end-to-end smoke test."""

from __future__ import annotations

import numpy as np
import pytest
from terranav.sim.plan_eval import compare_counts, paired_bootstrap_ci


def test_bootstrap_ci_of_constant_differences_is_degenerate():
    d = np.full(12, 0.3)
    mean, lo, hi = paired_bootstrap_ci(d, n_resamples=2000, seed=0)
    assert mean == pytest.approx(0.3) and lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    d = np.random.default_rng(1).normal(0.5, 1.0, 40)
    a = paired_bootstrap_ci(d, n_resamples=2000, seed=7)
    b = paired_bootstrap_ci(d, n_resamples=2000, seed=7)
    assert a == b
    mean, lo, hi = a
    assert lo < mean < hi and mean == pytest.approx(d.mean())


def test_bootstrap_ci_empty_is_nan():
    assert all(np.isnan(v) for v in paired_bootstrap_ci(np.array([]), 2000, 0))


def test_compare_counts_lower_is_better():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 2.0, 1.0, 4.0 + 1e-12])
    # better: a < b (1 < 2); equal within 1e-9: (2, 2) and (4, 4 + 1e-12); worse: (3 > 1)
    assert compare_counts(a, b) == (1, 2, 1)


@pytest.mark.slow
def test_evaluate_plan_job_smoke():
    from terranav.sim.plan_eval import evaluate_plan_job

    from terranav.types import SensorConfig

    r = evaluate_plan_job(("mixed", 0, "flank+start"), sensor=SensorConfig(width=48, height=36))
    assert set(r["planners"]) == {"terranav", "baseline_occ", "oracle"}
    oracle = r["planners"]["oracle"]
    assert oracle["success"] and oracle["metrics"]["length_m"] > 0.0
    assert "gt_cost_m" in oracle  # definition-dependent, from the pipeline
