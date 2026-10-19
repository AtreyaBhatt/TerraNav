"""Benchmark machinery: config hash, the alpha = 0 invariant, summary rendering, smoke run.

Only dev seeds (0..9) are used here; the held-out benchmark seeds are never touched by tests.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from terranav.config import SimConfig
from terranav.perception.traversability import TraversabilityConfig
from terranav.sim.benchmark import (
    BenchmarkConfig,
    benchmark_job,
    config_dict,
    config_hash,
    render_summary,
    run_benchmark,
    scan_parameters,
)
from terranav.sim.evaluate import PhysicalLimits
from terranav.types import GridSpec, SensorConfig

# 96 x 72 is the smallest sensor tried that still maps a route on dev scenarios (48 x 36 left
# start and goal disconnected, so both planners failed and the invariant was vacuous).
SMALL = SensorConfig(width=96, height=72)


def perturb(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1 if value == 0 else value * 1.1
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return value[:-1] if len(value) > 1 else (*value, value[0] if value else 0)
    raise TypeError(type(value))


def test_config_hash_is_stable_and_short():
    a, b = config_hash(), config_hash()
    assert a == b and len(a) == 12 and int(a, 16) >= 0
    assert json.dumps(config_dict(), sort_keys=True)  # serialisable


@pytest.mark.parametrize(
    ("kwarg", "obj"),
    [
        ("bcfg", BenchmarkConfig()),
        ("sensor", SensorConfig()),
        ("tcfg", TraversabilityConfig()),
        ("limits", PhysicalLimits()),
    ],
    ids=["benchmark", "sensor", "traversability", "limits"],
)
def test_config_hash_changes_with_every_field(kwarg, obj):
    base = config_hash()
    for f in dataclasses.fields(obj):
        changed = replace(obj, **{f.name: perturb(getattr(obj, f.name))})
        assert config_hash(**{kwarg: changed}) != base, f"{kwarg}.{f.name}"


def test_config_hash_changes_with_scan_parameters_and_grids():
    base = config_hash()
    for key, value in scan_parameters().items():
        scan = dict(scan_parameters())
        scan[key] = perturb(value) if not isinstance(value, dict) else {**value, "x": 1}
        assert config_hash(scan=scan) != base, key
    g = SimConfig().map_grid
    assert config_hash(map_grid=GridSpec(0.2, g.size_x, g.size_y, g.origin)) != base
    t = SimConfig().terrain_grid
    assert config_hash(terrain_grid=GridSpec(t.resolution, 10.0, t.size_y, t.origin)) != base


def test_alpha_zero_invariant_on_random_maps():
    # Planning-level invariant on synthetic traversability maps: with alpha = 0 the graded and the
    # ungraded planner see identical edge costs, so they return identical smoothed paths.
    from terranav.perception.traversability import Traversability
    from terranav.planning.planners import plan

    grid = GridSpec(resolution=0.1, size_x=3.0, size_y=3.0, origin=(0.0, 0.0))
    rng = np.random.default_rng(0)
    for _ in range(20):
        lethal = rng.random((30, 30)) < 0.2
        lethal[0, 0] = lethal[29, 29] = False
        cost = np.where(lethal, np.nan, rng.random((30, 30)) * 0.99)
        t = Traversability(lethal=lethal, core_reason=lethal.astype(np.uint8), cost=cost)
        graded = plan(t, (0.05, 0.05), (2.95, 2.95), grid, 0.0, graded=True)
        ungraded = plan(t, (0.05, 0.05), (2.95, 2.95), grid, 0.0, graded=False)
        assert (graded is None) == (ungraded is None)
        if graded is not None:
            np.testing.assert_array_equal(graded.path_rc, ungraded.path_rc)


@pytest.mark.parametrize(
    "job",
    [
        ("mixed", 0),
        pytest.param(("rough", 1), marks=pytest.mark.slow),
        pytest.param(("steps", 2), marks=pytest.mark.slow),
    ],
)
def test_alpha_zero_terranav_equals_baseline(job):
    # alpha = 0: both planners' edge cost is the step length, identical floats, heuristic and
    # tie-breaking, so the paths must be identical (dev seeds, small sensor).
    bcfg = BenchmarkConfig(families=(job[0],), seeds=(job[1],), alphas=(0.0,), default_alpha=0.0)
    recs = benchmark_job(job, bcfg, sensor=SMALL)
    tn = next(r for r in recs if r["planner"] == "terranav" and r["alpha"] == 0.0)
    bl = next(r for r in recs if r["planner"] == "baseline_occ")
    assert tn["success"] and bl["success"]
    assert tn["path_rc"] == bl["path_rc"]
    assert tn["alpha0_matches_baseline"] is True


def hand_record(family, seed, planner, alpha, length, exposure=0.0, rise=0.0, violations=()):
    return {
        "family": family,
        "seed": seed,
        "planner": planner,
        "alpha": alpha,
        "success": True,
        "path_rc": [[0, 0], [1, 1]],
        "metrics": {
            "length_m": length,
            "min_clearance_m": math.inf,
            "max_rise_m": rise,
            "max_slope_deg": 1.0,
            "max_slope_raw_deg": 1.0,
            "rough_exposure_m2": exposure,
            "length_in_rough_m": 0.0,
            "violations": list(violations),
        },
        "gt_cost_m": 0.1,
        "timings_s": {"simulation": 1.0, "perception": 0.02, "features_trav": 0.006, "plan": 0.01},
        "alpha0_matches_baseline": True if planner == "terranav" and alpha == 0.0 else None,
    }


def test_summary_renders_from_hand_built_records():
    bcfg = BenchmarkConfig(families=("rough",), seeds=(0, 1), alphas=(0.0, 4.0))
    recs = []
    for seed in (0, 1):
        recs.append(hand_record("rough", seed, "baseline_occ", None, 10.0, 0.20, 0.06))
        for a in (0.0, 4.0):
            ex = 0.20 if a == 0.0 else 0.02
            recs.append(
                hand_record("rough", seed, "terranav", a, 10.0 if a == 0 else 10.5, ex, 0.01)
            )
            recs.append(hand_record("rough", seed, "oracle", a, 10.0 if a == 0 else 10.4, 0.0, 0.0))
    recs[-1]["metrics"]["violations"] = ["collision"]
    meta = {"git": "abc1234", "config_hash": "0123456789ab", "started": "t0", "wall_s": 1.0}
    md = render_summary(recs, bcfg, meta, latency={})
    assert "## (a) Headline" in md and "## (b) Paired" in md and "## (d) Alpha sweep" in md
    # Paired TerraNav - Baseline-Occ rough exposure at alpha = 4: 0.02 - 0.20 = -0.18 in both seeds.
    assert "-0.1800" in md
    # One oracle violation, listed with its seed.
    assert "rough seed 1 oracle alpha 4: collision" in md
    assert "alpha = 0 invariant: 0 mismatches" in md
    assert "latency not measured at this commit" in md


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    bcfg = BenchmarkConfig(families=("mixed", "rough"), seeds=(0,), alphas=(0.0, 4.0))
    out = run_benchmark(bcfg, tmp_path / "run", workers=1, sensor=SMALL, git="dev0000", started="t")
    lines = (out / "raw.jsonl").read_text().splitlines()
    assert len(lines) == 2 * (1 + 2 * 2)  # 2 scenarios x (baseline + 2 planners x 2 alphas)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["config_hash"] == config_hash(bcfg=bcfg, sensor=SMALL)
    md = (out / "summary.md").read_text()
    assert "alpha = 0 invariant: 0 mismatches" in md
    for name in ("tradeoff_mixed.png", "tradeoff_rough.png", "headline.png"):
        assert (out / name).stat().st_size > 0
    rec = json.loads(lines[0])
    assert {"family", "seed", "planner", "alpha", "success", "metrics", "timings_s"} <= set(rec)
    assert np.isfinite(rec["timings_s"]["perception"])
