"""Presentation figures: plotted values come from raw.jsonl and match the committed summary.md.

Reads results/0cf7be5_96a4e7ffad40 (the committed benchmark run) without modifying it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from terranav.sim.benchmark_report import render_summary
from terranav.sim.plan_eval import paired_bootstrap_ci
from terranav.viz.figures import (
    HEADLINE_SCALE,
    TRADEOFF_Y,
    caption,
    check_headline_against_summary,
    headline_data,
    load_run,
    make_figures,
    sample_note,
    summary_paired_table,
    tradeoff_data,
)

RUN = Path(__file__).resolve().parents[1] / "results" / "0cf7be5_96a4e7ffad40"


@pytest.fixture(scope="module")
def run():
    records, bcfg, meta = load_run(RUN)
    return records, bcfg, meta, (RUN / "summary.md").read_text()


def test_summary_md_is_reproduced_from_raw_jsonl(run):
    # raw.jsonl is the summary's source: sections (a)-(e) re-render byte for byte. (f) depends
    # on latency files in out/ at run time, not on raw.jsonl.
    records, bcfg, meta, summary = run
    rendered = render_summary(records, bcfg, meta, {})
    assert rendered.split("## (f)")[0] == summary.split("## (f)")[0]


def test_headline_values_match_summary(run):
    records, bcfg, _, summary = run
    data = headline_data(records, bcfg)
    # 3 metrics x 2 comparisons x 5 families x (mean, lo, hi)
    assert check_headline_against_summary(data, summary) == 90


def test_headline_values_equal_a_direct_recomputation(run):
    # Independent of the summary's helpers: select records by hand, bootstrap directly.
    records, bcfg, _, _ = run
    data = headline_data(records, bcfg)
    a = bcfg.default_alpha
    for (key, other, fam), stat in data.items():

        def by_seed(planner, fam=fam, key=key):
            return {
                r["seed"]: r["metrics"][key]
                for r in records
                if r["family"] == fam
                and r["planner"] == planner
                and (r["alpha"] is None or r["alpha"] == a)
                and r["success"]
            }

        tn, ot = by_seed("terranav"), by_seed(other)
        d = np.array([tn[s] - ot[s] for s in sorted(tn) if s in ot])
        expected = paired_bootstrap_ci(d, bcfg.bootstrap_resamples, bcfg.bootstrap_seed)
        # plotted in the panel's unit (max rise in cm)
        np.testing.assert_allclose(stat, HEADLINE_SCALE[key] * np.array(expected), atol=1e-9)


def test_max_rise_is_plotted_in_cm(run):
    records, bcfg, _, _ = run
    data = headline_data(records, bcfg)
    # summary.md (b): rough terranav - baseline_occ max rise m = -0.0254
    assert f"{data[('max_rise_m', 'baseline_occ', 'rough')][0]:.2f}" == "-2.54"


def test_check_rejects_a_value_that_differs_from_summary(run):
    records, bcfg, _, summary = run
    data = headline_data(records, bcfg)
    key = ("max_rise_m", "baseline_occ", "rough")
    mu, lo, hi = data[key]
    data[key] = (mu + 0.1, lo, hi)  # 0.1 cm = 0.001 m: visible at the summary's 4 decimals
    with pytest.raises(AssertionError, match="rough terranav - baseline_occ max_rise_m"):
        check_headline_against_summary(data, summary)


def _sweep_means(summary: str, family: str) -> tuple[dict, dict]:
    """From summary (d): ({alpha: {"terranav"|"oracle": {metric: str}}}, {metric: str} Baseline)."""
    block = summary.split(f"### {family}\n")[1].split("### ")[0]
    rows = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 9 and re.fullmatch(r"[\d.]+", cells[0]):
            rows[float(cells[0])] = {
                "terranav": {"rough_exposure_m2": cells[2], "max_rise_m": cells[3]},
                "oracle": {"rough_exposure_m2": cells[6], "max_rise_m": cells[7]},
            }
    m = re.search(r"rough exposure ([\d.]+), max rise ([\d.]+)", block)
    return rows, {"rough_exposure_m2": m.group(1), "max_rise_m": m.group(2)}


@pytest.mark.parametrize("family", list(TRADEOFF_Y))
def test_tradeoff_values_match_summary(run, family):
    records, bcfg, _, summary = run
    curves = tradeoff_data(records, bcfg, family)
    key = TRADEOFF_Y[family]
    sweep, baseline = _sweep_means(summary, family)

    def mean(planner, alpha):
        return np.mean(
            [
                r["metrics"][key]
                for r in records
                if r["family"] == family
                and r["planner"] == planner
                and (planner == "baseline_occ" or r["alpha"] == alpha)
            ]
        )

    # every scenario succeeded in this run, so mean of paired reductions = difference of means
    assert all(r["success"] for r in records if r["family"] == family)
    assert f"{mean('baseline_occ', None):.4f}" == baseline[key]
    for planner, curve in curves.items():
        for a, (_, y) in curve.items():
            assert f"{mean(planner, a):.4f}" == sweep[a][planner][key]
            assert abs(y[0] - (mean("baseline_occ", None) - mean(planner, a))) <= 1e-9
    # at the primary alpha, TerraNav's y is exactly minus summary (b)'s TerraNav - Baseline-Occ
    label = {"rough_exposure_m2": "rough exposure m2", "max_rise_m": "max rise m"}[key]
    mu, lo, hi = summary_paired_table(summary)[(family, "terranav", "baseline_occ", label)]
    y = curves["terranav"][bcfg.default_alpha][1]
    assert [f"{-v:+.4f}" for v in (y[0], y[2], y[1])] == [mu, lo, hi]


def test_terranav_at_alpha_zero_sits_on_the_origin(run):
    # The alpha = 0 invariant: identical paths, so every paired difference is exactly zero.
    records, bcfg, _, _ = run
    for fam in TRADEOFF_Y:
        x, y = tradeoff_data(records, bcfg, fam)["terranav"][0.0]
        assert x == (0.0, 0.0, 0.0) and y == (0.0, 0.0, 0.0)


def test_caption_is_derived_from_the_records(run):
    records, bcfg, _, _ = run
    assert sample_note(records, bcfg) == "held-out seeds 1000-1019, n = 20 per family"
    assert caption(records, bcfg) == (
        "held-out seeds 1000-1019, n = 20 per family; TerraNav at alpha = 0 equals Baseline-Occ"
    )
    broken = [dict(r) for r in records]
    next(r for r in broken if r.get("alpha0_matches_baseline"))["alpha0_matches_baseline"] = False
    with pytest.raises(ValueError, match="invariant"):
        caption(broken, bcfg)


def test_load_run_uses_the_runs_own_config():
    _, bcfg, meta = load_run(RUN)
    assert bcfg.seeds == tuple(range(1000, 1020))
    assert bcfg.bootstrap_resamples == meta["configs"]["benchmark"]["bootstrap_resamples"]
    assert json.loads((RUN / "meta.json").read_text())["config_hash"] == "96a4e7ffad40"


@pytest.mark.slow
def test_figures_regenerate_deterministically(tmp_path):
    first = make_figures(RUN, tmp_path / "a")
    second = make_figures(RUN, tmp_path / "b")
    assert [p.name for p in first] == [
        "headline.png",
        "tradeoff_rough.png",
        "tradeoff_mixed.png",
        "tradeoff_slopes.png",
        "tradeoff_steps.png",
    ]
    for p, q in zip(first, second, strict=True):
        assert p.read_bytes() == q.read_bytes(), p.name
