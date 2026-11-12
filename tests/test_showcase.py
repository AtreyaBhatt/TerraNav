"""Showcase selection rule, replay bookkeeping, and offscreen smoke tests of the 3D renders."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import pyvista as pv

from terranav.config import SimConfig
from terranav.perception.traversability import TraversabilityConfig
from terranav.sim.evaluate import PhysicalLimits, evaluate_path
from terranav.sim.scenario import generate_scenario
from terranav.sim.showcase import (
    ShowcaseRun,
    dev_records,
    paired_metric,
    path_prefix,
    path_slice,
    pick_median,
    rough_exposure_along,
    rough_intervals,
)
from terranav.sim.trav_eval import ground_truth


def test_pick_median_even_count_takes_the_upper_middle():
    # sorted: -0.3 (s2), -0.2 (s0), -0.1 (s3), 0.0 (s1): middle ranks 2 and 3 -> the upper,
    # i.e. the one less favourable to TerraNav (s3)
    ranking, seed = pick_median({0: -0.2, 1: 0.0, 2: -0.3, 3: -0.1})
    assert [s for s, _ in ranking] == [2, 0, 3, 1]
    assert seed == 3


def test_pick_median_odd_count_and_ties_by_seed():
    assert pick_median({5: 1.0, 4: 2.0, 9: 3.0})[1] == 4
    ranking, seed = pick_median({7: 0.0, 3: 0.0, 5: 0.0, 1: 0.0})
    assert [s for s, _ in ranking] == [1, 3, 5, 7] and seed == 5
    with pytest.raises(ValueError):
        pick_median({})


def test_dev_records_refuses_held_out_seeds():
    with pytest.raises(ValueError, match="held-out"):
        dev_records("rough", (0, 1000))


def test_paired_metric_skips_failed_plans():
    def rec(seed, planner, alpha, value):
        ok = value is not None
        return {
            "seed": seed,
            "planner": planner,
            "alpha": alpha,
            "success": ok,
            "metrics": {"m": value} if ok else None,
        }

    records = [
        rec(0, "baseline_occ", None, 0.5),
        rec(0, "terranav", 4.0, 0.2),
        rec(0, "oracle", 4.0, 0.1),
        rec(1, "baseline_occ", None, 0.5),
        rec(1, "terranav", 4.0, None),
    ]
    assert paired_metric(records, "m", 4.0) == {0: pytest.approx(-0.3)}


def test_path_prefix_cuts_inside_a_segment():
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(path_prefix(path, 1.0), [[0, 0], [1, 0]])
    np.testing.assert_allclose(path_prefix(path, 5.0), [[0, 0], [3, 0], [3, 2]])
    np.testing.assert_allclose(path_prefix(path, 99.0), path)


def test_rough_exposure_along_ends_at_the_evaluators_value():
    sc = generate_scenario("rough", 0)
    rough = [f for f in sc.params["features"] if f["kind"] == "rough"]
    cx, cy = rough[0]["center_xy"]
    path = np.array([sc.start_xy, (cx, cy), sc.goal_xy])  # through the first patch's centre
    total = evaluate_path(sc, path, PhysicalLimits()).rough_exposure_m2
    assert total > 0
    assert rough_exposure_along(sc, path, 1e9) == total
    s = np.linspace(0.0, 20.0, 41)
    acc = [rough_exposure_along(sc, path, v) for v in s]
    # non-decreasing up to rounding (prefixes ending past a patch re-split the last segment)
    assert acc[0] == 0.0 and np.all(np.diff(acc) >= -1e-12)


def test_path_slice_between_arc_lengths():
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(path_slice(path, 1.0, 5.0), [[1, 0], [3, 0], [3, 2]])
    np.testing.assert_allclose(path_slice(path, 0.0, 99.0), path)


def test_rough_intervals_match_the_evaluators_rough_length():
    sc = generate_scenario("rough", 0)
    rough = [f for f in sc.params["features"] if f["kind"] == "rough"]
    cx, cy = rough[0]["center_xy"]
    path = np.array([sc.start_xy, (cx, cy), sc.goal_xy])
    m = evaluate_path(sc, path, PhysicalLimits())
    iv = rough_intervals(sc, path)
    assert iv and all(a < b for a, b in iv)
    assert all(b1 < a2 for (_, b1), (a2, _) in itertools.pairwise(iv))  # merged
    # union of disk intersections = the evaluator's per-patch sum when patches do not overlap
    disks = [(np.asarray(f["center_xy"]), f["radius"]) for f in rough]
    overlap = any(
        np.hypot(*(c1 - c2)) < r1 + r2
        for i, (c1, r1) in enumerate(disks)
        for c2, r2 in disks[i + 1 :]
    )
    if not overlap:
        assert sum(b - a for a, b in iv) == pytest.approx(m.length_in_rough_m, abs=1e-9)
    far = np.array([[0.2, 0.2], [0.2, 0.4]])  # map corner, outside every patch
    assert rough_intervals(sc, far) == []


def _offscreen_gl_available() -> bool:
    try:
        pl = pv.Plotter(off_screen=True, window_size=[16, 16])
        pl.add_mesh(pv.Sphere())
        pl.screenshot(return_img=True)
        pl.close()
    except Exception:  # noqa: BLE001 - VTK/GL failures vary by platform; any means "cannot render"
        return False
    return True


@pytest.fixture(scope="module")
def offscreen_gl():
    if not _offscreen_gl_available():
        pytest.skip("offscreen OpenGL rendering is unavailable in this environment")


@pytest.fixture(scope="module")
def tiny_run():
    """A cheap ShowcaseRun: GT traversability stands in for the perceived map, two fake poses."""
    from terranav.frames import T_world_cam

    sc = generate_scenario("rough", 0)
    grid = SimConfig().map_grid
    _, gt = ground_truth(sc, grid, TraversabilityConfig())
    straight = np.array([sc.start_xy, sc.goal_xy])
    detour = np.array([sc.start_xy, (6.0, 1.0), sc.goal_xy])
    paths = {"terranav": detour, "baseline_occ": straight, "oracle": detour}
    limits = PhysicalLimits()
    poses = [T_world_cam(np.array([1.0, 6.0, 0.5]), 0.0, 0.4) for _ in range(2)]
    rng = np.random.default_rng(0)
    return ShowcaseRun(
        scenario=sc,
        poses=poses,
        points=[rng.uniform(0, 12, size=(50, 3)), np.zeros((0, 3))],
        perceived=gt,
        ground_truth=gt,
        paths_xy=paths,
        metrics={n: evaluate_path(sc, p, limits) for n, p in paths.items()},
    )


@pytest.mark.slow
def test_hero_renders_at_full_size(tmp_path, offscreen_gl, tiny_run):
    from PIL import Image

    from terranav.viz.showcase3d import HERO_SIZE, render_hero

    out = render_hero(tiny_run, SimConfig().map_grid, "caption", tmp_path / "hero.png", 4.0)
    assert Image.open(out).size == HERO_SIZE


@pytest.mark.slow
def test_replay_frame_count_and_final_counters(offscreen_gl, tiny_run):
    from terranav.viz.showcase3d import ReplayTiming, replay_frames

    timing = ReplayTiming(per_pose=2, mapping=2, planning=2, executing=3, hold=2)
    frames = list(replay_frames(tiny_run, SimConfig().map_grid, "t", timing, (160, 90)))
    assert len(frames) == 2 * 2 + 2 + 2 + 3 + 2
    assert all(f.shape == (90, 160, 3) for f in frames)
    assert np.array_equal(frames[-1], frames[-2])  # held final frame


@pytest.mark.slow
def test_run_showcase_reproduces_the_benchmark_job():
    # The renders recompute the scenario; their paths must be the benchmark job's paths.
    from terranav.config import SimConfig as Cfg
    from terranav.sim.showcase import run_showcase

    records = dev_records("rough", (1,), 4.0)
    run = run_showcase("rough", 1, 4.0)
    grid = Cfg().map_grid
    for r in records:
        x, y = grid.rc_to_xy(*np.array(r["path_rc"]).T)
        np.testing.assert_array_equal(run.paths_xy[r["planner"]], np.column_stack([x, y]))
        assert run.metrics[r["planner"]].rough_exposure_m2 == r["metrics"]["rough_exposure_m2"]
