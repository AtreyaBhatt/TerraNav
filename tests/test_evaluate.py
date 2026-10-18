"""Independent path evaluator on hand-built worlds (expected values derived by hand)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from terranav.sim.evaluate import PhysicalLimits, evaluate_path
from terranav.sim.world import add_terrace
from terranav.types import Box, GridSpec, Scenario, World

TERRAIN = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
LIMITS = PhysicalLimits()


def scenario(hf=None, boxes=(), features=()) -> Scenario:
    hf = np.zeros((TERRAIN.ny, TERRAIN.nx)) if hf is None else hf
    world = World(hf, TERRAIN, tuple(boxes), "test", 0)
    return Scenario(world, (1.0, 6.0), (11.0, 6.0), "test", 0, {"features": list(features)})


def path(*xy) -> np.ndarray:
    return np.array(xy, dtype=float)


def test_limits():
    assert (LIMITS.robot_radius_m, LIMITS.max_step_m, LIMITS.max_slope_deg) == (0.3, 0.20, 30.0)


def test_flat_world_is_all_zero():
    m = evaluate_path(scenario(), path((2.0, 6.0), (9.0, 6.0), (9.0, 3.0)), LIMITS)
    assert m.length_m == pytest.approx(10.0, abs=1e-12)
    assert m.min_clearance_m == math.inf and not m.collision
    assert m.max_rise_m == 0.0 and m.max_slope_deg == 0.0 and m.mean_slope_deg == 0.0
    assert m.max_pitch_deg == 0.0 and m.max_roll_deg == 0.0
    assert m.rough_exposure_m2 == 0.0 and m.length_in_rough_m == 0.0
    assert not m.violation and m.violations == ()


@pytest.mark.parametrize(("y", "collision"), [(4.75, True), (4.65, False)])
def test_box_clearance(y, collision):
    # Box footprint x in [5, 6], y in [5, 7]; a path along y = 4.75 passes 0.25 m below it
    # (< robot radius 0.3 -> collision), along y = 4.65 passes 0.35 m below (clear).
    box = Box((5.0, 5.0, 0.0), (6.0, 7.0, 0.5))
    m = evaluate_path(scenario(boxes=[box]), path((3.0, y), (8.0, y)), LIMITS)
    assert m.min_clearance_m == pytest.approx(5.0 - y, abs=1e-9)
    assert m.collision is collision
    assert (m.violations == ("collision",)) is collision
    if collision:
        assert 5.0 <= m.collision_xy[0] <= 6.0 and m.collision_xy[1] == pytest.approx(y)


def test_terrace_edge_rise_and_footprint_length():
    # One-step 0.25 m mesa over x in [4, 8], y in [4, 8]. A path along y = 6 from x = 2 to 5
    # climbs its edge at x = 4: max rise 0.25 m (> max_step 0.20 -> violation) and 1.0 m of the
    # path lies on the terrace footprint. The edge is a discontinuity: it does not count as slope.
    hf = add_terrace(
        np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (6.0, 6.0), (4.0, 4.0), 0.0, 0.25, 1, 2.0, 1
    )
    feat = {"kind": "terrace", "center_xy": (6.0, 6.0), "size_xy": (4.0, 4.0), "yaw_rad": 0.0}
    m = evaluate_path(scenario(hf, features=[feat]), path((2.0, 6.0), (5.0, 6.0)), LIMITS)
    assert m.max_rise_m == pytest.approx(0.25, abs=1e-3)
    assert 3.8 <= m.max_rise_xy[0] <= 4.2
    assert m.length_on_features_m["terrace"] == pytest.approx(1.0, abs=1e-12)
    assert m.max_slope_deg == 0.0
    assert m.violations == ("rise",)


def plane_20deg() -> np.ndarray:
    """Terrain z = tan(20 deg) * x on the 0.05 m grid (cell centres)."""
    x = (np.arange(TERRAIN.nx) + 0.5) * TERRAIN.resolution
    return np.tile(math.tan(math.radians(20.0)) * x, (TERRAIN.ny, 1))


def test_ramp_uphill_pitch_and_contour_roll():
    sc = scenario(plane_20deg())
    up = evaluate_path(sc, path((2.0, 6.0), (8.0, 6.0)), LIMITS)  # along the gradient (+x)
    assert up.max_pitch_deg == pytest.approx(20.0, abs=1e-6)
    assert up.max_roll_deg == pytest.approx(0.0, abs=1e-6)
    assert up.max_slope_deg == pytest.approx(20.0, abs=1e-6)
    assert up.mean_slope_deg == pytest.approx(20.0, abs=1e-6)
    # Rise over 0.2 m along a 20 deg slope: 0.2 * tan(20 deg) = 0.0728 m, no violation.
    assert up.max_rise_m == pytest.approx(0.2 * math.tan(math.radians(20.0)), abs=1e-6)
    assert not up.violation
    contour = evaluate_path(sc, path((6.0, 2.0), (6.0, 10.0)), LIMITS)  # along a contour (+y)
    assert contour.max_pitch_deg == pytest.approx(0.0, abs=1e-6)
    assert contour.max_roll_deg == pytest.approx(20.0, abs=1e-6)
    assert contour.max_rise_m == pytest.approx(0.0, abs=1e-9)


def test_rough_exposure_on_a_chord():
    # Patch centre (6, 6), radius 1, rms 0.05; a straight path along y = 6.6 cuts a chord of
    # length 2 * sqrt(1 - 0.6^2) = 1.6 m: exposure 1.6 * 0.05 = 0.08 m^2.
    feat = {
        "kind": "rough",
        "center_xy": (6.0, 6.0),
        "radius": 1.0,
        "rms_m": 0.05,
        "corr_len_m": 0.2,
    }
    m = evaluate_path(scenario(features=[feat]), path((3.0, 6.6), (9.0, 6.6)), LIMITS)
    assert m.length_in_rough_m == pytest.approx(1.6, abs=1e-9)
    assert m.rough_exposure_m2 == pytest.approx(0.08, abs=1e-9)
    assert m.length_on_features_m["rough"] == pytest.approx(1.6, abs=1e-9)


def test_rotated_ramp_footprint_length():
    # Ramp footprint 2 x 1 m centred at (6, 6), rotated 90 deg: it spans x in [5.5, 6.5],
    # y in [5, 7]. A path along y = 6 from x = 3 to 9 runs 1.0 m inside it.
    feat = {"kind": "ramp", "center_xy": (6.0, 6.0), "size_xy": (2.0, 1.0), "yaw_rad": math.pi / 2}
    m = evaluate_path(scenario(features=[feat]), path((3.0, 6.0), (9.0, 6.0)), LIMITS)
    assert m.length_on_features_m["ramp"] == pytest.approx(1.0, abs=1e-9)


def test_small_rotated_step_is_not_a_slope():
    # A 0.05 m one-step pit, rotated 0.725 rad: its staircase-shaped edge has 5 cm jumps along
    # both axes, where a raw gradient reads ~35 deg. It is a step (45 deg locally > 40 deg), so
    # it counts for max_rise (0.05 m) but not as slope: no slope violation.
    hf = add_terrace(
        np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (6.0, 6.0), (2.0, 2.0), 0.725, 0.05, 1, 1.0, -1
    )
    m = evaluate_path(scenario(hf), path((3.0, 5.3), (9.0, 6.7)), LIMITS)
    assert m.max_rise_m == pytest.approx(0.05, abs=1e-3)
    assert m.max_slope_deg < 1.0
    assert not m.violation


def test_steep_slope_is_measured_as_slope():
    # A 35 deg plane (per-cell rise 0.035 m < the 40 deg step threshold 0.042 m) is a slope:
    # 35 deg > 30 deg -> slope violation.
    x = (np.arange(TERRAIN.nx) + 0.5) * TERRAIN.resolution
    hf = np.tile(math.tan(math.radians(35.0)) * x, (TERRAIN.ny, 1))
    m = evaluate_path(scenario(hf), path((2.0, 6.0), (4.0, 6.0)), LIMITS)
    assert m.max_slope_deg == pytest.approx(35.0, abs=1e-6)
    assert m.violations == ("slope",)


def test_ramp_slope_unchanged_by_smoothing_away_from_edges():
    # 20 deg frustum ramp (4 x 4 m footprint, top at 0.6 m): along the centre line of one face,
    # away from the footprint edge and the plateau kink, the smoothed slope is the face slope.
    from terranav.sim.world import add_ramp

    hf = add_ramp(
        np.zeros((TERRAIN.ny, TERRAIN.nx)), TERRAIN, (6.0, 6.0), (4.0, 4.0), 0.0, 20.0, 0.6
    )
    # face x in [4, 5.65] (rises 0.364 m/m until 0.6 m); sample x in [4.4, 5.2] on y = 6
    m = evaluate_path(scenario(hf), path((4.4, 6.0), (5.2, 6.0)), LIMITS)
    assert m.max_slope_deg == pytest.approx(20.0, abs=0.5)
    assert m.max_pitch_deg == pytest.approx(20.0, abs=0.5)
    assert m.max_slope_raw_deg == pytest.approx(20.0, abs=0.5)


def test_texture_slope_is_smoothed_away():
    # Texture z = 0.05 sin(2 pi x / 0.4) sin(2 pi y / 0.4) (0.4 m wavelength, k = 15.7 /m). Raw
    # gradient max ~ A k = 0.785, times the central-difference factor sin(kh)/(kh) = 0.90 for
    # h = 0.05 m: ~0.71, about 35 deg. Adjacent cells differ by <= 2A sin(kh/2) = 0.038 m < the
    # 40 deg step threshold (0.042 m), so the texture is a surface, not steps. A 0.1 m Gaussian
    # damps each direction by exp(-k^2 sigma^2 / 2) = 0.29; the product texture by 0.29^2 = 0.085:
    # about 3 deg (measured 31.1 deg raw, 2.9 deg smoothed along the crest line).
    xs = (np.arange(TERRAIN.nx) + 0.5) * TERRAIN.resolution
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    hf = 0.05 * np.sin(2 * np.pi * X / 0.4) * np.sin(2 * np.pi * Y / 0.4)
    # Along the crest line y = 6.1 (sin(k y) = 1) the gradient reaches A k: a diagonal path
    # would only see A k / 2 (~26 deg raw).
    m = evaluate_path(scenario(hf), path((3.0, 6.1), (9.0, 6.1)), LIMITS)
    assert m.max_slope_raw_deg > 30.0
    assert m.max_slope_deg < 15.0
    assert "slope" not in m.violations
