"""Gap filling, footprint marking and ground-truth elevation tests (hand-derived)."""

from __future__ import annotations

import numpy as np
import pytest

from terranav.perception.elevation_map import (
    FILLED,
    FOOTPRINT,
    OBSERVED,
    SIGMA_FLOOR_M,
    UNKNOWN,
    fuse_points,
)
from terranav.perception.hole_fill import fill_gaps, mark_footprint
from terranav.sim.ground_truth import ground_truth_elevation
from terranav.types import Box, GridSpec, Scenario, World

MAP = GridSpec(resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))
TERRAIN = GridSpec(resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0))


def map_with_cells(rows: np.ndarray, cols: np.ndarray, z: np.ndarray):
    """One point at the centre of each given map cell, with the given height."""
    x = (cols + 0.5) * MAP.resolution
    y = (rows + 0.5) * MAP.resolution
    return fuse_points(np.column_stack([x, y, z]), np.full(len(z), 0.01), MAP)


GAP_ROWS = np.array([r for r in range(117) if r % 3])
# The guard's plane fit (radius 0.3 m) around a gap cell sees two observed rows. In the map's
# first and last column only dx >= 0 (or <= 0) exists: 3 + 3 = 6 cells < MIN_SUPPORT = 8, so
# those two columns are not filled; from column 1 on there are 4 + 4 = 8.
INNER_COLS = slice(1, 119)


def striped_fill(z_of_col, **guards):
    """Observed every 3rd row (0, 3, ..., 117), one point per cell; 2-row (0.2 m) gaps."""
    rr, cc = np.meshgrid(np.arange(0, 120, 3), np.arange(120), indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    return fill_gaps(map_with_cells(rows, cols, z_of_col(cols)), **guards)


def striped(z_of_col):
    return striped_fill(z_of_col)


def test_fill_rows_on_inclined_plane():
    # z = 0.2 x, x = (col + 0.5) * 0.1. The window spans 0.4 m in x -> 0.08 m <= STEP_GUARD_M,
    # and the plane residual is ~0, so every gap cell away from the first/last column fills.
    # Rows 118-119 lie outside the last observed row (not a gap).
    slope = 0.2
    m = striped(lambda cols: slope * (cols + 0.5) * 0.1)
    gap = m.source[GAP_ROWS, INNER_COLS]
    assert np.all(gap == FILLED)
    assert np.all(m.count[GAP_ROWS, INNER_COLS] == 0)
    x = (np.arange(120) + 0.5) * 0.1
    np.testing.assert_allclose(
        m.z_mean[GAP_ROWS, INNER_COLS], np.broadcast_to(slope * x[INNER_COLS], gap.shape), atol=0.01
    )
    # z_min / z_max of a filled cell bracket its neighbours.
    assert np.all(m.z_min[GAP_ROWS, INNER_COLS] <= m.z_mean[GAP_ROWS, INNER_COLS] + 1e-12)
    assert np.all(m.z_max[GAP_ROWS, INNER_COLS] >= m.z_mean[GAP_ROWS, INNER_COLS] - 1e-12)
    # z_mean_var of a filled cell = max over observed neighbours = 1 / (1 / 0.01^2) = 1e-4.
    np.testing.assert_allclose(m.z_mean_var[GAP_ROWS, INNER_COLS], 1e-4)
    assert np.all(m.source[118:, :] == UNKNOWN)
    assert np.all(m.source[::3, :][:40] == OBSERVED)


def test_steep_ramp_is_not_filled():
    # 25 deg ramp: planar (residual ~0, the residual guard alone would fill), but its window
    # spans 0.4 m * tan(25 deg) = 0.187 m > STEP_GUARD_M, so the span guard refuses it.
    m = striped(lambda cols: np.tan(np.radians(25.0)) * (cols + 0.5) * 0.1)
    assert not (m.source == FILLED).any()
    unguarded = striped_fill(
        lambda cols: np.tan(np.radians(25.0)) * (cols + 0.5) * 0.1, step_guard_m=np.inf
    )
    assert np.all(unguarded.source[GAP_ROWS, INNER_COLS] == FILLED)


def test_step_edge_is_not_filled_across():
    # Striped rows on flat ground, 0 m for cols <= 59 and 0.3 m for cols >= 60. A gap cell's
    # plane fit (radius 3 cells) only contains the observed rows at |dy| = 1 and 2, where the
    # disk spans |dx| <= 2; so it mixes both levels iff the column is within 58..61:
    # residual >> 0.02 m, those stay UNKNOWN. Columns 57 and 62 see one level only (in the
    # fit and in the +-2-column fill window) and fill with it exactly.
    m = striped(lambda cols: np.where(cols >= 60, 0.3, 0.0))
    assert np.all(m.source[GAP_ROWS, 58:62] == UNKNOWN)
    assert np.all(m.source[GAP_ROWS, 1:58] == FILLED)
    assert np.all(m.source[GAP_ROWS, 62:119] == FILLED)
    np.testing.assert_allclose(m.z_mean[GAP_ROWS, 1:58], 0.0, atol=1e-12)
    np.testing.assert_allclose(m.z_mean[GAP_ROWS, 62:119], 0.3, atol=1e-12)


def test_large_hole_stays_unknown_except_rim():
    # Everything observed (z = 0) except a 10 x 10 block, rows/cols 50..59. Closing with a
    # radius-2 disk can only reach 2 cells into the hole: its inner 6 x 6 stays unknown.
    hole = np.zeros((120, 120), dtype=bool)
    hole[50:60, 50:60] = True
    rr, cc = np.nonzero(~hole)
    m = fill_gaps(map_with_cells(rr, cc, np.zeros(len(rr))))
    assert np.all(m.source[52:58, 52:58] == UNKNOWN)
    filled = m.source == FILLED
    assert not filled[~hole].any()
    inner = np.zeros_like(hole)
    inner[52:58, 52:58] = True
    assert not (filled & inner).any()


def test_box_rim_is_not_filled():
    # Ground (z = 0) observed everywhere except a 10 x 10 box footprint, rows/cols 50..59.
    # Its rim cells see side-face points from z = 0.05 to 0.5; the top was never seen, so the
    # interior (rows/cols 51..58) is unknown. Two rim cells, (50, 54) and (50, 55), are also
    # unobserved. Any plane fit touching the rim has residual >> FILL_RESIDUAL_GUARD_M, so neither the
    # interior nor the rim gap may be filled.
    box = np.zeros((120, 120), dtype=bool)
    box[50:60, 50:60] = True
    interior = np.zeros_like(box)
    interior[51:59, 51:59] = True
    rim = box & ~interior
    rim_gap = np.zeros_like(box)
    rim_gap[50, 54:56] = True
    gr, gc = np.nonzero(~box)
    rr, cc = np.nonzero(rim & ~rim_gap)
    rows = np.concatenate([gr, rr, rr])
    cols = np.concatenate([gc, cc, cc])
    z = np.concatenate([np.zeros(len(gr)), np.full(len(rr), 0.05), np.full(len(rr), 0.5)])
    m = fill_gaps(map_with_cells(rows, cols, z))
    assert np.all(m.source[interior] == UNKNOWN)
    assert np.all(m.source[rim_gap] == UNKNOWN)
    assert np.all(m.source[rim & ~rim_gap] == OBSERVED)
    assert not (m.source == FILLED).any()


def test_box_rim_fills_without_guard():
    # Same scene with the guard disabled: the closing does reach the interior's outer corners
    # (e.g. (51, 51), whose radius-2 disk is covered by the dilated rim), so it is the guard,
    # not the closing, that keeps them unknown above. (Mid-edge rim gaps such as (50, 54)
    # are not closable either way: the interior below them is too far from observed cells.)
    box = np.zeros((120, 120), dtype=bool)
    box[50:60, 50:60] = True
    rim = box.copy()
    rim[51:59, 51:59] = False
    gr, gc = np.nonzero(~box)
    rr, cc = np.nonzero(rim)
    rows, cols = np.concatenate([gr, rr]), np.concatenate([gc, cc])
    z = np.concatenate([np.zeros(len(gr)), np.full(len(rr), 0.5)])
    unguarded = fill_gaps(
        map_with_cells(rows, cols, z), step_guard_m=np.inf, residual_guard_m=np.inf
    )
    guarded = fill_gaps(map_with_cells(rows, cols, z))
    for corner in ((51, 51), (51, 58), (58, 51), (58, 58)):
        assert unguarded.source[corner] == FILLED
        assert guarded.source[corner] == UNKNOWN


def test_footprint_after_fill():
    # Nothing observed near start (1, 6); the caller supplies a flat 0.3 m height field.
    # Footprint cells: map-cell centres within 0.4 m of (1, 6).
    rr, cc = np.meshgrid(np.arange(120), np.arange(60, 120), indexing="ij")
    m = fill_gaps(map_with_cells(rr.ravel(), cc.ravel(), np.zeros(rr.size)))
    m = mark_footprint(m, (1.0, 6.0), 0.4, np.full((120, 120), 0.3))
    xc = (np.arange(120) + 0.5) * 0.1
    X, Y = np.meshgrid(xc, xc, indexing="xy")
    inside = (X - 1.0) ** 2 + (Y - 6.0) ** 2 <= 0.4**2
    assert inside.sum() == 52  # 8 x 8 block of centres minus 3 at each corner
    np.testing.assert_array_equal(m.source == FOOTPRINT, inside)
    np.testing.assert_allclose(m.z_mean[inside], 0.3)
    np.testing.assert_allclose(m.z_min[inside], 0.3)
    np.testing.assert_allclose(m.z_max[inside], 0.3)
    np.testing.assert_allclose(m.z_mean_var[inside], SIGMA_FLOOR_M**2)


def test_ground_truth_elevation_block_mean_and_box_top():
    # Terrain height 0.01 * terrain column: map column C covers terrain columns 2C, 2C+1,
    # so its block mean is 0.01 * (2C + 0.5). A 0.8 m-high box over x in [2, 3], y in [4, 5]
    # covers map cells with centres x in 2.05..2.95 (cols 20..29), y in 4.05..4.95 (rows 40..49).
    hf = np.tile(0.01 * np.arange(TERRAIN.nx, dtype=float), (TERRAIN.ny, 1))
    box = Box((2.0, 4.0, 0.0), (3.0, 5.0, 0.8))
    sc = Scenario(World(hf, TERRAIN, (box,), "test", 0), (1.0, 6.0), (11.0, 6.0), "test", 0, {})
    gt = ground_truth_elevation(sc, MAP)
    assert gt.shape == (120, 120)
    expected = np.tile(0.01 * (2 * np.arange(120) + 0.5), (120, 1))
    expected[40:50, 20:30] = 0.8
    np.testing.assert_allclose(gt, expected, atol=1e-12)
    assert gt[45, 19] == pytest.approx(0.01 * 38.5)  # centre x = 1.95: not covered


def test_low_box_top_inside_planar_ground_is_not_filled():
    # Flat observed ground (z = 0) and a 0.25 m box whose rim was seen (z_min 0, z_max 0.25)
    # but whose top is a 2-cell unobserved gap. Shape: rows 60..63, cols 60..63 is the box;
    # its inner 2 x 2 (rows/cols 61..62) is unobserved. The rim cells' MEANS are set to 0 (as
    # if dominated by ground returns), so a plane fits the neighbourhood perfectly and the
    # residual guard alone would fill the top as ground; the span guard (0.25 > 0.10) refuses.
    gap = np.zeros((120, 120), dtype=bool)
    gap[61:63, 61:63] = True
    rim = np.zeros_like(gap)
    rim[60:64, 60:64] = True
    rim &= ~gap
    rr, cc = np.nonzero(~gap)
    pts = np.column_stack([(cc + 0.5) * 0.1, (rr + 0.5) * 0.1, np.zeros(len(rr))])
    tops = np.column_stack([(np.nonzero(rim)[1] + 0.5) * 0.1, (np.nonzero(rim)[0] + 0.5) * 0.1])
    # Rim cells get two extra points, 0.25 and -0.25: their mean stays 0, their span is 0.5 m.
    extra = np.vstack([np.column_stack([tops, np.full(len(tops), z)]) for z in (0.25, -0.25)])
    allp = np.vstack([pts, extra])
    m = fuse_points(allp, np.full(len(allp), 0.01), MAP)
    guarded = fill_gaps(m)
    assert np.all(guarded.source[gap] == UNKNOWN)
    residual_only = fill_gaps(m, step_guard_m=np.inf)
    assert np.all(residual_only.source[gap] == FILLED)
