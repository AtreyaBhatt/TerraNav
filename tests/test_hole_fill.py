"""Gap filling, footprint marking and ground-truth elevation tests (hand-derived)."""

from __future__ import annotations

import numpy as np
import pytest

from terranav.perception.elevation_map import FILLED, FOOTPRINT, OBSERVED, UNKNOWN, fuse_points
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


def test_fill_rows_on_inclined_plane():
    # Observed every 3rd row (0, 3, ..., 117) on z = 0.2 x; rows in between are 0.2 m gaps,
    # within max_gap_m = 0.3. Rows 118-119 lie outside the last observed row (not a gap).
    rr, cc = np.meshgrid(np.arange(0, 120, 3), np.arange(120), indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    m = fill_gaps(map_with_cells(rows, cols, 0.2 * (cols + 0.5) * 0.1))
    gap_rows = np.array([r for r in range(117) if r % 3])
    assert np.all(m.source[gap_rows, :] == FILLED)
    assert np.all(m.count[gap_rows, :] == 0)
    x = (np.arange(120) + 0.5) * 0.1
    np.testing.assert_allclose(
        m.z_mean[gap_rows, :], np.broadcast_to(0.2 * x, (len(gap_rows), 120)), atol=0.01
    )
    # z_min / z_max of a filled cell bracket its neighbours: the plane rises 0.02 m per column.
    assert np.all(m.z_min[gap_rows, :] <= m.z_mean[gap_rows, :] + 1e-12)
    assert np.all(m.z_max[gap_rows, :] >= m.z_mean[gap_rows, :] - 1e-12)
    assert np.all(m.source[118:, :] == UNKNOWN)
    assert np.all(m.source[::3, :][:40] == OBSERVED)


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
    # unobserved. Any window touching the rim spans 0.5 m > STEP_GUARD_M, so neither the
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
    unguarded = fill_gaps(map_with_cells(rows, cols, z), step_guard_m=np.inf)
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
