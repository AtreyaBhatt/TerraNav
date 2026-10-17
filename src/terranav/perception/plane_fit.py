"""Windowed weighted plane fit over an ElevationMap, fully vectorised.

For every cell (known or not) a plane z = a*dx + b*dy + c is fitted to the known cells whose
centres lie within radius_m, with (dx, dy) the neighbour's offset from the cell in metres and
weights 1 / z_mean_var. Moment sums come from ndimage correlations of per-cell arrays
(w, w*z, w*z^2) with offset kernels (1, dx, dy, dx^2, dx*dy, dy^2) over the disk; the 3x3
normal equations are then solved for all cells at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from terranav.perception.elevation_map import UNKNOWN, ElevationMap

PLANE_FIT_RADIUS_M = 0.3
MIN_SUPPORT = 8  # ILLUSTRATIVE: fewer known cells in the disk -> no fit (NaN)
# Rejects (near-)collinear supports: det of the normal matrix divided by the weight sum,
# in cell units (a full radius-3 disk gives ~16).
MIN_NORMALISED_DET = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class PlaneFit:
    """Per-cell plane fit, all arrays (ny, nx), read-only; NaN where no valid fit.

    a, b: dz/dx, dz/dy (m/m, world frame). c: fitted height at the cell centre (m).
    residual_m: weighted RMS residual, dof-corrected (sqrt(SSE / sum_w * n / (n - 3))), so it
        estimates the per-cell deviation from the plane.
    fit_noise_var: support / sum_w (m^2) = harmonic mean of the fitted cells' z_mean_var, the
        residual^2 expected from measurement noise alone.
    support: int32 number of known cells in the disk (always filled in).
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    residual_m: np.ndarray
    fit_noise_var: np.ndarray
    support: np.ndarray

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "residual_m", "fit_noise_var", "support"):
            dtype = np.int32 if name == "support" else np.float64
            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def disk_offsets(radius_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(disk mask, dx, dy) kernels of shape (2r+1, 2r+1); dx along columns, dy along rows."""
    r = radius_cells
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1].astype(float)
    return (dx**2 + dy**2 <= r**2 + 1e-9), dx, dy


def plane_fit(
    emap: ElevationMap,
    radius_m: float = PLANE_FIT_RADIUS_M,
    mask: np.ndarray | None = None,
    min_support: int = MIN_SUPPORT,
) -> PlaneFit:
    """Fit planes to the cells in `mask` (default: source != UNKNOWN) around every cell."""
    res = emap.grid.resolution
    if mask is None:
        mask = emap.source != UNKNOWN
    use = mask & np.isfinite(emap.z_mean) & (emap.z_mean_var > 0)
    w = np.where(use, 1.0 / np.where(use, emap.z_mean_var, 1.0), 0.0)
    z_ref = float(np.sum(w * np.where(use, emap.z_mean, 0.0)) / max(np.sum(w), 1e-300))
    z = np.where(use, emap.z_mean - z_ref, 0.0)  # centred heights reduce cancellation

    disk, dx, dy = disk_offsets(math.floor(radius_m / res + 1e-9))

    def corr(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return ndimage.correlate(arr, np.where(disk, kernel, 0.0), mode="constant", cval=0.0)

    one = np.ones_like(dx)
    support = np.rint(corr(use.astype(float), one)).astype(np.int32)
    s = corr(w, one)
    s = np.where(s > 0, s, 1.0)  # placeholder where empty; masked out below
    sx, sy = corr(w, dx) / s, corr(w, dy) / s
    sxx, sxy, syy = corr(w, dx * dx) / s, corr(w, dx * dy) / s, corr(w, dy * dy) / s
    wz = w * z
    sz, sxz, syz = corr(wz, one) / s, corr(wz, dx) / s, corr(wz, dy) / s
    szz = corr(wz * z, one) / s

    m = np.stack(
        [
            np.stack([sxx, sxy, sx], -1),
            np.stack([sxy, syy, sy], -1),
            np.stack([sx, sy, np.ones_like(s)], -1),
        ],
        -2,
    )  # (ny, nx, 3, 3), normalised by the weight sum, offsets in cells
    rhs = np.stack([sxz, syz, sz], -1)
    ok = (support >= min_support) & (np.linalg.det(m) > MIN_NORMALISED_DET)
    sol = np.full(rhs.shape, np.nan)
    if ok.any():
        sol[ok] = np.linalg.solve(m[ok], rhs[ok][..., None])[..., 0]
    a_c, b_c, c_c = sol[..., 0], sol[..., 1], sol[..., 2]
    # SSE / sum_w = E_w[z^2] - 2 p.r + p.M.p for parameters p = (a, b, c).
    sse = (
        szz
        - 2 * (a_c * sxz + b_c * syz + c_c * sz)
        + np.einsum("...i,...ij,...j->...", sol, m, sol)
    )
    n = np.maximum(support, 4).astype(float)
    residual = np.sqrt(np.maximum(sse, 0.0) * n / (n - 3))
    fit_noise_var = np.where(ok, support / s, np.nan)
    return PlaneFit(
        a=a_c / res,
        b=b_c / res,
        c=c_c + z_ref,
        residual_m=residual,
        fit_noise_var=fit_noise_var,
        support=support,
    )
