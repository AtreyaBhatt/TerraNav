"""Ideal (noise-free) depth-camera ray-caster against a heightfield plus axis-aligned boxes.

Frames: rays are generated in the camera optical frame (x right, y down, z forward) and
rotated to world (x forward, y left, z up) with T_world_cam (docs/frames.md). Ray
directions in world are UNIT vectors, so the ray parameter is the Euclidean range range_t.
depth_z = range_t * (unit camera ray).z. Noise, dropout and grazing invalidation are NOT
applied here (sensor model, Step 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from terranav.frames import pixel_rays_cam
from terranav.types import Box, CameraIntrinsics, GridSpec, World

HIT_NONE, HIT_TERRAIN, HIT_BOX = 0, 1, 2

# Terrain march step as a fraction of the grid resolution (<= 1/2 per spec).
MARCH_STEP_FRACTION = 0.5
BISECTION_STEPS = 12
# Rays per terrain-march chunk: memory ~ RAY_CHUNK * n_steps * 8 B per temporary array.
RAY_CHUNK = 2048


@dataclass(frozen=True, slots=True, eq=False)
class RaycastResult:
    """Per-pixel ray-cast output, every array indexed [v, u] and read-only.

    range_t: (H, W) float64 Euclidean range along the unit ray, metres; NaN = miss.
    points_world: (H, W, 3) float64 hit points in world, metres; NaN = miss.
    normals_world: (H, W, 3) float64 unit surface normals in world; NaN = miss.
    hit_kind: (H, W) int8, HIT_NONE / HIT_TERRAIN / HIT_BOX.
    """

    range_t: np.ndarray
    points_world: np.ndarray
    normals_world: np.ndarray
    hit_kind: np.ndarray

    def __post_init__(self) -> None:
        for name in ("range_t", "points_world", "normals_world", "hit_kind"):
            arr = np.array(getattr(self, name), copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def unit_rays_cam(intrinsics: CameraIntrinsics) -> np.ndarray:
    """(H, W, 3) float64 unit ray directions in the camera optical frame."""
    rays = pixel_rays_cam(intrinsics)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def range_to_depth(range_t: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert (H, W) Euclidean range_t to camera-frame depth_z (both metres; NaN kept).

    depth_z = range_t * z-component of the unit camera ray (= range_t / ||d_cam||).
    """
    return np.asarray(range_t, dtype=np.float64) * unit_rays_cam(intrinsics)[..., 2]


def _rowcol(grid: GridSpec, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (row, col) for map_coordinates under the cell-centre convention."""
    col = (x - grid.origin[0]) / grid.resolution - 0.5
    row = (y - grid.origin[1]) / grid.resolution - 0.5
    return row, col


def _inside(grid: GridSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, y0 = grid.origin
    return (x >= x0) & (x <= x0 + grid.size_x) & (y >= y0) & (y <= y0 + grid.size_y)


def _bilinear(field: np.ndarray, grid: GridSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a (ny, nx) grid field at world (x, y). Edge half-cells clamp."""
    row, col = _rowcol(grid, x, y)
    coords = np.stack([np.ravel(row), np.ravel(col)])
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest").reshape(np.shape(x))


def _terrain_hits(
    hf: np.ndarray, grid: GridSpec, origin: np.ndarray, dirs: np.ndarray, t_max: np.ndarray
) -> np.ndarray:
    """Range of the first terrain crossing per ray (inf = none). dirs (N, 3) unit, world.

    Marches t = 0, step, 2*step, ... up to t_max + step, finds the first sample pair with
    (ray_z - h) going from > 0 to <= 0 (both samples inside the map), refines the bracket by
    bisection and finishes with one linear (secant) interpolation inside the final bracket.
    """
    step = MARCH_STEP_FRACTION * grid.resolution
    t_hit = np.full(len(dirs), np.inf)
    for s in range(0, len(dirs), RAY_CHUNK):
        d, tm = dirs[s : s + RAY_CHUNK], t_max[s : s + RAY_CHUNK]
        t = np.arange(int(np.ceil(tm.max() / step)) + 2) * step  # (K,)
        p = origin[None, None, :] + t[None, :, None] * d[:, None, :]  # (n, K, 3)
        inside = _inside(grid, p[..., 0], p[..., 1]) & (t[None, :] <= tm[:, None] + step)
        f = p[..., 2] - _bilinear(hf, grid, p[..., 0], p[..., 1])
        cross = inside[:, :-1] & inside[:, 1:] & (f[:, :-1] > 0) & (f[:, 1:] <= 0)
        has = cross.any(axis=1)
        if not has.any():
            continue
        k = np.argmax(cross[has], axis=1)
        dh, rows = d[has], np.flatnonzero(has)
        lo, hi = t[k], t[k + 1]
        f_lo, f_hi = f[rows, k], f[rows, k + 1]
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            q = origin + mid[:, None] * dh
            f_mid = q[:, 2] - _bilinear(hf, grid, q[:, 0], q[:, 1])
            above = f_mid > 0
            lo, f_lo = np.where(above, mid, lo), np.where(above, f_mid, f_lo)
            hi, f_hi = np.where(above, hi, mid), np.where(above, f_hi, f_mid)
        # f_lo > 0 >= f_hi, so the denominator is > 0.
        t_hit[s + rows] = lo + (hi - lo) * f_lo / (f_lo - f_hi)
    return t_hit


def _box_hits(box: Box, origin: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slab test. Returns (t_enter, normal) per ray; t_enter = inf where not hit at t > 0.

    Zero direction components are handled with explicit masks: such a ray is inside that
    slab for all t iff the origin lies within [min, max] (boundaries inclusive), otherwise
    it misses. No division by zero is ever evaluated, so no NaN/inf leaks.
    """
    lo, hi = np.asarray(box.min_xyz), np.asarray(box.max_xyz)
    moving = dirs != 0.0  # (N, 3)
    safe = np.where(moving, dirs, 1.0)
    t1, t2 = (lo - origin) / safe, (hi - origin) / safe
    in_slab = (origin >= lo) & (origin <= hi)  # (3,), for rays parallel to an axis
    t_near = np.where(moving, np.minimum(t1, t2), np.where(in_slab, -np.inf, np.inf))
    t_far = np.where(moving, np.maximum(t1, t2), np.where(in_slab, np.inf, -np.inf))
    axis = np.argmax(t_near, axis=1)
    t_enter = np.take_along_axis(t_near, axis[:, None], axis=1)[:, 0]
    t_exit = t_far.min(axis=1)
    hit = (t_enter <= t_exit) & (t_enter > 0.0)  # origin inside the box: not a hit
    normal = np.zeros_like(dirs)
    sign = np.sign(np.take_along_axis(dirs, axis[:, None], axis=1)[:, 0])
    normal[np.arange(len(dirs)), axis] = -sign  # entered through the face facing the ray
    return np.where(hit, t_enter, np.inf), normal


def cast_frame(
    world: World,
    T_world_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth_z_m: float,
) -> RaycastResult:
    """Cast one ray per pixel from the camera at T_world_cam into `world`.

    The nearest positive hit (terrain or box) wins; hits with depth_z > max_depth_z_m are
    then discarded (range_t NaN, hit_kind HIT_NONE). Raises ValueError if the camera origin
    is inside the map and at or below the terrain surface, or inside (or on) any box.
    """
    T = np.asarray(T_world_cam, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_world_cam must have shape (4,4), got {T.shape}")
    grid, hf = world.grid, world.heightfield
    origin = T[:3, 3]
    ox, oy = np.array([origin[0]]), np.array([origin[1]])
    if _inside(grid, ox, oy)[0] and origin[2] <= _bilinear(hf, grid, ox, oy)[0]:
        raise ValueError(f"camera origin {origin} is at or below the terrain")
    for box in world.boxes:
        if np.all((origin >= box.min_xyz) & (origin <= box.max_xyz)):
            raise ValueError(f"camera origin {origin} is inside (or on) box {box}")

    H, W = intrinsics.height, intrinsics.width
    rays_cam = unit_rays_cam(intrinsics).reshape(-1, 3)
    dirs = rays_cam @ T[:3, :3].T  # unit, world frame
    t_max = max_depth_z_m / rays_cam[:, 2]  # depth_z = t * z_cam  <=>  t = depth_z / z_cam

    t_best = _terrain_hits(hf, grid, origin, dirs, t_max)
    kind = np.where(np.isfinite(t_best), HIT_TERRAIN, HIT_NONE).astype(np.int8)
    normals = np.full_like(dirs, np.nan)
    for box in world.boxes:
        t_box, n_box = _box_hits(box, origin, dirs)
        nearer = t_box < t_best
        t_best = np.where(nearer, t_box, t_best)
        kind[nearer] = HIT_BOX
        normals[nearer] = n_box[nearer]

    keep = np.isfinite(t_best) & (t_best * rays_cam[:, 2] <= max_depth_z_m)
    kind[~keep] = HIT_NONE
    range_t = np.where(keep, t_best, np.nan)
    points = origin + range_t[:, None] * dirs

    terrain = kind == HIT_TERRAIN
    if terrain.any():
        gy, gx = np.gradient(hf, grid.resolution)  # d/drow = d/dy, d/dcol = d/dx
        px, py = points[terrain, 0], points[terrain, 1]
        n = np.stack(
            [-_bilinear(gx, grid, px, py), -_bilinear(gy, grid, px, py), np.ones(len(px))],
            axis=1,
        )
        normals[terrain] = n / np.linalg.norm(n, axis=1, keepdims=True)
    normals[~keep] = np.nan

    return RaycastResult(
        range_t=range_t.reshape(H, W),
        points_world=points.reshape(H, W, 3),
        normals_world=normals.reshape(H, W, 3),
        hit_kind=kind.reshape(H, W),
    )
