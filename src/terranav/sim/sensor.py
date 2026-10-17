"""Depth sensor model: ideal ray-cast -> noisy, partially invalid DepthFrame.

Pipeline, applied per pixel in this order:
  1. depth_z_ideal = range_to_depth(range_t). Ray misses (NaN) are flagged MISS.
  2. GRAZING: incidence angle between the reversed ray direction and the surface normal;
     invalid if > cfg.grazing_cutoff_rad. Purely geometric, so it uses the ideal hit.
  3. Noise: depth_z = depth_z_ideal + N(0, sigma), sigma = s0 + s1 * depth_z_ideal^2. Axial
     noise on depth_z (what a structured-light/stereo sensor reports), not on range_t.
  4. BEYOND_MAX: depth_z > cfg.max_depth_z_m, tested on the NOISY depth because the real
     sensor can only cut on what it measured. capture() therefore ray-casts a few sigma past
     the limit so that near-limit hits are still available to this test.
  5. DROPOUT: Bernoulli(cfg.dropout_prob) per pixel, among pixels still valid after 1-4.
  6. Every flagged pixel -> NaN; depth_z cast to float32 only at the very end.
GRAZING and BEYOND_MAX are independent properties of a hit and may both be set; MISS
excludes all others; DROPOUT is only set on otherwise-valid pixels.

Randomness: all random numbers for the full image are drawn in a fixed order (one standard
normal per pixel, then one uniform per pixel) regardless of which pixels are valid, so the
validity masks never shift the random stream. No RNG is created here: callers pass a
per-frame generator, derived as np.random.default_rng((scenario_seed, frame_index)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terranav.sim.raycast import RaycastResult, cast_frame, range_to_depth, unit_rays_cam
from terranav.types import CameraIntrinsics, DepthFrame, SensorConfig, World

# invalid_reason bits (uint8 bitmask).
MISS = 1
BEYOND_MAX = 2
GRAZING = 4
DROPOUT = 8

# capture() ray-casts this many noise sigmas (evaluated at max depth) past max_depth_z_m,
# so the noisy BEYOND_MAX cut sees hits whose ideal depth is just over the limit.
RAYCAST_MARGIN_SIGMAS = 6.0


@dataclass(frozen=True, slots=True, eq=False)
class SensorDebug:
    """Simulator-only metadata for one frame (a real sensor would not provide this).

    depth_z_ideal: (H, W) float64 noise-free camera-frame depth, metres; NaN = ray miss.
    invalid_reason: (H, W) uint8 bitmask of MISS | BEYOND_MAX | GRAZING | DROPOUT; 0 = valid.
    """

    depth_z_ideal: np.ndarray
    invalid_reason: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype in (("depth_z_ideal", np.float64), ("invalid_reason", np.uint8)):
            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def noise_sigma(depth_z: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Axial noise std (metres) at camera depth depth_z (metres): s0 + s1 * depth_z^2."""
    return cfg.noise_s0_m + cfg.noise_s1_per_m * np.square(depth_z)


def simulate_depth(
    ray: RaycastResult,
    intrinsics: CameraIntrinsics,
    T_world_cam: np.ndarray,
    cfg: SensorConfig,
    rng: np.random.Generator,
) -> tuple[DepthFrame, SensorDebug]:
    """Apply the sensor model (see module docstring) to an ideal ray-cast.

    Args:
        ray: cast_frame output for this camera, arrays (H, W) indexed [v, u].
        intrinsics: camera intrinsics matching ray's (H, W).
        T_world_cam: (4,4) camera-optical -> world transform used for the cast.
        cfg: sensor parameters (metres, radians).
        rng: per-frame generator, np.random.default_rng((scenario_seed, frame_index)).

    Returns:
        (DepthFrame with float32 depth_z, NaN = invalid; SensorDebug).
    """
    H, W = intrinsics.height, intrinsics.width
    if ray.range_t.shape != (H, W):
        raise ValueError(f"ray shape {ray.range_t.shape} != intrinsics (H, W) {(H, W)}")
    T = np.asarray(T_world_cam, dtype=np.float64)

    # Fixed draw order for the whole image, independent of validity.
    std_normal = rng.standard_normal((H, W))
    uniform = rng.random((H, W))

    # 1. Ideal depth.
    depth_ideal = range_to_depth(ray.range_t, intrinsics)
    hit = np.isfinite(depth_ideal)
    reason = np.where(hit, 0, MISS).astype(np.uint8)

    # 2. Grazing: cos(incidence) = (-d_world) . n.
    dirs_world = unit_rays_cam(intrinsics) @ T[:3, :3].T
    cos_inc = -np.einsum("vuk,vuk->vu", dirs_world, np.where(hit[..., None], ray.normals_world, 0))
    incidence = np.arccos(np.clip(cos_inc, -1.0, 1.0))
    reason[hit & (incidence > cfg.grazing_cutoff_rad)] |= GRAZING

    # 3. Axial noise on depth_z, sigma from the ideal depth.
    depth = np.where(hit, depth_ideal + std_normal * noise_sigma(depth_ideal, cfg), np.nan)

    # 4. Sensor range cut on the noisy depth.
    reason[hit & (depth > cfg.max_depth_z_m)] |= BEYOND_MAX

    # 5. Dropout among pixels still valid.
    reason[(reason == 0) & (uniform < cfg.dropout_prob)] |= DROPOUT

    # 6. Invalidate, then cast.
    depth_z = np.where(reason == 0, depth, np.nan).astype(np.float32)
    frame = DepthFrame(depth_z=depth_z, intrinsics=intrinsics, T_world_cam=T)
    return frame, SensorDebug(depth_z_ideal=depth_ideal, invalid_reason=reason)


def capture(
    world: World, T_world_cam: np.ndarray, cfg: SensorConfig, rng: np.random.Generator
) -> tuple[DepthFrame, SensorDebug]:
    """Ray-cast `world` from T_world_cam with cfg.intrinsics(), then simulate_depth.

    The cast extends RAYCAST_MARGIN_SIGMAS noise sigmas past cfg.max_depth_z_m (see module
    docstring, step 4). rng: per-frame generator, np.random.default_rng((seed, frame_index)).
    """
    intrinsics = cfg.intrinsics()
    margin = RAYCAST_MARGIN_SIGMAS * float(noise_sigma(cfg.max_depth_z_m, cfg))
    ray = cast_frame(world, T_world_cam, intrinsics, cfg.max_depth_z_m + margin)
    return simulate_depth(ray, intrinsics, T_world_cam, cfg, rng)
