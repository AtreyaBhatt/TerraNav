"""Backprojection: DepthFrame (camera-frame depth_z image) -> world-frame point cloud."""

from __future__ import annotations

import numpy as np

from terranav.frames import pixel_rays_cam, transform_points
from terranav.types import DepthFrame


def backproject(frame: DepthFrame) -> tuple[np.ndarray, np.ndarray]:
    """Lift every valid (non-NaN) pixel of `frame` to a world-frame 3D point.

    For pixel (u, v) with camera-frame depth depth_z (metres, NOT Euclidean range):
        p_cam = depth_z * [(u - cx)/fx, (v - cy)/fy, 1]
        p_world = T_world_cam @ p_cam
    Computed in float64 from the stored float32 depth.

    Returns:
        points_world: (N, 3) float64, metres, world frame.
        pixel_vu: (N, 2) int64 (v, u) pixel index of each point, row-major order.
    """
    depth = frame.depth_z.astype(np.float64)
    v, u = np.nonzero(~np.isnan(depth))
    p_cam = depth[v, u, None] * pixel_rays_cam(frame.intrinsics)[v, u]
    points_world = transform_points(frame.T_world_cam, p_cam)
    return points_world, np.stack([v, u], axis=1).astype(np.int64)
