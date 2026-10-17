"""Frame conventions: rotations, camera pose, and pixel rays.

See docs/frames.md for the authoritative description. Summary:

- World: x forward, y left, z up (metres).
- Camera optical: x right, y down, z forward. Pixel u right, v down.
- R_wc = Rz(yaw) @ Ry(pitch_down) @ R_BASE maps optical-frame vectors to world.

All angles are in radians. All arrays returned here are float64.
"""

from __future__ import annotations

import numpy as np

from terranav.types import CameraIntrinsics

# Optical -> world rotation at yaw=0, pitch_down=0.
# Columns are the optical axes expressed in world: x_c -> -y_w, y_c -> -z_w, z_c -> +x_w.
R_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def rot_y(t: float) -> np.ndarray:
    """Rotation by angle t (radians) about the +y axis. Returns (3,3) float64.

    Ry(t) = [[c,0,s],[0,1,0],[-s,0,c]]. For positive t this rotates +x toward -z.
    """
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(t: float) -> np.ndarray:
    """Rotation by angle t (radians) about the +z axis. Returns (3,3) float64.

    Rz(t) = [[c,-s,0],[s,c,0],[0,0,1]]. For positive t this rotates +x toward +y.
    """
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def R_wc(yaw: float, pitch_down: float) -> np.ndarray:
    """Rotation mapping camera-optical-frame vectors to world-frame vectors.

    Args:
        yaw: heading in radians about world +z; 0 looks along world +x, +pi/2 along +y.
        pitch_down: tilt in radians; positive tilts the optical axis toward world -z.

    Returns:
        (3,3) float64 rotation R_wc = Rz(yaw) @ Ry(pitch_down) @ R_BASE.
    """
    return rot_z(yaw) @ rot_y(pitch_down) @ R_BASE


def T_world_cam(position_xyz: np.ndarray, yaw: float, pitch_down: float) -> np.ndarray:
    """Homogeneous transform mapping camera-optical-frame points to world points.

    Args:
        position_xyz: camera optical centre in world coordinates, metres, shape (3,).
        yaw: radians, see R_wc.
        pitch_down: radians, see R_wc.

    Returns:
        (4,4) float64 matrix [[R_wc, p], [0, 0, 0, 1]].
    """
    p = np.asarray(position_xyz, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"position_xyz must have shape (3,), got {p.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc(yaw, pitch_down)
    T[:3, 3] = p
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a (4,4) homogeneous transform to (N,3) points.

    Frames are whatever T maps between (e.g. T_world_cam: camera points in, world out).
    Returns (N,3) float64.
    """
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4,4), got {T.shape}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must have shape (N,3), got {pts.shape}")
    return pts @ T[:3, :3].T + T[:3, 3]


def pixel_rays_cam(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Per-pixel ray directions in the camera optical frame, scaled so z = 1.

    Pixel centres are at integer (u, v). For pixel (u, v):
        d_cam = [(u - cx) / fx, (v - cy) / fy, 1]
    Because z = 1, a point at depth depth_z along this ray is depth_z * d_cam, and its
    Euclidean range is range_t = depth_z * ||d_cam||.

    Returns:
        (H, W, 3) float64 array indexed [v, u, :]. Rays are NOT unit length.
    """
    u = np.arange(intrinsics.width, dtype=np.float64)
    v = np.arange(intrinsics.height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # each (H, W)
    rays = np.empty((intrinsics.height, intrinsics.width, 3), dtype=np.float64)
    rays[..., 0] = (uu - intrinsics.cx) / intrinsics.fx
    rays[..., 1] = (vv - intrinsics.cy) / intrinsics.fy
    rays[..., 2] = 1.0
    return rays
