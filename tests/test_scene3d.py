"""3D scene helpers: hand-derived frustum geometry and an offscreen render smoke test."""

from __future__ import annotations

import math

import numpy as np

from terranav.frames import T_world_cam
from terranav.sim.scenario import generate_scenario, scan_poses
from terranav.types import CameraIntrinsics
from terranav.viz.scene3d import frustum_corners_world, render_scene


def test_frustum_corners_orientation():
    # 4 x 2 image, hfov 90 deg: fx = fy = 2, cx = 1.5, cy = 0.5. Outer corner (u, v) =
    # (-0.5, -0.5) has camera ray (-1, -0.5, 1). With yaw 0, pitch 0 at the origin, camera
    # x (right) -> world -y, y (down) -> world -z, z -> world +x, so at depth 2 the top-left
    # corner is (2, +2, +1): left (+y) and up (+z), as seen from behind the camera.
    intr = CameraIntrinsics.from_hfov(4, 2, math.radians(90.0))
    T = T_world_cam(np.zeros(3), 0.0, 0.0)
    corners = frustum_corners_world(T, intr, 2.0)
    expected = [[2.0, 2.0, 1.0], [2.0, -2.0, 1.0], [2.0, -2.0, -1.0], [2.0, 2.0, -1.0]]
    np.testing.assert_allclose(corners, expected, atol=1e-12)


def test_render_scene_writes_both_views(tmp_path):
    sc = generate_scenario("mixed", 0)
    poses = scan_poses(sc, k=3)
    rng = np.random.default_rng(0)
    points = [rng.uniform(0, 12, size=(50, 3)) for _ in poses]
    oblique, top = render_scene(sc, poses, points, tmp_path / "scene.png", window_size=(320, 200))
    assert oblique == tmp_path / "scene.png" and top == tmp_path / "scene_top.png"
    assert oblique.stat().st_size > 0 and top.stat().st_size > 0
