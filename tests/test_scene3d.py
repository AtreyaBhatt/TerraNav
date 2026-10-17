"""3D scene helpers: hand-derived frustum geometry and an offscreen render smoke test."""

from __future__ import annotations

import math

import numpy as np
import pytest
import pyvista as pv

from terranav.frames import T_world_cam
from terranav.sim.scenario import generate_scenario, scan_poses
from terranav.types import CameraIntrinsics, SensorConfig
from terranav.viz.scene3d import _populate, frustum_corners_world, render_scene


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


def test_every_pose_gets_its_own_actors(offscreen_gl):
    sc = generate_scenario("mixed", 0)
    poses = scan_poses(sc)
    pl = pv.Plotter(off_screen=True, window_size=[64, 64])
    points = [np.zeros((0, 3)) for _ in poses]
    _populate(pl, sc, poses, points, SensorConfig().intrinsics())
    names = set(pl.actors)
    assert len(poses) == 7
    for i in range(len(poses)):
        assert f"frustum_{i}" in names and f"axis_{i}" in names
    assert {"terrain", "start", "goal"} <= names
    assert {f"box_{j}" for j in range(len(sc.world.boxes))} <= names
    pl.close()


def test_render_scene_writes_both_views(tmp_path, offscreen_gl):
    sc = generate_scenario("mixed", 0)
    poses = scan_poses(sc)
    rng = np.random.default_rng(0)
    points = [rng.uniform(0, 12, size=(50, 3)) for _ in poses]
    oblique, top = render_scene(sc, poses, points, tmp_path / "scene.png", window_size=(320, 200))
    assert oblique == tmp_path / "scene.png" and top == tmp_path / "scene_top.png"
    assert oblique.stat().st_size > 0 and top.stat().st_size > 0
