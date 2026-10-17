"""Offscreen 3D scene render (PyVista): terrain, boxes, camera frusta, fused point cloud.

World frame throughout: x forward, y left, z up, metres (docs/frames.md). Camera views are
fixed so renders are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

from terranav.frames import transform_points
from terranav.sim.world import bilinear_sample
from terranav.types import CameraIntrinsics, Scenario, SensorConfig, World

TERRAIN_MESH_STEP_M = 0.1  # terrain mesh is block-averaged to this spacing for speed
TERRAIN_OPACITY = 0.6
FRUSTUM_DEPTH_M = 0.4  # drawn frustum length along the optical axis
AXIS_ARROW_M = 0.6
MARKER_RADIUS_M = 0.15
WINDOW_SIZE = (1600, 1000)

# Fixed view directions: (camera position, focal point, view-up), world metres. Each view is
# then fitted to the scene bounds with reset_camera(), which keeps the direction, so renders
# are reproducible for a given scene.
OBLIQUE_VIEW = [(-4.0, -7.0, 9.0), (6.0, 6.0, 0.0), (0.0, 0.0, 1.0)]
TOP_VIEW = [(6.0, 6.0, 30.0), (6.0, 6.0, 0.0), (0.0, 1.0, 0.0)]  # +x right, +y up
OBLIQUE_ZOOM, TOP_ZOOM = 1.6, 1.3  # reset_camera() leaves a generous margin


def frustum_corners_world(
    T_world_cam: np.ndarray, intrinsics: CameraIntrinsics, depth_z_m: float
) -> np.ndarray:
    """(4, 3) world points of the image's outer corners at camera depth depth_z_m.

    Corners are the outer pixel edges (u, v) = (-0.5, -0.5), (W-0.5, -0.5), (W-0.5, H-0.5),
    (-0.5, H-0.5): top-left, top-right, bottom-right, bottom-left (pixel centres at integers).
    """
    W, H = intrinsics.width, intrinsics.height
    u = np.array([-0.5, W - 0.5, W - 0.5, -0.5])
    v = np.array([-0.5, -0.5, H - 0.5, H - 0.5])
    rays = np.column_stack(
        [(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, np.ones(4)]
    )
    return transform_points(T_world_cam, depth_z_m * rays)


def terrain_mesh(world: World, step_m: float = TERRAIN_MESH_STEP_M) -> pv.StructuredGrid:
    """Terrain surface with "height" point data, block-averaged to step_m cells."""
    grid, hf = world.grid, world.heightfield
    f = max(1, round(step_m / grid.resolution))
    ny, nx = (grid.ny // f) * f, (grid.nx // f) * f
    coarse = hf[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))
    xs = grid.origin[0] + (np.arange(nx // f) + 0.5) * f * grid.resolution
    ys = grid.origin[1] + (np.arange(ny // f) + 0.5) * f * grid.resolution
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # rows = y, cols = x, like the heightfield
    mesh = pv.StructuredGrid(X, Y, coarse)
    mesh["height"] = mesh.points[:, 2]
    return mesh


def _frustum_lines(T: np.ndarray, intrinsics: CameraIntrinsics) -> pv.PolyData:
    corners = frustum_corners_world(T, intrinsics, FRUSTUM_DEPTH_M)
    points = np.vstack([T[:3, 3], corners])  # index 0 = apex (camera centre)
    # Four apex->corner edges plus the corner rectangle; VTK line cells are [2, i, j].
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return pv.PolyData(points, lines=np.array([[2, i, j] for i, j in edges]).ravel())


def _populate(
    pl: pv.Plotter,
    scenario: Scenario,
    poses: Sequence[np.ndarray],
    points_list: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
) -> None:
    world = scenario.world
    colours = matplotlib.colormaps["tab10"]
    pl.set_background("white")
    # Light-to-dark greys so the frame-coloured points stand out; base height is light grey.
    z_lo, z_hi = float(world.heightfield.min()), float(world.heightfield.max())
    pl.add_mesh(
        terrain_mesh(world),
        scalars="height",
        cmap="Greys",
        clim=(z_lo - 0.1, z_hi + 0.4),
        opacity=TERRAIN_OPACITY,
        scalar_bar_args={
            "title": "terrain z [m]",
            "vertical": True,
            "position_x": 0.9,
            "position_y": 0.3,
            "height": 0.4,
            "width": 0.04,
        },
    )
    for b in world.boxes:
        lo, hi = b.min_xyz, b.max_xyz
        pl.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])), color="firebrick")

    k = len(points_list)
    frames = [i for i in range(k) if len(points_list[i])]
    if frames:
        cloud = pv.PolyData(np.vstack([points_list[i] for i in frames]))
        cloud["frame"] = np.concatenate([np.full(len(points_list[i]), float(i)) for i in frames])
        pl.add_mesh(
            cloud,
            scalars="frame",
            cmap=ListedColormap([colours(i % 10) for i in range(k)]),
            clim=(-0.5, k - 0.5),
            point_size=3,
            show_scalar_bar=False,
        )
    for i, T in enumerate(poses):
        colour = colours(i % 10)[:3]
        pl.add_mesh(_frustum_lines(T, intrinsics), color=colour, line_width=3)
        arrow = pv.Arrow(start=T[:3, 3], direction=T[:3, 2], scale=AXIS_ARROW_M)
        pl.add_mesh(arrow, color=colour)

    for (x, y), colour in ((scenario.start_xy, "limegreen"), (scenario.goal_xy, "gold")):
        z = float(bilinear_sample(world.heightfield, world.grid, x, y)) + MARKER_RADIUS_M
        pl.add_mesh(pv.Sphere(radius=MARKER_RADIUS_M, center=(x, y, z)), color=colour)
    if poses:
        legend = [[f"frame {i}", colours(i % 10)[:3]] for i in range(len(poses))]
        pl.add_legend(legend, bcolor="white", face="rectangle", loc="upper right", size=(0.1, 0.15))
    pl.add_axes()


def _snapshot(pl: pv.Plotter, view: list, path: Path, parallel: bool, zoom: float) -> None:
    if parallel:
        pl.enable_parallel_projection()
    pl.camera_position = view
    pl.reset_camera()  # fit the scene, keeping the view direction
    pl.camera.zoom(zoom)
    pl.render()
    pl.screenshot(str(path))


def render_scene(
    scenario: Scenario,
    poses: Sequence[np.ndarray],
    points_list: Sequence[np.ndarray],
    out_path: str | Path,
    intrinsics: CameraIntrinsics | None = None,
    window_size: tuple[int, int] = WINDOW_SIZE,
    show: bool = False,
) -> tuple[Path, Path]:
    """Render the scene offscreen to out_path (oblique 3/4 view) and <stem>_top<suffix>.

    poses: (4,4) T_world_cam per frame; points_list: (N_i, 3) world points per frame,
    coloured by frame index. Returns (oblique_path, top_path). With show=True an interactive
    window opens afterwards (requires a display).
    """
    if len(poses) != len(points_list):
        raise ValueError("poses and points_list must have the same length")
    intrinsics = intrinsics if intrinsics is not None else SensorConfig().intrinsics()
    out_path = Path(out_path)
    top_path = out_path.with_name(f"{out_path.stem}_top{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    _populate(pl, scenario, poses, points_list, intrinsics)
    _snapshot(pl, OBLIQUE_VIEW, out_path, parallel=False, zoom=OBLIQUE_ZOOM)
    _snapshot(pl, TOP_VIEW, top_path, parallel=True, zoom=TOP_ZOOM)
    pl.close()

    if show:
        viewer = pv.Plotter(window_size=list(window_size))
        _populate(viewer, scenario, poses, points_list, intrinsics)
        viewer.camera_position = OBLIQUE_VIEW
        viewer.reset_camera()
        viewer.show()
    return out_path, top_path
