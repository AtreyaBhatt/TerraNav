"""Presentation 3D renders of one showcase scenario: a hero still and an animated replay.

World frame throughout: x forward, y left, z up, metres. The camera is pinned (derived from the
map extent only) so both renders are reproducible and share one view. The replay replays the
scan-once, plan-once pipeline's recorded products; it is not a live simulation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

from terranav.config import SimConfig
from terranav.perception.traversability import R_UNKNOWN
from terranav.sim.evaluate import sample_polyline, surface_z
from terranav.sim.showcase import (
    ShowcaseRun,
    path_prefix,
    path_slice,
    rough_exposure_along,
    rough_intervals,
)
from terranav.sim.world import coarsen_factor
from terranav.types import GridSpec, Scenario
from terranav.viz.figures import PLANNER_COLOURS, PLANNER_LABELS
from terranav.viz.scene3d import _frustum_lines, terrain_mesh

HERO_SIZE = (1920, 1080)
REPLAY_SIZE = (1280, 720)
BACKGROUND = "white"
INK = "#0b0b0b"
# Terrain cost: near-white (cost 0) to deep orange-brown (the scene's maximum GT cost, see
# cost_limits); lethal cells dark red (NaN colour).
COST_CMAP = LinearSegmentedColormap.from_list(
    "cost", ["#fdfaf5", "#f8cf98", "#ee9a45", "#cf6416", "#9a4310"]
)
LETHAL_COLOUR = "#5a0f0f"
NEUTRAL_TERRAIN = "#d9d6cf"
BOX_COLOUR = "#6b7480"
POINT_COLOUR = "#3b4a5a"
POSE_COLOUR = "#262626"
START_COLOUR, GOAL_COLOUR = "#2ca02c", "#ffd700"
# Stretches inside a rough patch: a brighter colour per planner, thicker tube, extra ambient light
PLANNER_BRIGHT = {"terranav": "#7dffc8", "baseline_occ": "#ff2d55", "oracle": "#8cc4ff"}
ROUGH_RADIUS_SCALE = 1.9
ROUGH_AMBIENT = 0.35
PATH_LIFT_M = 0.10  # tubes float this far above the surface so they are never buried
ORACLE_LIFT_M = 0.16  # Oracle's thin dashed tube is drawn above the others
TUBE_RADIUS_M = 0.06
ORACLE_RADIUS_M = 0.03
DASH_M, GAP_M = 0.25, 0.15
MARKER_RADIUS_M = 0.18
ROBOT_RADIUS_M = 0.3  # footprint disc of the replay robots
# Pinned oblique view: from behind-right of the map's start edge (x = 0 side), looking across.
CAMERA_OFFSET = (-0.62, -1.08, 0.95)  # (x, y, z) offset from the map centre, in map sizes
FOCAL_Z_M = -0.5


def camera_view(grid: GridSpec) -> list[tuple[float, float, float]]:
    """(position, focal point, view-up) of the pinned oblique view, metres."""
    size = max(grid.size_x, grid.size_y)
    cx, cy = grid.origin[0] + grid.size_x / 2, grid.origin[1] + grid.size_y / 2
    ox, oy, oz = (size * o for o in CAMERA_OFFSET)
    return [(cx + ox, cy + oy, oz), (cx, cy, FOCAL_Z_M), (0.0, 0.0, 1.0)]


def map_to_terrain_points(values: np.ndarray, map_grid: GridSpec, terrain: GridSpec) -> np.ndarray:
    """Map-grid cell values -> one value per terrain-mesh point (point i = terrain cell i)."""
    f = coarsen_factor(terrain, map_grid)
    fine = np.repeat(np.repeat(values, f, axis=0), f, axis=1)
    return fine.ravel(order="F")  # StructuredGrid points are in Fortran (x fastest) order


def cost_scalars(trav, map_grid: GridSpec, terrain: GridSpec) -> np.ndarray:
    """Traversability cost per terrain-mesh point; NaN (drawn as LETHAL_COLOUR) where lethal."""
    return map_to_terrain_points(np.where(trav.lethal, np.nan, trav.cost), map_grid, terrain)


def cost_limits(run: ShowcaseRun) -> tuple[float, float]:
    """(0, maximum GT traversability cost over non-lethal cells): the colour range of the scene."""
    gt = run.ground_truth
    top = float(np.max(np.where(gt.lethal, 0.0, gt.cost)))
    return 0.0, max(top, 1e-6)


def unknown_blocked_cells(run: ShowcaseRun, name: str, map_grid: GridSpec) -> int:
    """Cells under planner `name`'s path that are lethal on the perceived map only because they
    lie within the robot radius of an unknown cell (dilated R_UNKNOWN)."""
    from terranav.perception.traversability import TraversabilityConfig, dilate_lethal
    from terranav.planning.smooth import segment_cells

    path = run.paths_xy[name]
    if path is None:
        return 0
    unknown = (run.perceived.core_reason & R_UNKNOWN) != 0
    blocked = dilate_lethal(unknown, map_grid, TraversabilityConfig().robot_radius_m)
    r, c = map_grid.xy_to_rc(path[:, 0], path[:, 1])
    rc = np.column_stack([r, c]).astype(int)
    cells = {(rr, cc) for a, b in itertools.pairwise(rc) for rr, cc, _ in segment_cells(a, b)}
    return sum(bool(blocked[cell]) for cell in cells)


def path_points(scenario: Scenario, path_xy: np.ndarray, lift_m: float) -> np.ndarray:
    """(M, 3) densified path on the GT surface (terrain or box top) lifted by lift_m."""
    pts, _, _ = sample_polyline(np.asarray(path_xy, dtype=np.float64), 0.05)
    z = surface_z(scenario, pts[:, 0], pts[:, 1]) + lift_m
    return np.column_stack([pts, z])


def tube(points: np.ndarray, radius_m: float) -> pv.PolyData:
    return pv.lines_from_points(points).tube(radius=radius_m, n_sides=16, capping=True)


def dashed_tube(points: np.ndarray, radius_m: float) -> pv.PolyData:
    """Tube drawn only on [k (DASH_M + GAP_M), k (DASH_M + GAP_M) + DASH_M) of arc length."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    on = (s % (DASH_M + GAP_M)) < DASH_M
    pieces, start = [], None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        if (not flag or i == len(on) - 1) and start is not None:
            if i - start >= 1:
                pieces.append(tube(points[start : i + 1], radius_m))
            start = None
    return pv.merge(pieces) if pieces else pv.PolyData()


def path_meshes(
    scenario: Scenario,
    path_xy: np.ndarray,
    lift_m: float,
    radius_m: float,
    s_end_m: float | None = None,
    dashed: bool = False,
) -> tuple[pv.PolyData, pv.PolyData | None]:
    """(tube of the path's first s_end_m metres (all if None), thicker tube of the stretches of
    that part inside rough patches, or None). Rough stretches: rough_intervals (evaluator
    geometry from scenario.params)."""
    length = float(np.sum(np.hypot(*np.diff(path_xy, axis=0).T)))
    s_end = length if s_end_m is None else min(s_end_m, length)
    pts = path_points(scenario, path_prefix(path_xy, s_end), lift_m)
    base = dashed_tube(pts, radius_m) if dashed else tube(pts, radius_m)
    pieces = []
    for s0, s1 in rough_intervals(scenario, path_xy):
        if s0 >= s_end:
            continue
        stretch = path_points(scenario, path_slice(path_xy, s0, min(s1, s_end)), lift_m)
        if len(stretch) >= 2 and np.ptp(stretch[:, :2], axis=0).any():
            pieces.append(tube(stretch, radius_m * ROUGH_RADIUS_SCALE))
    return base, (pv.merge(pieces) if pieces else None)


def _add_path(pl: pv.Plotter, key: str, name: str, meshes, opacity: float = 1.0) -> None:
    """Add (or replace) a path's base and rough-stretch actors under key / key_rough."""
    base, rough = meshes
    pl.add_mesh(base, name=key, color=PLANNER_COLOURS[name], opacity=opacity)
    if rough is not None:
        pl.add_mesh(
            rough,
            name=f"{key}_rough",
            color=PLANNER_BRIGHT[name],
            ambient=ROUGH_AMBIENT,
            opacity=opacity,
        )


def _terrain(run: ShowcaseRun) -> pv.StructuredGrid:
    world = run.scenario.world
    return terrain_mesh(world, step_m=world.grid.resolution)


def _add_static(pl: pv.Plotter, run: ShowcaseRun) -> None:
    """Boxes and start/goal spheres."""
    sc = run.scenario
    for j, b in enumerate(sc.world.boxes):
        lo, hi = b.min_xyz, b.max_xyz
        box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        pl.add_mesh(box, name=f"box_{j}", color=BOX_COLOUR, smooth_shading=False)
    for name, (x, y), colour in (
        ("start", sc.start_xy, START_COLOUR),
        ("goal", sc.goal_xy, GOAL_COLOUR),
    ):
        z = float(surface_z(sc, np.array([x]), np.array([y]))[0]) + MARKER_RADIUS_M
        pl.add_mesh(pv.Sphere(radius=MARKER_RADIUS_M, center=(x, y, z)), name=name, color=colour)


def _finish_view(pl: pv.Plotter, grid: GridSpec) -> None:
    pl.camera_position = camera_view(grid)
    pl.reset_camera_clipping_range()


def metric_label(name: str, run: ShowcaseRun) -> str:
    m = run.metrics[name]
    if m is None:
        return f"{PLANNER_LABELS[name]}: no path"
    return (
        f"{PLANNER_LABELS[name]}: {m.length_m:.2f} m, rough exposure {m.rough_exposure_m2:.3f} m²"
    )


def render_hero(
    run: ShowcaseRun,
    map_grid: GridSpec,
    caption: str,
    out_path: str | Path,
    alpha: float,
    note: str | None = None,
) -> Path:
    """Still: terrain coloured by GT traversability cost, the three paths, legend and caption.
    note: an extra legend line (an observation about this scene), if given.

    The 3D scene is rendered by PyVista; legend, colour bar and captions are composited with
    matplotlib (exact layout and a true dashed legend line for Oracle). Output HERO_SIZE pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    sc, grid = run.scenario, run.scenario.world.grid
    pl = pv.Plotter(off_screen=True, window_size=list(HERO_SIZE))
    pl.set_background(BACKGROUND)
    mesh = _terrain(run)
    mesh["gt_cost"] = cost_scalars(run.ground_truth, map_grid, grid)
    pl.add_mesh(
        mesh,
        name="terrain",
        scalars="gt_cost",
        cmap=COST_CMAP,
        clim=cost_limits(run),
        nan_color=LETHAL_COLOUR,
        smooth_shading=True,
        show_scalar_bar=False,
    )
    _add_static(pl, run)
    for name in ("baseline_occ", "terranav"):
        if run.paths_xy[name] is not None:
            meshes = path_meshes(sc, run.paths_xy[name], PATH_LIFT_M, TUBE_RADIUS_M)
            _add_path(pl, name, name, meshes)
    if run.paths_xy["oracle"] is not None:
        meshes = path_meshes(
            sc, run.paths_xy["oracle"], ORACLE_LIFT_M, ORACLE_RADIUS_M, dashed=True
        )
        _add_path(pl, "oracle", "oracle", meshes)
    _finish_view(pl, grid)
    img = pl.screenshot(return_img=True)
    pl.close()

    w, h = HERO_SIZE
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    handles = [
        Line2D([], [], color=PLANNER_COLOURS["terranav"], lw=6),
        Line2D([], [], color=PLANNER_COLOURS["baseline_occ"], lw=6),
        Line2D([], [], color=PLANNER_COLOURS["oracle"], lw=3, ls=(0, (3, 2))),
    ]
    labels = [metric_label(n, run) for n in ("terranav", "baseline_occ", "oracle")]
    labels[2] += " (dashed)"
    handles += [
        Line2D([], [], color=PLANNER_BRIGHT["baseline_occ"], lw=11),
        Line2D([], [], color=START_COLOUR, marker="o", ms=15, ls="none"),
        Line2D([], [], color=GOAL_COLOUR, marker="o", ms=15, ls="none", mec="#a08000"),
    ]
    labels += ["brighter, thicker: stretch inside a rough patch", "start", "goal"]
    if run.ground_truth.lethal.any():
        handles.append(Patch(color=LETHAL_COLOUR))
        labels.append("GT lethal")
    if note:
        handles.append(Line2D([], [], alpha=0.0))
        labels.append(note)
    leg = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.975),
        fontsize=16,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=3,
        title=f"alpha = {alpha:g}; evaluator metrics (ground truth)",
        title_fontsize=15,
    )
    leg.get_frame().set_linewidth(1.0)
    cax = fig.add_axes((0.70, 0.11, 0.25, 0.022))
    lo, hi = cost_limits(run)
    sm = plt.cm.ScalarMappable(cmap=COST_CMAP, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[lo, hi / 2, hi])
    cb.ax.set_xticklabels([f"{lo:g}", f"{hi / 2:.2f}", f"{hi:.2f}"])
    cb.set_label(
        f"terrain colour: GT traversability cost, 0 to {hi:.2f} (max in scene)",
        fontsize=14,
        color=INK,
    )
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=12)
    fig.text(0.012, 0.022, caption, fontsize=15, color=INK, ha="left", va="bottom")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    return out_path


# ---- replay ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayTiming:
    """Frames per phase at `fps` frames per second (defaults: 19.6 s at 10 fps)."""

    fps: int = 10
    per_pose: int = 8  # scanning: frames per scan pose
    mapping: int = 30
    planning: int = 30
    executing: int = 60
    hold: int = 20  # final frame repeated (2 s)


REPLAY_PLANNERS = ("terranav", "baseline_occ")


def _caption(pl: pv.Plotter, text: str) -> None:
    pl.add_text(text, position="upper_left", font_size=12, color=INK, name="caption")


def _counter(pl: pv.Plotter, lines: list[str]) -> None:
    pl.add_text("\n".join(lines), position="lower_right", font_size=12, color=INK, name="counter")


def replay_frames(
    run: ShowcaseRun,
    map_grid: GridSpec,
    title: str,
    timing: ReplayTiming | None = None,
    window_size: tuple[int, int] = REPLAY_SIZE,
) -> Iterator[np.ndarray]:
    """Yield (H, W, 3) uint8 frames of the replay: scanning, mapping, planning, executing, hold.

    Rough-exposure counters use the evaluator's definition on the travelled prefix
    (rough_exposure_along), so the final totals equal the evaluator's.
    """
    timing = timing if timing is not None else ReplayTiming()
    sc, grid = run.scenario, run.scenario.world.grid
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background(BACKGROUND)
    mesh = _terrain(run)
    pl.add_mesh(mesh, name="terrain", color=NEUTRAL_TERRAIN, smooth_shading=True)
    _add_static(pl, run)
    _finish_view(pl, grid)
    pl.add_text(
        "green sphere = start, gold = goal; brighter, thicker = path inside a rough patch",
        position="lower_left",
        font_size=10,
        color=INK,
        name="key",
    )

    def shot() -> np.ndarray:
        pl.render()
        return pl.screenshot(return_img=True)

    intr = SimConfig().sensor.intrinsics()
    head = f"REPLAY of the scan-once, plan-once pipeline (not a live simulation). {title}"

    # a) scanning: pose i appears with its frustum and that frame's points
    k = len(run.poses)
    for i, (T, pts) in enumerate(zip(run.poses, run.points, strict=True)):
        pl.add_mesh(_frustum_lines(T, intr), name=f"frustum_{i}", color=POSE_COLOUR, line_width=3)
        if len(pts):
            pl.add_mesh(
                pv.PolyData(pts),
                name=f"points_{i}",
                color=POINT_COLOUR,
                point_size=2,
                render_points_as_spheres=False,
            )
        _caption(pl, f"{head}\n1/4 Scanning: pose {i + 1} of {k}, frustum + that frame's points")
        frame = shot()
        for _ in range(timing.per_pose):
            yield frame

    # b) mapping: perceived traversability fades in over known cells; points fade out
    known = map_to_terrain_points(
        (run.perceived.core_reason & R_UNKNOWN) == 0, map_grid, grid
    ).astype(float)
    overlay = mesh.copy()
    overlay.points[:, 2] += 0.01
    overlay["known"] = known
    overlay["cost"] = cost_scalars(run.perceived, map_grid, grid)
    overlay = overlay.threshold(0.5, scalars="known", all_scalars=True)
    actor = pl.add_mesh(
        overlay,
        name="perceived",
        scalars="cost",
        cmap=COST_CMAP,
        clim=cost_limits(run),  # the hero's scale: max GT cost in the scene
        nan_color=LETHAL_COLOUR,
        smooth_shading=True,
        opacity=0.0,
        show_scalar_bar=False,
    )
    point_actors = [pl.actors[f"points_{i}"] for i in range(k) if f"points_{i}" in pl.actors]
    _caption(
        pl,
        f"{head}\n2/4 Mapping: perceived traversability (dark red = lethal, uncoloured = unknown)",
    )
    for j in range(timing.mapping):
        t = (j + 1) / timing.mapping
        actor.prop.opacity = t
        for a in point_actors:
            a.prop.opacity = 1.0 - t
        yield shot()
    for i in range(k):
        pl.remove_actor(f"points_{i}")

    # c) planning: both paths draw progressively
    paths = {n: run.paths_xy[n] for n in REPLAY_PLANNERS if run.paths_xy[n] is not None}
    lengths = {n: float(run.metrics[n].length_m) for n in paths}
    _caption(
        pl, f"{head}\n3/4 Planning: TerraNav (green) and Baseline-Occ (red) on the perceived map"
    )
    for j in range(timing.planning):
        t = (j + 1) / timing.planning
        for n, p in paths.items():
            meshes = path_meshes(sc, p, PATH_LIFT_M, TUBE_RADIUS_M * 0.5, t * lengths[n])
            _add_path(pl, f"plan_{n}", n, meshes, opacity=0.6)
        yield shot()

    # d) executing: same speed for both robots; trails; accumulated rough exposure
    speed = max(lengths.values()) / timing.executing  # metres per frame
    _caption(pl, f"{head}\n4/4 Executing: both robots at the same speed (0.3 m footprint discs)")

    def counters(s_of) -> list[str]:
        return ["accumulated rough exposure"] + [
            f"{PLANNER_LABELS[n]}: {rough_exposure_along(sc, paths[n], s_of(n)):.3f} m^2"
            for n in paths
        ]

    for j in range(timing.executing):
        s = (j + 1) * speed
        for n, p in paths.items():
            s_n = min(s, lengths[n])
            _add_path(pl, f"trail_{n}", n, path_meshes(sc, p, PATH_LIFT_M, TUBE_RADIUS_M, s_n))
            x, y, z = path_points(sc, path_prefix(p, s_n), PATH_LIFT_M)[-1]
            disc = pv.Cylinder(
                center=(x, y, z), direction=(0, 0, 1), radius=ROBOT_RADIUS_M, height=0.06
            )
            pl.add_mesh(disc, name=f"robot_{n}", color=PLANNER_COLOURS[n])
        _counter(pl, counters(lambda n, s=s: min(s, lengths[n])))
        yield shot()

    # e) hold the final frame with both totals (evaluator values)
    _caption(pl, f"{head}\nDone: totals from the independent evaluator")
    _counter(
        pl,
        ["totals (evaluator)"]
        + [
            f"{PLANNER_LABELS[n]}: {lengths[n]:.2f} m, rough exposure "
            f"{run.metrics[n].rough_exposure_m2:.3f} m^2"
            for n in paths
        ],
    )
    frame = shot()
    pl.close()
    for _ in range(timing.hold):
        yield frame
