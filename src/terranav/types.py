"""Core immutable data types. Frames and units follow docs/frames.md.

These classes hold data plus small validation/conversion helpers only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

# --- Defaults from CLAUDE.md --------------------------------------------------
# This block is the ONLY place where degree-valued defaults are converted to radians.
DEFAULT_HFOV_RAD = math.radians(87.0)
DEFAULT_PITCH_DOWN_RAD = math.radians(25.0)
DEFAULT_GRAZING_CUTOFF_RAD = math.radians(85.0)

DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240


def _focal_from_hfov(width: int, hfov_rad: float) -> float:
    """Focal length in pixels: f = (W/2) / tan(hfov/2)."""
    return (width / 2.0) / math.tan(hfov_rad / 2.0)


_DEFAULT_F = _focal_from_hfov(DEFAULT_IMAGE_WIDTH, DEFAULT_HFOV_RAD)


def _readonly_copy(arr, dtype) -> np.ndarray:
    """Copy `arr` as `dtype` and mark it read-only, so frozen dataclasses own immutable data."""
    out = np.array(arr, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Regular 2D grid over the world x-y plane.

    Attributes:
        resolution: cell edge length, metres.
        size_x, size_y: grid extent along world x and y, metres.
        origin: world (x, y) of the grid's lower corner (cell [0, 0] edge), metres.

    Arrays on this grid are indexed [row=iy, col=ix], shape (ny, nx).
    Cell (iy, ix) has its centre at origin + ((ix + 0.5) * res, (iy + 0.5) * res).
    """

    resolution: float
    size_x: float
    size_y: float
    origin: tuple[float, float]

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")
        for name, size in (("size_x", self.size_x), ("size_y", self.size_y)):
            if size <= 0:
                raise ValueError(f"{name} must be > 0, got {size}")
            n = size / self.resolution
            if abs(n - round(n)) > 1e-9:
                raise ValueError(f"{name}={size} is not a multiple of resolution")

    @property
    def nx(self) -> int:
        """Number of columns (cells along world x)."""
        return round(self.size_x / self.resolution)

    @property
    def ny(self) -> int:
        """Number of rows (cells along world y)."""
        return round(self.size_y / self.resolution)

    def xy_to_rc(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """World (x, y) metres -> integer cell indices (row, col) = (iy, ix).

        Accepts scalars or arrays. No clipping: out-of-grid points give out-of-range
        indices; callers decide how to handle them.
        """
        ix = np.floor((np.asarray(x, dtype=np.float64) - self.origin[0]) / self.resolution)
        iy = np.floor((np.asarray(y, dtype=np.float64) - self.origin[1]) / self.resolution)
        return iy.astype(np.int64), ix.astype(np.int64)

    def rc_to_xy(self, row, col) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices (row, col) = (iy, ix) -> world (x, y) of the cell centre, metres."""
        x = self.origin[0] + (np.asarray(col, dtype=np.float64) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(row, dtype=np.float64) + 0.5) * self.resolution
        return x, y


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels. Pixel centres at integer (u, v); u right, v down."""

    width: int = DEFAULT_IMAGE_WIDTH
    height: int = DEFAULT_IMAGE_HEIGHT
    fx: float = _DEFAULT_F
    fy: float = _DEFAULT_F
    cx: float = (DEFAULT_IMAGE_WIDTH - 1) / 2.0
    cy: float = (DEFAULT_IMAGE_HEIGHT - 1) / 2.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"image size must be positive, got {self.width}x{self.height}")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"focal lengths must be > 0, got fx={self.fx}, fy={self.fy}")

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_rad: float) -> CameraIntrinsics:
        """Square-pixel intrinsics from horizontal FOV (radians).

        fx = fy = (W/2) / tan(hfov/2); cx = (W-1)/2; cy = (H-1)/2.
        """
        f = _focal_from_hfov(width, hfov_rad)
        return cls(width, height, f, f, (width - 1) / 2.0, (height - 1) / 2.0)

    @property
    def K(self) -> np.ndarray:
        """(3,3) float64 camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Simulated depth sensor parameters. Angles in radians, distances in metres.

    Noise model: sigma(depth_z) = noise_s0_m + noise_s1_per_m * depth_z**2 (illustrative).
    Validity cut-off: max_depth_z_m applies to depth_z (camera-frame z), NOT range_t.
    """

    width: int = DEFAULT_IMAGE_WIDTH
    height: int = DEFAULT_IMAGE_HEIGHT
    hfov_rad: float = DEFAULT_HFOV_RAD
    mount_height_m: float = 0.5
    pitch_down_rad: float = DEFAULT_PITCH_DOWN_RAD
    max_depth_z_m: float = 8.0
    noise_s0_m: float = 0.001
    noise_s1_per_m: float = 0.0015
    dropout_prob: float = 0.02
    grazing_cutoff_rad: float = DEFAULT_GRAZING_CUTOFF_RAD

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        if self.max_depth_z_m <= 0:
            raise ValueError(f"max_depth_z_m must be > 0, got {self.max_depth_z_m}")
        if not 0.0 < self.hfov_rad < math.pi:
            raise ValueError(f"hfov_rad must be in (0, pi), got {self.hfov_rad}")

    def intrinsics(self) -> CameraIntrinsics:
        """Intrinsics implied by width, height and hfov_rad."""
        return CameraIntrinsics.from_hfov(self.width, self.height, self.hfov_rad)


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in world frame, metres. Requires min_xyz < max_xyz on every axis."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.min_xyz) != 3 or len(self.max_xyz) != 3:
            raise ValueError("min_xyz and max_xyz must each have 3 elements")
        if not all(lo < hi for lo, hi in zip(self.min_xyz, self.max_xyz, strict=True)):
            raise ValueError(f"Box requires min < max per axis, got {self.min_xyz}, {self.max_xyz}")


@dataclass(frozen=True, slots=True, eq=False)
class World:
    """Heightfield world.

    heightfield: (ny, nx) float64 terrain height z in metres, indexed [iy, ix] on `grid`.
    boxes: axis-aligned obstacles in world frame.
    family: name of the generator family; seed: RNG seed used to build it.
    """

    heightfield: np.ndarray
    grid: GridSpec
    boxes: tuple[Box, ...]
    family: str
    seed: int

    def __post_init__(self) -> None:
        heightfield = _readonly_copy(self.heightfield, np.float64)
        expected = (self.grid.ny, self.grid.nx)
        if heightfield.shape != expected:
            raise ValueError(f"heightfield shape {heightfield.shape} != (ny, nx) {expected}")
        object.__setattr__(self, "heightfield", heightfield)


@dataclass(frozen=True, slots=True, eq=False)
class Scenario:
    """A generated world plus navigation task.

    start_xy, goal_xy: world (x, y) in metres. params: read-only mapping of the sampled
    generator parameters, for logging/reproduction. family and seed must match `world`.
    """

    world: World
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    family: str
    seed: int
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (self.family, self.seed) != (self.world.family, self.world.seed):
            raise ValueError("Scenario family/seed must match its World")
        object.__setattr__(self, "start_xy", (float(self.start_xy[0]), float(self.start_xy[1])))
        object.__setattr__(self, "goal_xy", (float(self.goal_xy[0]), float(self.goal_xy[1])))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True, eq=False)
class DepthFrame:
    """One depth image with its camera pose.

    depth_z: (H, W) float32 camera-frame z in metres (NOT Euclidean range); NaN = invalid.
    T_world_cam: (4,4) float64 transform mapping camera-optical points to world.
    """

    depth_z: np.ndarray
    intrinsics: CameraIntrinsics
    T_world_cam: np.ndarray

    def __post_init__(self) -> None:
        depth_z = _readonly_copy(self.depth_z, np.float32)
        expected = (self.intrinsics.height, self.intrinsics.width)
        if depth_z.shape != expected:
            raise ValueError(f"depth_z shape {depth_z.shape} != (H, W) {expected}")
        T = _readonly_copy(self.T_world_cam, np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_world_cam must have shape (4,4), got {T.shape}")
        object.__setattr__(self, "depth_z", depth_z)
        object.__setattr__(self, "T_world_cam", T)
