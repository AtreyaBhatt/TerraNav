from __future__ import annotations

from dataclasses import dataclass, field

from terranav.types import CameraIntrinsics, GridSpec, SensorConfig


@dataclass(frozen=True)
class SimConfig:
    terrain_grid: GridSpec = field(
        default_factory=lambda: GridSpec(
            resolution=0.05, size_x=12.0, size_y=12.0, origin=(0.0, 0.0)
        )
    )
    map_grid: GridSpec = field(
        default_factory=lambda: GridSpec(
            resolution=0.1, size_x=12.0, size_y=12.0, origin=(0.0, 0.0)
        )
    )
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    num_scan_poses: int = 6
    ray_step_m: float = 0.025
