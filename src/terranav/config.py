from __future__ import annotations

from dataclasses import dataclass, field

from terranav.types import CameraIntrinsics, GridSpec, SensorConfig


@dataclass(frozen=True)
class SimConfig:
    grid: GridSpec = field(default_factory=GridSpec)
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    num_scan_poses: int = 6
    camera_height_m: float = 0.5
    camera_pitch_down_deg: float = 25.0
    ray_step_m: float = 0.025
