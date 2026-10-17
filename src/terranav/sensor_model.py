"""Depth-sensor noise model: the "datasheet" shared by the simulator and perception.

The simulator draws noise from it; perception only uses it to weight measurements. It is
the one piece of sensor knowledge allowed across the sensor boundary (CLAUDE.md).
"""

from __future__ import annotations

import numpy as np

from terranav.types import SensorConfig


def noise_sigma(depth_z, cfg: SensorConfig) -> np.ndarray:
    """Axial noise std (metres) at camera depth depth_z (metres): s0 + s1 * depth_z^2."""
    return cfg.noise_s0_m + cfg.noise_s1_per_m * np.square(depth_z)
