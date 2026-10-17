"""Shared pytest configuration."""

from __future__ import annotations

import os

from hypothesis import settings

# Derandomized so every run explores the same examples (reproducible CI and local runs).
# Override with HYPOTHESIS_PROFILE=<name> if another profile is registered.
settings.register_profile("deterministic", derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "deterministic"))
