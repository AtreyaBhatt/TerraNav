# TerraNav — project rules for Claude Code

## What this is
Software-only terrain-aware perception + navigation pipeline (resume project).
Simulated depth camera over a heightfield world → backprojection → grid map →
traversability → terrain-aware A*. Author must be able to explain every line.

## Stack (do not add dependencies without asking)
Python, NumPy, SciPy (ndimage), matplotlib, PyVista (offscreen), pytest,
hypothesis, ruff. Managed with uv. NO ROS, Gazebo, GPU libs, Rust, Open3D, torch.

## Working rules
- Work on ONE roadmap step per session. Stop and report when the step is done.
- Write or update tests alongside code; tests must pass and ruff must be clean.
- Never use global RNG (np.random.seed / np.random.rand). Always pass a
  np.random.Generator from np.random.default_rng(seed).
- Poses, rotations, rays: float64. Depth images may be float32.
- Invalid depth = NaN, never 0.
- Never invent benchmark numbers or claim results that were not measured.
- Prefer small pure functions with docstrings stating units and frames.
- Every angle parameter name carries a unit suffix (_rad/_deg), e.g. yaw_rad, slope_deg.
- Do not touch files outside the current step except for imports/stubs.

## Frame conventions (authoritative; see docs/frames.md)
- World: x forward, y left, z up. Origin at map corner; x, y in [0, 12] m.
- Camera optical: x right, y down, z forward. Pixel u right, v down.
- depth_z = z in camera frame (NOT Euclidean range). Name range variables
  `range_t` and depth variables `depth_z`; never mix them.
- R_base = [[0,0,1],[-1,0,0],[0,-1,0]]  (optical → world at yaw=0, pitch=0)
- R_wc = Rz(yaw) @ Ry(pitch_down) @ R_base; positive pitch_down tilts the
  optical axis toward -z. T_world_cam maps camera-frame points to world.
- Ry(t) = [[c,0,s],[0,1,0],[-s,0,c]],  Rz(t) = [[c,-s,0],[s,c,0],[0,0,1]]
- Intrinsics: fx = fy = (W/2) / tan(hfov/2). Pixel centers at integer
  indices: cx = (W-1)/2, cy = (H-1)/2. Ray for pixel (u,v):
  d_cam = [(u-cx)/fx, (v-cy)/fy, 1].
- Heightfield array indexed [row=iy, col=ix]; cell size 0.05 m.
  scipy map_coordinates takes (row, col). Convert metres↔cells explicitly.

## Defaults
World 12x12 m, heightfield 0.05 m. Camera 320x240, hfov 87°, mounted 0.5 m
above ground, pitch_down 25°, max range 8 m, noise sigma(z) = s0 + s1*z^2
(s0=0.001, s1=0.0015, illustrative), dropout 2%, grazing cutoff 85°.
Map grid 0.1 m (120x120), unknown = blocked.
