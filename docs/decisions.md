# Design decisions

Short records: decision, alternatives considered, reason, evidence. Numbers are measured
(see the cited files/commits); none are targets.

## D1. Grazing cutoff 85° (was 80°)
- **Decision:** invalidate depth returns with incidence > 85° (`SensorConfig.grazing_cutoff_rad`).
- **Alternatives:** 80° (initial default), no cutoff.
- **Reason:** on flat ground, visibility ends at `h · tan(cutoff)`: 2.8 m at 80° vs 5.7 m at 85°
  for the 0.5 m mount (docs/frames.md). 80° left the robot nearly blind beyond 3 m.
- **Evidence:** mixed seed 0 start frame, GRAZING pixels 5415 (80°) → 1237 (85°)
  (`scripts/preview_sensor.py`, Steps 4–5).

## D2. Flank scan strategy (two offset rows, toe-in)
- **Decision:** `scan_poses(strategy="flank")`: rows at ±2.5 m from the start→goal line, fractions
  0.1/0.45/0.8, yawed 30° toward the line; poses only on eroded easy ground. Since D9 the
  default survey is "flank+start" (these 6 poses plus one at the start).
- **Alternatives:** poses on the start→goal line ("line", kept as baseline).
- **Reason:** line poses sit inside obstacles or look along them; flank poses see the corridor
  from the side and never need a pose inside a box.
- **Evidence:** `out/eval_scan.txt` at commit ab5b5a2 (seeds 0–9): corridor coverage mean flank vs
  line 0.680/0.379 (flat_obstacles) … 0.743/0.580 (mixed); flank higher for every family and
  region; line skipped 43 of 300 poses.

## D3. No ground segmentation: a 2.5D elevation map
- **Decision:** fuse all points into per-cell height statistics (`perception/elevation_map.py`);
  boxes appear as large height steps, not as a separate obstacle class.
- **Alternatives:** RANSAC/normal-based ground segmentation; 3D voxel map.
- **Reason:** the world has no overhangs (heightfield + solid boxes), so one height distribution
  per cell carries what the cost function needs; segmentation would add thresholds and failure
  modes (ramps, mesas) without new information.
- **Evidence:** `out/eval_map.txt` at ab5b5a2-dirty (seeds 0–9, "flank" survey): observed-cell
  |z_mean − GT| median ≤ 0.4 mm, p95 ≤ 4.4 mm in every family.

## D4. Unknown = blocked, with small-gap filling
- **Decision:** planning treats UNKNOWN cells as blocked; `fill_gaps` bridges only gaps ≤ 0.3 m
  (illustrative) by normalized convolution, marking them FILLED.
- **Alternatives:** unknown = free (optimistic); fill everything (inpainting).
- **Reason:** optimistic unknowns hide unseen obstacles; but without filling, scan-line and
  dropout gaps would fragment otherwise traversable ground.
- **Evidence:** `out/eval_map.txt` at ab5b5a2-dirty ("flank" survey, no fill guard): filling
  raised corridor coverage by 0.019–0.029. The fill error it exposed is handled by D8.

## D5. Robot footprint: the only non-sensor input to perception
- **Decision:** `mark_footprint(emap, center_xy, radius_m, heights)` sets cells within 0.4 m of the
  start to FOOTPRINT after filling, with heights passed in by the caller (proprioception; the
  evaluation code supplies them from ground truth). Perception itself imports no ground truth (D7).
- **Alternatives:** no footprint (start cell unknown → planner cannot leave it); GT everywhere.
- **Reason:** the robot stands there, so pose and wheel contact (proprioception) give that terrain;
  everything else must come from the sensor.
- **Evidence:** renders `out/elevation_mixed_s{0,1}.png` ("flank" survey): the start area was never
  in view; without the footprint the start would be unknown (D9 adds the missing view).

## D6. Sensing and scope
- **Decision:** one simulated depth camera (no RGB, no lidar); known camera poses; static world;
  scan once, then plan once. No replanning, no control, no localisation.
- **Alternatives:** active/online mapping with replanning; SLAM with estimated poses; multi-sensor.
- **Reason:** isolates the perception → map → traversability → planning chain so every stage can
  be checked against ground truth; each excluded item adds an error source without testing it.
- **Evidence:** scope choice, not a measurement.

## D7. Sensor boundary and extension point
- **Decision:** perception consumes only DepthFrames, known poses and the sensor noise model
  (`terranav.sensor_model`); it never imports `terranav.sim` (tests/test_architecture.py).
  Other knowledge (footprint heights) is passed in explicitly. See CLAUDE.md.
- **Alternatives:** let perception read simulator state (ideal depth, GT heights) for convenience.
- **Reason:** keeps results honest (no GT leaks into the map) and makes DepthFrame the extension
  point: RGB depth estimation, stereo or lidar can feed the same pipeline. RGB-only terrain
  perception is planned future work.
- **Evidence:** architecture test passes; perception imports only frames, types, sensor_model.

## D8. Fill guard: no interpolation across steps
- **Decision:** `fill_gaps` skips a cell when the observed heights in its window span more than
  `STEP_GUARD_M` = 0.10 m (illustrative; M2-B's step threshold is meant to reuse it).
- **Alternatives:** unguarded fill; fill only from z_max (conservative but still invents surfaces).
- **Reason:** box tops above the camera are never seen, so a box appears as an observed rim
  around unknown cells; unguarded filling interpolated low rim/ground heights into the box.
- **Evidence:** before the guard, filled-cell |z_mean − GT| p95 was 0.47 m (flat_obstacles) and
  0.26 m (mixed) (`out/eval_map.txt` at ab5b5a2-dirty). After: TBD from the clean-tree run.
  tests/test_hole_fill.py: a box rim stays unknown with the guard, its interior corners fill
  without it.

## D9. Start pose added to the survey ("flank+start", K = 7)
- **Decision:** default `scan_poses` strategy "flank+start": one pose at the start (0.5 m up, yaw
  toward the goal, pitch 25°) followed by the 6 flank poses; frame 0 is the start pose.
- **Alternatives:** larger footprint radius; flank poses closer to the start.
- **Reason:** no flank pose sees the ground next to the robot, so the footprint was an island
  of known cells and, with unknown = blocked, the robot could not leave it.
- **Evidence:** start→goal known-connectivity on mixed seeds 0 and 1: False with "flank",
  True with "flank+start". The start camera's nearest visible ground is ~0.285 m ahead on flat
  ground (tests/test_world.py), inside the 0.4 m footprint. Family-level rate: TBD (clean run).
