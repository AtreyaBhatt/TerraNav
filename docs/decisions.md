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
  |z_mean − GT| median ≤ 0.4 mm, p95 ≤ 4.4 mm in every family. At 3b8996a ("flank+start"):
  median 0.0003 m in every family, p95 0.0032–0.0050 m. Perception latency (fuse + fill +
  footprint, serial, `out/eval_map_latency.txt` at 3b8996a): median 24.4–26.4 ms, p95
  25.0–36.2 ms per scenario.

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
  in view; without the footprint the start would be unknown (D9 adds the missing view). With
  footprint + start pose, start and goal share a known component in 1.00 of scenarios in every
  family (`out/eval_map.txt`, 3b8996a, seeds 0–9).

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
  0.26 m (mixed) (`out/eval_map.txt` at ab5b5a2-dirty). After (span guard, 3b8996a): 0.0028
  (flat_obstacles), 0.0124 (slopes), 0.0179 (rough), 0.0526 (steps), 0.0101 m (mixed).
  Superseded by the plane-residual guard (D12).
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
  ground (tests/test_world.py), inside the 0.4 m footprint. Family-level rate with
  "flank+start": connected 1.00 in all five families (`out/eval_map.txt`, 3b8996a, seeds 0–9).

## D10. Traversability model
- **Decision:** `perception/traversability.py`. Lethal = unknown | no feature estimate |
  slope ≥ 30° | roughness ≥ 0.05 m | step ≥ 0.20 m, then dilated by the robot radius 0.3 m.
  Cost = 0.4·slope/30° + 0.3·rough/0.05 + 0.3·step/0.20 in [0, 1), NaN where lethal.
  ALL values illustrative (`TraversabilityConfig`), not from a vehicle model.
- **Alternatives:** learned or vehicle-dynamics costs; a single height-difference threshold.
- **Reason:** three interpretable features with hard limits and a bounded, monotone cost that
  the planner can trade against distance; ground truth uses the same functions on GT heights.
- **Evidence:** TBD from the clean `scripts/eval_trav.py` run.

## D11. Noise-aware features
- **Decision:** roughness = sqrt(max(0, residual² − fit_noise_var)), where residual is the
  dof-corrected weighted plane-fit RMS and fit_noise_var = support / Σ(1/z_mean_var) (noise of the
  fitted cell means). A neighbour jump counts as a step only if |Δz| > 3·sqrt(var_i + var_j).
- **Alternatives:** raw residual and raw jumps (noise reads as roughness/steps at long range);
  subtracting per-point noise_var (overstates the noise of a many-point cell mean ~n times).
- **Reason:** sensor noise grows as s0 + s1·z²; without these corrections far, noisy cells would
  look rough or stepped. On a clean slope adjacent jumps stay ≤ tan(slope)·0.1·√2 (0.066 m at 25°).
- **Evidence:** tests/test_features.py (iid noise: roughness ≈ σ without, ≈ 0 with the
  subtraction; bounded noise passes no steps). Pipeline numbers: TBD from the clean eval run.

## D12. Fill guard v2: plane residual
- **Decision:** `fill_gaps` fits a plane to the observed cells around each gap cell (radius 0.3 m)
  and fills only if support ≥ 8 and residual ≤ 0.02 m (illustrative). Replaces the 0.10 m span guard.
- **Alternatives:** span guard (D8), which also refused legitimate fills on slopes (a 25° ramp
  spans ~0.19 m over the window).
- **Reason:** the residual separates "tilted but planar" (fill) from steps, box rims and rough
  ground (do not fill).
- **Evidence:** tests/test_hole_fill.py: a striped 25° ramp fills, a striped 0.3 m step and a
  box rim do not. Pipeline numbers: TBD from the clean eval run.
