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
  Kept, combined with the plane-residual guard (D12).
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
- **Change (M2-B round 2): r_step band removed.** The first version spread each cell's step to
  all cells within r_step = 0.2 m before the 0.3 m robot dilation: double inflation. It also made
  GT and perception disagree (GT edges got the band, unseen perceived edges are unknown and got
  only the dilation). Now a step belongs only to the endpoints of the jumping pair; the robot's
  extent comes only from the dilation.
- **Evidence:** eval_trav at 3b8996a-dirty (with band): flat_obstacles perceived start-goal
  connectivity 0.90 vs GT 1.00. The one closure, flat_obstacles seed 3 at 3ceb792, reopens if
  either the unknown cells or the banded step cells are removed (`scripts/diag_trav.py`); without
  the band it connects. Clean run at 6c4fd3f (`out/eval_trav.txt`, seeds 0–9) (flat_obstacles / slopes / rough / steps / mixed):
  perceived start-goal connectivity 1.00 / 1.00 / 1.00 / 1.00 / 1.00 (GT 1.00 in all); core FF
  0.002 / n/a (0) / 0.000 / 0.173 / 0.006; band FF 0.030 / n/a (0) / 0.028 / 0.425 / 0.076;
  FL 0.317 / 0.294 / 0.264 / 0.285 / 0.292, of which unknown 0.221 / 0.196 / 0.171 / 0.185 / 0.195
  and dilation only 0.096 / 0.098 / 0.092 / 0.100 / 0.097. Features + traversability latency
  (`out/eval_trav_latency.txt`, serial): median 6.0 / 5.8 / 5.8 / 5.8 / 5.8 ms.

## D11. Noise-aware features
- **Decision:** roughness = sqrt(max(0, residual² − fit_noise_var)), where residual is the
  dof-corrected weighted plane-fit RMS and fit_noise_var = support / Σ(1/z_mean_var) (noise of the
  fitted cell means). A jump counts as a step only if |Δz| > 3·sqrt(var_i + var_j) (D13).
- **Alternatives:** raw residual and raw jumps (noise reads as roughness/steps at long range);
  subtracting per-point noise_var (overstates the noise of a many-point cell mean ~n times).
- **Reason:** sensor noise grows as s0 + s1·z²; without these corrections far, noisy cells would
  look rough or stepped. On a clean slope pair jumps stay ≤ tan(slope)·0.2 m (0.115 m at 30°).
  The r_step band once attached to the step feature is gone (D10).
- **Evidence:** tests/test_features.py (iid noise: roughness ≈ σ without, ≈ 0 with the
  subtraction; bounded noise passes no steps). Clean run at 6c4fd3f (`out/eval_trav.txt`)
  (flat_obstacles / slopes / rough / steps / mixed): roughness MAE 0.0150 / 0.0008 / 0.0010 / 0.0020 / 0.0054 m; step MAE 0.0334 / 0.0015 /
  0.0024 / 0.0040 / 0.0119 m; slope MAE 2.35 / 0.24 / 0.33 / 0.59 / 1.07 deg.

## D12. Fill guard: span AND plane residual
- **Decision:** `fill_gaps` fills a gap cell only if (span) the observed heights in its window
  span ≤ `STEP_GUARD_M` = 0.10 m AND (residual) a plane fitted to the observed cells around it
  (radius 0.3 m) has support ≥ 8 and residual ≤ `FILL_RESIDUAL_GUARD_M` = 0.02 m. Illustrative.
- **Alternatives:** v1 span only (D8); v2 residual only.
- **Reason:** v2 (residual only) fills a low box whose top was never seen when the observed
  cells around it are planar ground: the hazard is erased. v1 (span only) cannot tell a planar
  tilt from a step at small spans. The AND refuses both; the cost is that steep planes (a 25°
  ramp spans ~0.19 m over the window) are not filled.
- **Evidence (exploratory, not final):** filled-cell |z_mean − GT| p95 per family
  (flat_obstacles, slopes, rough, steps, mixed): v1 at 3b8996a 0.0028 / 0.0124 / 0.0179 / 0.0526
  / 0.0101 m; v2 at 3ceb792-dirty 0.0476 / 0.0158 / 0.0279 / 0.0842 / 0.0244 m. Diagnosis
  (flat_obstacles seed 0): 27 of 28 box-top cells next to core false-free cells were FILLED,
  0.237 m low (median), under v2. Core false-free cells, flat_obstacles + steps, seeds 0–9: see
  D13 table. tests/test_hole_fill.py: 0.2-slope stripes fill, 25° stripes, a 0.3 m step, a box
  rim and a low box top do not. Clean run at 6c4fd3f with the AND guard (`out/eval_map.txt`)
  (flat_obstacles / slopes / rough / steps / mixed): filled-cell |z_mean − GT| p95 0.0027 / 0.0119 / 0.0179 / 0.0515 / 0.0095 m; corridor
  coverage raw → fill 0.718 → 0.734, 0.787 → 0.804, 0.806 → 0.829, 0.798 → 0.817, 0.778 → 0.795;
  start-goal known-connectivity 1.00 in all families.

## D13. Step lethality: significance gate kept; pairs up to 0.2 m
- **Decision:** a step is lethal when a jump ≥ 0.20 m is also 3-sigma significant
  (`TraversabilityConfig.step_gate_for_lethal = True`). Jumps are measured between cell pairs up
  to `STEP_PAIR_DIST_M` = 0.2 m apart (illustrative), endpoints only; GT uses the same function.
- **Alternatives:** ungated raw jumps for lethality; adjacent pairs only; z_max/z_min jumps.
- **Reason:** gate: in the diagnosis no GT step cell had a raw perceived jump ≥ 0.2 m that the
  gate rejected, so gated and ungated results were identical; keeping the gate protects against
  noise-driven steps at range. Pair distance: an edge inside a cell splits one jump into two
  adjacent half-jumps (a 0.24 m box edge read as 2 × 0.12 m); pairs two cells apart see the whole
  jump. z_max/z_min would track noise extremes, which grow with range. A 30° slope rises
  0.115 m over 0.2 m, below step_crit.
- **Evidence (exploratory, `scripts/diag_trav.py`, flat_obstacles + steps, seeds 0–9,
  3ceb792-dirty; core false-free cells = GT lethal before dilation, perceived non-lethal):**
  flat_obstacles 86 (30 GT-step) adjacent pairs + v2 guard → 45 (30) with 0.2 m pairs → 28 (15)
  with the AND guard; steps 81 → 81 → 64 (all GT-rough). Gate on/off identical in all three runs.
  Clean run at 6c4fd3f (`out/eval_trav.txt`) (flat_obstacles / slopes / rough / steps / mixed): GT core cells 11546 / 0 / 2 / 370 / 3495,
  core FF 0.002 / n/a (0) / 0.000 / 0.173 / 0.006. The steps family's 0.173 is examined in D14.

## D14. Edge-aware roughness; UCB step lethality not adopted
- **Decision:** roughness is set to 0 for every cell whose plane-fit window (0.3 m) contains an
  endpoint of a significant pair jump ≥ `ROUGH_EDGE_STEP_M` = 0.10 m (illustrative); edges are
  the step feature's job, roughness measures texture. Perception and GT use the same function.
  `TraversabilityConfig.lethal_ucb_k` stays at its default 0 (point-value step lethality).
- **Alternatives:** robust (outlier-rejecting) plane fit, so an edge does not inflate the
  residual: rejected for now, as it needs iterative per-cell fitting instead of the vectorised
  moment sums, and still mixes both levels where the edge passes near the cell centre.
  UCB lethality k = 1 or 2.
- **Reason:** a plane fitted across a sub-critical edge (e.g. a 0.15 m terrace step) reports
  the edge as roughness, creating a band of borderline "rough" cells one fit radius wide. At
  6c4fd3f all 64 core false-free cells of the steps family were such roughness-only cells, with
  GT within 1 cm of rough_crit.
- **Evidence (exploratory, `scripts/diag_trav.py`, 6c4fd3f-dirty, all families, seeds 0–9).**
  Core false-free cells / GT core cells, k = 0, before (`out/diag_trav_before_edge_rough.txt`)
  → after (`out/diag_trav.txt`): flat_obstacles 28 / 11546 → 15 / 8610; slopes 0 / 0 → 0 / 0;
  rough 0 / 2 → 0 / 0; steps 64 / 370 (0.173) → 0 / 1 (0.000); mixed 20 / 3495 → 4 / 2431.
  False-lethal rate before → after: 0.317 → 0.326, 0.294 → 0.294, 0.264 → 0.264, 0.285 → 0.291,
  0.292 → 0.297 (more GT-free cells near edges that perception still marks lethal, mostly via
  dilation: 0.096 → 0.107 in flat_obstacles). Remaining core false-free cells: 15 low-box step
  cells (D15), 2 slope cells < 1° over 30° (flat_obstacles), 4 step cells < 1 cm over 0.20 m
  (mixed). No route closures.
- **UCB decision:** after the edge-aware change, k = 1 and k = 2 each remove 1 core false-free
  cell (flat_obstacles 15 → 14; steps 0 and mixed 4 unchanged), while steps FL rises 0.291 →
  0.295 (k = 1) → 0.297 (k = 2), FL by feature 0.000 → 0.001 → 0.002. Not a material reduction,
  so the default stays k = 0. (Before the edge-aware change k = 2 removed 32 of 64 steps cells,
  but only indirectly, through dilation of extra step-lethal cells over borderline rough cells.)
- **Limitation (measured):** texture whose own height jumps reach 0.10 m masks itself. In the
  rough family (GT, seeds 0–9), 1340 of 7122 cells with raw roughness ≥ 1 cm are zeroed
  (median 0.025 m, max 0.055 m), including the 2 cells that were rough-lethal. Because GT uses the
  same function, the core false-free metric cannot show this loss. Texture within ~0.3 m of a
  ≥ 0.10 m edge is ignored as well; the generator does not overlap features.

## D15. Known residual: low-box faces perceived at about half height
- **Observation:** at the edge of a box only slightly taller than step_crit (0.22–0.27 m), cells
  partly covered by the box average ground and top points, so the perceived jump is about half
  the true one (perceived step median 11.42 cm on 15 core false-free cells whose GT step is 2–5 cm
  over the 20 cm limit). flat_obstacles at 6c4fd3f: 15 of 11,546 GT core cells (0.002) before
  the edge-aware roughness change; 15 of 8610 after.
- **Not fixed here:** the 0.2 m step pairs (D13) recover full jumps only where the edge falls
  inside one cell; z_max/z_min would track noise extremes (D13). The GT convention (box top if the
  box covers the cell centre) also differs from perception's cell average.
- **Impact:** to be measured by the M3 path evaluator as the GT collision rate of planned paths
  (whether a planner actually routes through these cells), not just as a cell rate.

## D16. Planner design
- **Decision:** grid A* on the 0.1 m traversability map (`planning/astar.py`): 8-connected, no
  corner cutting (a diagonal is refused if either orthogonal cell it passes is lethal); edge cost
  step_len · (1 + α · (c_u + c_v) / 2) with c ∈ [0, 1) the traversability cost and α ≥ 0
  (`DEFAULT_ALPHA` = 4.0, illustrative, swept in M4); octile heuristic in metres (admissible and
  consistent: every edge costs at least its length); deterministic ties (f, then h, then
  insertion order). Then cost-preserving shortcutting (`planning/smooth.py`): a sub-path is
  replaced by a straight segment only if the segment touches no lethal cell (supercover, corners
  included) and its line-integral cost under the same model is ≤ the sub-path's.
- **Alternatives:** 4-connected grid (longer, blockier paths); Theta*/any-angle search; plain
  line-of-sight smoothing; costs as hard thresholds only.
- **Reason:** the edge model trades metres against terrain cost with one interpretable knob;
  line-of-sight smoothing would cut straight back through costly-but-passable terrain that A*
  had deliberately avoided; the cost-preserving rule cannot do that, and never increases cost.
- **Evidence:** tests/test_astar.py (exact hand-derived costs; equals scipy Dijkstra on random
  12×12 grids within 1e-9) and tests/test_smooth.py (never crosses lethal cells, never increases
  cost, collapses a free path to one segment, keeps a detour around a costly band). Runtime on
  the four rendered scenarios (`scripts/render_plans.py`, A* + smoothing, single process):
  4.6–12.7 ms per plan, 340–2063 expansions.

## D17. Planner comparison design
- **Decision:** three planners differing only in their input (`planning/planners.py`):
  TerraNav = perceived traversability, graded (cost-weighted); Baseline-Occ = perceived
  traversability, ungraded (non-lethal cells cost 0: shortest collision-free path on the same
  lethal mask); Oracle = ground-truth traversability, graded. Same A*, smoothing and α.
- **What each pair isolates:** TerraNav vs Baseline-Occ: the value of graded costs given the same
  perception (same lethal mask). TerraNav vs Oracle: the cost of imperfect perception (noise,
  occlusion, unknown = blocked) with the same planner. Baseline-Occ vs Oracle: both effects
  together.
- **Reason:** changing one factor per pair makes differences attributable.
- **Evidence:** TBD (M3-B evaluator).

## D18. Evaluation must be independent of the traversability definitions
- **Decision:** the M3-B path evaluator will judge paths against the simulated world directly
  (heightfield and boxes along the swept robot footprint: true heights, slopes, step heights,
  box contact), not against GT traversability computed by the same feature/lethal functions the
  planners use.
- **Alternatives:** score paths by GT traversability (cost, lethal mask) only.
- **Reason:** a metric built from the same definitions cannot detect a flaw in them. Motivating
  case at b286a6b: edge-aware roughness (D14) zeroes 19% of textured GT cells in the rough
  family, including both formerly rough-lethal cells; because GT traversability uses the same
  function, the core false-free metric showed nothing.
- **Evidence:** D14 limitation measurement; evaluator TBD in M3-B.
