# Interview notes

Every number here is copied from a committed source, cited inline:
**[S]** = [results/0cf7be5_96a4e7ffad40/summary.md](../results/0cf7be5_96a4e7ffad40/summary.md)
(held-out benchmark, seeds 1000–1019), **[D#]** = [decisions.md](decisions.md) entry (dev seeds
0–9 unless stated), **[P]** = [benchmark_protocol.md](benchmark_protocol.md),
**[SC]** = [img/showcase.json](img/showcase.json) at commit baafd60. Numbers marked *derived* are
arithmetic on cited numbers; the README's *Derived numbers* section shows the arithmetic.

## Pitches

**30 seconds.** TerraNav is a software-only terrain-navigation pipeline. A simulated depth camera
scans a rough, sloped, stepped world; I turn the depth images into a 2.5D elevation map,
compute slope, roughness and step features, and plan with A* on a graded traversability cost.
Paths are scored by an independent evaluator against the raw world. On 100 held-out scenarios
it cut rough exposure in the rough family by about 81 % (*derived*, [S] §a: 0.1186 → 0.0228 m²)
for 1–2 % more path length (*derived*), with zero violations [S] §c.

**2 minutes.** Add to the above:
- *Sensor boundary.* Perception only sees DepthFrames (depth, intrinsics, known pose) and the
  noise model; a test forbids it from importing the simulator [D7]. That makes the results honest
  and makes DepthFrame the plug-in point for stereo, lidar or RGB depth.
- *Evaluation design.* Three planners that differ in one input each: TerraNav (perceived map,
  graded), Baseline-Occ (same map, occupancy only), Oracle (true map, graded) [D17]. The
  evaluator never uses the traversability definitions [D18, D19].
- *Protocol.* All tuning on dev seeds 0–9; the held-out benchmark was pre-registered and run once
  [P]. Sanity invariant: at α = 0 TerraNav equals Baseline-Occ, 0 mismatches out of 100 [S] §e.
- *Honest finding.* Baseline-Occ also had zero violations [S] §a/§c, so here the gain is less
  terrain effort, not safety. Most of the engineering went into failures that the evaluation
  exposed: invented box tops, half-height box edges, roughness counting edges, paths grazing boxes.

## Numbers cheat sheet

| Claim | Number | Source |
|---|---|---|
| Held-out scenarios | 100 (5 families × 20, seeds 1000–1019) | [S] header, [P] |
| Rough exposure, rough family, TerraNav vs Baseline-Occ means | 0.0228 vs 0.1186 m² (−81 %, *derived*) | [S] §a |
| Paired rough-exposure diff, rough | −0.0958 [−0.1335, −0.0629], 18/1/1 | [S] §b |
| Paired max-rise diff | slopes −0.0420, rough −0.0254, steps −0.0743, mixed −0.0353 m | [S] §b |
| Max rise reduction, terrain families | −80 to −96 % (*derived* from §a means) | [S] §a |
| Paired length cost | +0.1563 (rough), +0.2098 (slopes), +0.2345 (steps), +0.1166 m (mixed) | [S] §b |
| Length cost relative | +1.0 to +2.1 % (*derived*: §b ÷ §d Baseline-Occ mean length) | [S] §b, §d |
| Violations | none, every planner, every family | [S] §c |
| Success rate | 1.00 everywhere | [S] §a |
| α = 0 invariant | 0 mismatches out of 100 | [S] §e |
| Perception latency (serial median) | 28.5–33.6 ms | [S] §f |
| Features + traversability (serial median) | 6.0–6.1 ms | [S] §f |
| TerraNav planning (serial median) | 6.8–8.3 ms | [S] §f |
| Per scenario, excluding simulation | ≈ 42–47 ms (*derived*: sum of medians) | [S] §f |
| Benchmark wall time | 376.4 s | [S] header |
| TerraNav − Oracle length, flat_obstacles | +0.1501 [+0.1024, +0.1971] | [S] §b |
| TerraNav − Oracle rough exposure, rough | +0.0164 [+0.0063, +0.0299] | [S] §b |
| False-lethal rate (dev) | 0.326 / 0.294 / 0.264 / 0.294 / 0.298 | [D20] |
| Showcase (rough dev seed 4) lengths | TerraNav 11.88, Baseline-Occ 10.92, Oracle 11.05 m (rounded) | [SC] |
| Showcase rough exposure | TerraNav 0.0, Baseline-Occ 0.123 m² (rounded), Oracle 0.0 | [SC] |
| Showcase rank | 6th of 10 dev seeds; diff −0.1226 m² (rounded) | [SC] |

## Stories (situation / problem / evidence / fix / result)

**1. Gap filling invented box tops [D8, D12].**
- *Situation:* scan lines and dropout leave small holes; unknown = blocked, so I filled gaps ≤ 0.3 m.
- *Problem:* box tops above the camera are never seen; a box shows as an observed rim around
  unknown cells, and filling interpolated ground height into the box.
- *Evidence:* filled-cell |z_mean − GT| p95 0.47 m (flat_obstacles), 0.26 m (mixed) [D8].
- *Fix:* fill only if the window's observed heights span ≤ 0.10 m, later AND a planar fit with
  residual ≤ 0.02 m [D12]. A residual-only guard was tried and rejected: under it 27 of 28
  box-top cells next to core false-free cells were FILLED, 0.237 m low [D12].
- *Result:* p95 0.0028 m (flat_obstacles) with the span guard [D8]; 0.0027 m with the AND
  guard at 6c4fd3f [D12].

**2. The robot was stranded on its own footprint [D5, D9].**
- *Situation:* flank poses look at the corridor from the side; the start area is known only from
  proprioception (footprint).
- *Problem:* no flank pose saw the ground next to the robot, so with unknown = blocked the
  footprint was an island.
- *Evidence:* start→goal known-connectivity on mixed seeds 0 and 1: False with "flank" [D9].
- *Fix:* add a start pose (frame 0); its nearest visible ground is ~0.285 m ahead, inside the
  0.4 m footprint [D9].
- *Result:* True on both seeds; connected 1.00 in all five families [D9].

**3. Low boxes looked half as tall [D15, D20].**
- *Situation:* boxes just over the 0.20 m step limit.
- *Problem:* a box face inside a 0.1 m cell puts ground and top points in one cell; the mean sits
  halfway, so the jump looks about half as high and the edge is only costed, not lethal.
- *Evidence:* perceived step median 11.42 cm on 15 core false-free cells whose GT step is 2–5 cm
  over the limit [D15]; the evaluator then found real collisions [D20].
- *Fix:* in-cell span step: a cell's own z_max − z_min counts as a step when it exceeds a
  noise-scaled threshold (K_SPAN = 2.0) [D20].
- *Result:* core false-free 0.000 (8610 GT core cells) [D15 update]; false-lethal unchanged on the
  three diagnosed maps (2202 / 1454 / 1802 corridor cells) [D20].

**4. Roughness was counting edges [D14, D18].**
- *Situation:* roughness = residual of a local plane fit.
- *Problem:* a plane fitted across a sub-critical terrace edge reports the edge as texture,
  creating a band of borderline "rough" cells.
- *Evidence:* steps family core false-free 64 / 370 (0.173); all 64 were roughness-only cells [D14].
- *Fix:* zero roughness where the fit window contains a significant ≥ 0.10 m jump; edges are the
  step feature's job [D14].
- *Result:* 0 / 1 (0.000). Cost: 1340 of 7122 textured GT cells in the rough family are zeroed,
  invisible to a GT-based metric. That gap is why the evaluator is independent [D14, D18].

**5. Paths grazed boxes, and the evaluator caught it [D19, D20].**
- *Situation:* first planning evaluation with an evaluator that measures box clearance directly.
- *Problem:* cell-level metrics looked fine, but paths passed closer than the robot radius.
- *Evidence:* at aeb08ea, collisions: terranav mixed seed 3 (clearance 0.239 m), baseline_occ
  flat_obstacles seed 0 (0.232 m) and mixed seed 3 (0.178 m), radius 0.3 m [D20].
- *Fix:* the vertical-surface (in-cell span) step of story 3; considered and rejected: a blanket
  extra safety margin [D20].
- *Result:* clearances 0.239 → 0.387, 0.232 → 0.371, 0.178 → 0.357 m [D20]; family collision
  rate 0.00 for every planner at 7426227 [D15 update, D20]; held-out: no violations [S] §c.

## Likely questions

1. **Why a depth camera?** One sensor with a known noise model keeps the chain checkable against
   ground truth; RGB, lidar and multi-sensor fusion each add error sources without testing them
   [D6]. DepthFrame is the extension point for other sensors [D7].
2. **Why no ground segmentation?** The world has no overhangs, so one height distribution per
   cell carries what the cost needs; segmentation adds thresholds and failure modes on ramps and
   mesas [D3]. Observed-cell height error median 0.0003 m [D3].
3. **Why unknown = lethal?** Optimistic unknowns hide unseen obstacles [D4]. The price is a
   conservative map (false-lethal 0.264–0.326, dev [D20]) and longer paths than Oracle
   (flat_obstacles +0.1501 m, [S] §b).
4. **How did you avoid overfitting?** All tuning on dev seeds 0–9; held-out seeds 1000–1019 were
   pre-registered, never generated before the run, and the run is pinned by commit and config
   hash [P]. α was fixed at 4 on dev seeds and the sweep is sensitivity only [P, D16].
5. **What does the Oracle comparison show?** The cost of imperfect perception with the same
   planner [D17]: TerraNav is +0.1501 m longer in flat_obstacles and has +0.0164 m² more rough
   exposure in rough than Oracle [S] §b. In the sweep TerraNav at α = 4 lands between Oracle's
   α = 2 and α = 4 [S] §d.
6. **Why does Baseline-Occ have zero violations too?** The lethal mask already excludes what the
   evaluator calls a violation; the graded cost buys less rough ground and lower rises, not
   safety, in this benchmark [S] §a, §c.
7. **Is the improvement statistically meaningful?** Paired bootstrap CIs per family: rough
   exposure in rough [−0.1335, −0.0629], 18/1/1 better/equal/worse [S] §b. Mixed rough exposure
   reaches 0 ([−0.0267, +0.0000]), so I don't claim it. No multiple-comparison correction [P].
8. **Why α = 4, and how sensitive?** Chosen on dev seeds [D16]; on held-out data rough exposure
   keeps falling past α = 4 (0.0228 → 0.0108 → 0.0028 at α = 8, 16) while lengths grow [S] §d.
9. **Why an independent evaluator?** A metric built from the same definitions cannot detect a
   flaw in them: edge-aware roughness zeroed both rough-lethal GT cells and the GT metric showed
   nothing [D18]. The evaluator measures clearance, rise, slope and rough-patch length on the raw
   world [D19].
10. **Why cost-preserving shortcutting instead of line-of-sight smoothing?** Line-of-sight would
    cut back through costly but passable terrain A* had avoided; the rule only accepts a shortcut
    whose cost is no higher [D16].
11. **How is sensor noise handled?** Roughness subtracts the plane fit's noise variance; a jump
    counts as a step only if it is 3σ significant; steps use cell pairs up to 0.2 m apart so an
    edge inside a cell isn't split [D11, D13].
12. **Why an 85° grazing cutoff?** At 80° flat ground was visible only to 2.8 m from a 0.5 m mount,
    5.7 m at 85° [D1].
13. **Why the flank scan?** Line poses sit inside obstacles or look along them; flank poses see the
    corridor from the side: corridor coverage 0.680 vs 0.379 (flat_obstacles) [D2].
14. **How do you know perception doesn't cheat?** It may not import the simulator (architecture
    test); the only non-sensor input, footprint heights, is passed in explicitly [D5, D7].
15. **How fast is it?** Serial medians: perception 28.5–33.6 ms, features + traversability
    6.0–6.1 ms, TerraNav planning 6.8–8.3 ms per scenario, dev seeds 0–4 [S] §f. The ray-cast
    simulation is not part of the robot-side pipeline and its timings in raw.jsonl were measured
    under parallel load, so I don't quote them as latency [P].
16. **What changes on a real robot?** Poses become estimates (localisation error), the world moves,
    the robot must rescan and replan while driving, and the illustrative limits need a vehicle
    model [D6, D10]. Footprint heights would come from proprioception [D5].
17. **How would you add RGB?** A monocular or stereo depth estimator that outputs DepthFrames with
    its own noise model; perception and planning stay unchanged [D7].
18. **Why a grid A* and not any-angle?** Simple, optimal on the grid graph and deterministic; the
    cost is 45° headings before shortcutting. Theta* or Field D* would remove that, and the Oracle
    ratios compare planners under the same limitation [D16].
19. **What would you do with more time?** Closed-loop replanning; RGB depth through DepthFrame;
    physics-based evaluation; any-angle planning.
20. **What is the biggest weakness?** Scan once, plan once with known poses [D6]: no recovery from
    what the scan missed, so unknown = blocked makes the map conservative (false-lethal up to
    0.326 [D20]), and every threshold is illustrative [D10].
21. **How did you pick the showcase?** By a rule fixed before looking: upper median of the paired
    rough-exposure difference over dev seeds 0–9; seed 4 ranked 6th of 10 [SC].

## Limitations to volunteer

- Scope: one simulated depth camera, known poses, static world, scan once and plan once; no
  replanning, control or localisation [D6].
- All thresholds, weights and robot limits are illustrative, not from a vehicle model [D10].
- Baseline-Occ was also violation-free [S] §c: the benchmark shows a terrain-effort gain, not a
  safety gain.
- 20 scenarios per family; CIs per family and metric, no multiple-comparison correction [P].
- Conservative map: false-lethal 0.264–0.326 on dev seeds [D20]; longer than Oracle in
  flat_obstacles (+0.1501 m, [S] §b).
- Edge-aware roughness ignores texture near ≥ 0.10 m edges (1340 of 7122 cells zeroed) [D14].
- Evaluator gaps: 40–45° sustained slopes would read as sub-limit steps; rough exposure uses the
  nominal rms [D19].
- Latencies are serial medians on dev seeds 0–4, summed per stage; not an end-to-end
  measurement [S] §f.
- The showcase's detour (11.88 vs 10.92 m [SC]) is longer than the rough family's mean paired
  length cost (+0.1563 m [S] §b).
