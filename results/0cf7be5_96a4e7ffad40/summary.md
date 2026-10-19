# TerraNav held-out benchmark

- git: `0cf7be5ce01d1ea2e5a5754140531f2dc62e5d1f`; config hash: `96a4e7ffad40`
- started: 2026-10-19T11:26:51; wall time: 376.4 s
- families: flat_obstacles, slopes, rough, steps, mixed; seeds: 1000..1019 (20 per family); alphas: 0, 0.5, 1, 2, 4, 8, 16; primary alpha: 4
- protocol: docs/benchmark_protocol.md (pre-registered); all numbers measured.

## (a) Headline, alpha = 4

| family | planner | n | success | violation | collision | rise | slope | len/Oracle median | rough exposure m2 | max rise m | max slope deg | gt_cost* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flat_obstacles | terranav | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.014 | 0.0000 | 0.0000 | 0.0000 | 0.2247 |
| flat_obstacles | baseline_occ | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.017 | 0.0000 | 0.0000 | 0.0000 | 0.3021 |
| flat_obstacles | oracle | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| slopes | terranav | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 | 0.0000 | 0.0026 | 2.2693 | 0.1003 |
| slopes | baseline_occ | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.985 | 0.0000 | 0.0445 | 14.0793 | 0.5868 |
| slopes | oracle | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 | 0.0000 | 0.0014 | 1.5862 | 0.0682 |
| rough | terranav | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.996 | 0.0228 | 0.0065 | 3.0592 | 0.1605 |
| rough | baseline_occ | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.976 | 0.1186 | 0.0319 | 10.2106 | 0.5264 |
| rough | oracle | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 | 0.0064 | 0.0013 | 1.7285 | 0.1078 |
| steps | terranav | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.004 | 0.0000 | 0.0045 | 0.0000 | 0.1009 |
| steps | baseline_occ | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.988 | 0.0000 | 0.0788 | 0.0000 | 0.6168 |
| steps | oracle | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0807 |
| mixed | terranav | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.010 | 0.0000 | 0.0015 | 0.8743 | 0.1289 |
| mixed | baseline_occ | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.002 | 0.0119 | 0.0368 | 3.5796 | 0.3568 |
| mixed | oracle | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 | 0.0015 | 0.0009 | 0.6939 | 0.0460 |

Rates over all scenarios of the family (a failed plan counts as not violating).
\* gt_cost: integral of GT traversability cost along the path; DEFINITION-DEPENDENT
(uses the same traversability definitions as the planners). All other metrics come from
the independent evaluator (terranav.sim.evaluate).

## (b) Paired differences at alpha = 4 (A - B; lower is better)

Scenarios where both succeed. 95% CI: percentile bootstrap, 2000 resamples, seed 20261018. better/equal/worse: A lower / within 1e-9 / higher.

| family | A - B | n | metric | mean diff | 95% CI | better/equal/worse |
|---|---|---|---|---|---|---|
| flat_obstacles | terranav - baseline_occ | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| flat_obstacles | terranav - baseline_occ | 20 | rough exposure m2 | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| flat_obstacles | terranav - baseline_occ | 20 | max rise m | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| flat_obstacles | terranav - baseline_occ | 20 | length m | -0.0192 | [-0.0595, +0.0176] | 10/0/10 |
| flat_obstacles | terranav - oracle | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| flat_obstacles | terranav - oracle | 20 | rough exposure m2 | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| flat_obstacles | terranav - oracle | 20 | max rise m | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| flat_obstacles | terranav - oracle | 20 | length m | +0.1501 | [+0.1024, +0.1971] | 1/0/19 |
| slopes | terranav - baseline_occ | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| slopes | terranav - baseline_occ | 20 | rough exposure m2 | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| slopes | terranav - baseline_occ | 20 | max rise m | -0.0420 | [-0.0531, -0.0303] | 19/1/0 |
| slopes | terranav - baseline_occ | 20 | length m | +0.2098 | [+0.0671, +0.3436] | 3/0/17 |
| slopes | terranav - oracle | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| slopes | terranav - oracle | 20 | rough exposure m2 | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| slopes | terranav - oracle | 20 | max rise m | +0.0012 | [+0.0002, +0.0025] | 0/13/7 |
| slopes | terranav - oracle | 20 | length m | +0.0019 | [-0.0430, +0.0407] | 10/0/10 |
| rough | terranav - baseline_occ | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| rough | terranav - baseline_occ | 20 | rough exposure m2 | -0.0958 | [-0.1335, -0.0629] | 18/1/1 |
| rough | terranav - baseline_occ | 20 | max rise m | -0.0254 | [-0.0347, -0.0166] | 20/0/0 |
| rough | terranav - baseline_occ | 20 | length m | +0.1563 | [+0.0613, +0.2602] | 4/0/16 |
| rough | terranav - oracle | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| rough | terranav - oracle | 20 | rough exposure m2 | +0.0164 | [+0.0063, +0.0299] | 0/12/8 |
| rough | terranav - oracle | 20 | max rise m | +0.0051 | [+0.0022, +0.0087] | 1/8/11 |
| rough | terranav - oracle | 20 | length m | -0.0633 | [-0.1164, -0.0141] | 14/0/6 |
| steps | terranav - baseline_occ | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| steps | terranav - baseline_occ | 20 | rough exposure m2 | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| steps | terranav - baseline_occ | 20 | max rise m | -0.0743 | [-0.0999, -0.0491] | 15/5/0 |
| steps | terranav - baseline_occ | 20 | length m | +0.2345 | [+0.1231, +0.3608] | 3/0/17 |
| steps | terranav - oracle | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| steps | terranav - oracle | 20 | rough exposure m2 | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| steps | terranav - oracle | 20 | max rise m | +0.0045 | [+0.0000, +0.0112] | 0/18/2 |
| steps | terranav - oracle | 20 | length m | +0.0646 | [-0.0447, +0.1920] | 6/0/14 |
| mixed | terranav - baseline_occ | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| mixed | terranav - baseline_occ | 20 | rough exposure m2 | -0.0119 | [-0.0267, +0.0000] | 3/17/0 |
| mixed | terranav - baseline_occ | 20 | max rise m | -0.0353 | [-0.0598, -0.0147] | 11/9/0 |
| mixed | terranav - baseline_occ | 20 | length m | +0.1166 | [+0.0197, +0.2402] | 5/0/15 |
| mixed | terranav - oracle | 20 | violation | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| mixed | terranav - oracle | 20 | rough exposure m2 | -0.0015 | [-0.0045, +0.0000] | 1/19/0 |
| mixed | terranav - oracle | 20 | max rise m | +0.0006 | [-0.0008, +0.0024] | 1/17/2 |
| mixed | terranav - oracle | 20 | length m | +0.1613 | [+0.0717, +0.2973] | 4/0/16 |

## (c) Failures and violations

- none

## (d) Alpha sweep (sensitivity only; alpha was fixed before the run)

TerraNav length ratio is against Oracle at the same alpha (scenarios where both succeed).

### flat_obstacles

| alpha | TN len/Oracle | TN rough exp | TN max rise | TN viol | Oracle length m | Oracle rough exp | Oracle max rise | Oracle viol |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0147 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |
| 0.5 | 1.0132 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |
| 1 | 1.0132 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |
| 2 | 1.0132 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |
| 4 | 1.0132 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |
| 8 | 1.0132 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |
| 16 | 1.0132 | 0.0000 | 0.0000 | 0.00 | 11.4691 | 0.0000 | 0.0000 | 0.00 |

Baseline-Occ (no alpha): mean length 11.6383 m, rough exposure 0.0000, max rise 0.0000, violation rate 0.0000.

### slopes

| alpha | TN len/Oracle | TN rough exp | TN max rise | TN viol | Oracle length m | Oracle rough exp | Oracle max rise | Oracle viol |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0301 | 0.0000 | 0.0445 | 0.00 | 10.5397 | 0.0000 | 0.0646 | 0.00 |
| 0.5 | 1.0250 | 0.0000 | 0.0387 | 0.00 | 10.5954 | 0.0000 | 0.0552 | 0.00 |
| 1 | 1.0172 | 0.0000 | 0.0212 | 0.00 | 10.7799 | 0.0000 | 0.0354 | 0.00 |
| 2 | 1.0027 | 0.0000 | 0.0084 | 0.00 | 10.9898 | 0.0000 | 0.0108 | 0.00 |
| 4 | 1.0004 | 0.0000 | 0.0026 | 0.00 | 11.0658 | 0.0000 | 0.0014 | 0.00 |
| 8 | 1.0025 | 0.0000 | 0.0001 | 0.00 | 11.0870 | 0.0000 | 0.0000 | 0.00 |
| 16 | 1.0001 | 0.0000 | 0.0000 | 0.00 | 11.1323 | 0.0000 | 0.0000 | 0.00 |

Baseline-Occ (no alpha): mean length 10.8579 m, rough exposure 0.0000, max rise 0.0445, violation rate 0.0000.

### rough

| alpha | TN len/Oracle | TN rough exp | TN max rise | TN viol | Oracle length m | Oracle rough exp | Oracle max rise | Oracle viol |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0230 | 0.1186 | 0.0319 | 0.00 | 10.5512 | 0.1952 | 0.0841 | 0.00 |
| 0.5 | 1.0213 | 0.0746 | 0.0246 | 0.00 | 10.5840 | 0.1802 | 0.0787 | 0.00 |
| 1 | 1.0132 | 0.0641 | 0.0196 | 0.00 | 10.7014 | 0.1188 | 0.0450 | 0.00 |
| 2 | 1.0038 | 0.0426 | 0.0126 | 0.00 | 10.8454 | 0.0498 | 0.0199 | 0.00 |
| 4 | 0.9943 | 0.0228 | 0.0065 | 0.00 | 11.0205 | 0.0064 | 0.0013 | 0.00 |
| 8 | 0.9972 | 0.0108 | 0.0032 | 0.00 | 11.0523 | 0.0005 | 0.0002 | 0.00 |
| 16 | 0.9862 | 0.0028 | 0.0005 | 0.00 | 11.2373 | 0.0000 | 0.0000 | 0.00 |

Baseline-Occ (no alpha): mean length 10.8009 m, rough exposure 0.1186, max rise 0.0319, violation rate 0.0000.

### steps

| alpha | TN len/Oracle | TN rough exp | TN max rise | TN viol | Oracle length m | Oracle rough exp | Oracle max rise | Oracle viol |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0243 | 0.0000 | 0.0788 | 0.00 | 10.7050 | 0.0000 | 0.1078 | 0.00 |
| 0.5 | 1.0216 | 0.0000 | 0.0545 | 0.00 | 10.7616 | 0.0000 | 0.0935 | 0.00 |
| 1 | 1.0122 | 0.0000 | 0.0333 | 0.00 | 10.9076 | 0.0000 | 0.0587 | 0.00 |
| 2 | 1.0086 | 0.0000 | 0.0221 | 0.00 | 10.9885 | 0.0000 | 0.0181 | 0.00 |
| 4 | 1.0060 | 0.0000 | 0.0045 | 0.00 | 11.1321 | 0.0000 | 0.0000 | 0.00 |
| 8 | 1.0096 | 0.0000 | 0.0000 | 0.00 | 11.1855 | 0.0000 | 0.0000 | 0.00 |
| 16 | 1.0107 | 0.0000 | 0.0000 | 0.00 | 11.1979 | 0.0000 | 0.0000 | 0.00 |

Baseline-Occ (no alpha): mean length 10.9622 m, rough exposure 0.0000, max rise 0.0788, violation rate 0.0000.

### mixed

| alpha | TN len/Oracle | TN rough exp | TN max rise | TN viol | Oracle length m | Oracle rough exp | Oracle max rise | Oracle viol |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0250 | 0.0119 | 0.0368 | 0.00 | 10.9540 | 0.0322 | 0.0777 | 0.00 |
| 0.5 | 1.0243 | 0.0110 | 0.0272 | 0.00 | 10.9663 | 0.0254 | 0.0744 | 0.00 |
| 1 | 1.0166 | 0.0031 | 0.0162 | 0.00 | 11.0662 | 0.0060 | 0.0500 | 0.00 |
| 2 | 1.0126 | 0.0031 | 0.0064 | 0.00 | 11.1315 | 0.0023 | 0.0086 | 0.00 |
| 4 | 1.0148 | 0.0000 | 0.0015 | 0.00 | 11.1868 | 0.0015 | 0.0009 | 0.00 |
| 8 | 1.0133 | 0.0000 | 0.0003 | 0.00 | 11.2188 | 0.0015 | 0.0004 | 0.00 |
| 16 | 1.0112 | 0.0000 | 0.0000 | 0.00 | 11.2713 | 0.0000 | 0.0000 | 0.00 |

Baseline-Occ (no alpha): mean length 11.2315 m, rough exposure 0.0119, max rise 0.0368, violation rate 0.0000.

## (e) Sanity invariant

TerraNav at alpha = 0 must reproduce Baseline-Occ exactly. alpha = 0 invariant: 0 mismatches out of 100 scenarios (expected 0); 0 of them without a path (vacuous match).

## (f) Latency

Serial latency tables measured at this commit (`make latency`):

### eval_map_latency.txt

```
# TerraNav perception latency (measured)
# git: 0cf7be5ce01d1ea2e5a5754140531f2dc62e5d1f
# started: 2026-10-19T11:12:05
# wall time: 299.4 s
# survey flank+start (k = 7), default SensorConfig, 0.1 m map grid
# fuse + fill_gaps + mark_footprint; serial (1 process), seeds 0..4
# per scenario: median of 3 repeats; machine: x86_64, 24 logical CPUs

family             median       p95
flat_obstacles     33.6ms    43.9ms
slopes             31.2ms    31.3ms
rough              29.5ms    31.7ms
steps              28.5ms    29.3ms
mixed              29.3ms    31.0ms
```

### eval_trav_latency.txt

```
# TerraNav traversability latency (measured)
# git: 0cf7be5ce01d1ea2e5a5754140531f2dc62e5d1f
# started: 2026-10-19T11:17:05
# wall time: 291.0 s
# survey flank+start (k = 7), default SensorConfig, default TraversabilityConfig, 0.1 m map grid
# terrain_features + traversability; serial, seeds 0..4
# per scenario: median of 3 repeats; machine: x86_64, 24 logical CPUs

family             median       p95
flat_obstacles      6.0ms     6.1ms
slopes              6.0ms     6.2ms
rough               6.1ms     6.2ms
steps               6.1ms     6.3ms
mixed               6.1ms     6.3ms
```

### eval_plan_latency.txt

```
# TerraNav planning latency (measured)
# git: 0cf7be5ce01d1ea2e5a5754140531f2dc62e5d1f
# started: 2026-10-19T11:21:56
# wall time: 294.4 s
# survey flank+start (k = 7), alpha = 4, default SensorConfig / TraversabilityConfig / PhysicalLimits, 0.1 m map grid
# plan() = A* + smoothing per planner; serial, seeds 0..4
# per scenario: median of 3 repeats; machine: x86_64, 24 logical CPUs

family          planner          median       p95
flat_obstacles  terranav          7.8ms     9.3ms
flat_obstacles  baseline_occ      7.0ms     7.3ms
flat_obstacles  oracle            8.4ms    12.1ms
slopes          terranav          6.8ms    11.3ms
slopes          baseline_occ      4.6ms     7.0ms
slopes          oracle            7.1ms    11.3ms
rough           terranav          8.3ms    11.2ms
rough           baseline_occ      3.7ms     4.6ms
rough           oracle           10.7ms    11.6ms
steps           terranav          7.6ms    10.4ms
steps           baseline_occ      4.5ms     9.9ms
steps           oracle            6.0ms    10.9ms
mixed           terranav          8.3ms     9.5ms
mixed           baseline_occ      4.6ms     6.6ms
mixed           oracle            9.1ms    11.9ms
```
