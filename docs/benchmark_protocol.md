# Held-out benchmark protocol (pre-registration)

Written before the benchmark is run. The run is `make benchmark` (`scripts/benchmark.py`),
pinned by git commit and configuration hash (`terranav.sim.benchmark.config_hash`); its output
directory is `results/<git-short>_<config-hash>/` and is committed as produced.

## Dev seeds vs held-out seeds
- **Dev seeds 0..9** (per family) were used for **every** design and tuning decision, with the
  dev-seed tables in `out/` and `docs/decisions.md`: scan strategy flank+start, k = 7 (D2, D9);
  85° grazing cutoff (D1); fill guards, span 0.10 m AND plane residual 0.02 m (D8, D12);
  noise-aware features, 0.2 m step pairs, significance gate, UCB k = 0 (D11, D13, D14);
  edge-aware roughness (D14); in-cell span step, K_SPAN = 2 (D20); critical values slope 30°,
  roughness 0.05 m, step 0.20 m, weights 0.4 / 0.3 / 0.3, robot radius 0.3 m (D10);
  α = 4 (D16); evaluator limits and scales: 40° discontinuity, 0.1 m slope scale (D19, D21).
- **Held-out seeds 1000..1019** per family, 5 families, **100 scenarios**. They were never
  generated, rendered or inspected before the run (tests and smoke runs use dev seeds only).
- No parameter is changed after seeing held-out results. Any deviation from this protocol is
  reported in the results, with the reason.

## Comparisons and metrics
- **Primary:** TerraNav vs Baseline-Occ at **α = 4** (same perception, graded vs occupancy-only).
- **Secondary:** TerraNav vs Oracle at α = 4 (cost of imperfect perception).
- **Primary metrics** (independent evaluator, `terranav.sim.evaluate`, D19): violation rate
  (collision, rise > 0.20 m, smoothed slope > 30°); rough exposure (m²); max rise (m); path length
  and its ratio to Oracle's at the same α.
- **Statistics:** per family, paired differences (A − B) over scenarios where both planners
  succeed, mean with 95% percentile bootstrap CI (2000 resamples, fixed seed), plus counts of
  better / equal / worse. 20 scenarios per family; no correction for multiple comparisons, so
  CIs are read per family and metric, not as a joint test.
- **Definition-dependent** GT traversability cost is reported separately and is not a primary
  metric (D18).
- **Sensitivity:** α ∈ {0, 0.5, 1, 2, 4, 8, 16} for TerraNav and Oracle (Baseline-Occ has no α).
  Reported as trade-off curves and tables; **not used to choose α** (α = 4 was fixed on dev seeds).

## Failures and how they are reported
- A **failure** is a planner returning no path (start and goal not connected through non-lethal
  cells of its map). Success rate is reported per family × planner, and every failure is listed
  (family, seed, planner, α).
- A **violation** is a successful path that breaks a physical limit; every violating path is
  listed with its reasons. Violation rates are over all scenarios (a failure is not a violation).
- Failed plans are excluded from means and from paired comparisons; the n of every paired
  comparison is reported.
- **Sanity invariant:** TerraNav at α = 0 must reproduce Baseline-Occ's path exactly in every
  scenario (identical edge costs). Mismatches are counted and listed; any mismatch means the
  run is invalid and is reported as such, not silently fixed. Scenarios where both fail are
  counted separately (the match is vacuous there).
- Per-stage timings in the raw records are measured inside parallel workers and are **not**
  latency; serial latency tables are included only if measured at the same commit.
