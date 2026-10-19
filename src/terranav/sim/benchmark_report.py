"""summary.md for a benchmark run, rendered from the raw records alone."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from terranav.sim.plan_eval import compare_counts, paired_bootstrap_ci

if TYPE_CHECKING:
    from terranav.sim.benchmark import BenchmarkConfig

PLANNERS = ("terranav", "baseline_occ", "oracle")
PAIRS = (("terranav", "baseline_occ"), ("terranav", "oracle"))
# (metric key, label); "violation" is the 0/1 indicator of any violation
PAIRED = (
    ("violation", "violation"),
    ("rough_exposure_m2", "rough exposure m2"),
    ("max_rise_m", "max rise m"),
    ("length_m", "length m"),
)
LATENCY_FILES = ("eval_map_latency.txt", "eval_trav_latency.txt", "eval_plan_latency.txt")


def _select(records, family, planner, alpha):
    """Records of one family / planner; Baseline-Occ has no alpha (alpha argument ignored)."""
    return {
        r["seed"]: r
        for r in records
        if r["family"] == family
        and r["planner"] == planner
        and (planner == "baseline_occ" or r["alpha"] == alpha)
    }


def _value(rec: dict, key: str) -> float:
    if key == "violation":
        return float(bool(rec["metrics"]["violations"]))
    v = rec["metrics"].get(key)
    return float("nan") if v is None else float(v)


def _mean(vals) -> str:
    vals = [v for v in vals if not math.isnan(v)]
    return f"{np.mean(vals):.4f}" if vals else "n/a"


def _ok(sel: dict) -> list[dict]:
    return [r for r in sel.values() if r["success"]]


def headline(records, bcfg) -> list[str]:
    a = bcfg.default_alpha
    lines = [
        f"## (a) Headline, alpha = {a:g}",
        "",
        (
            "| family | planner | n | success | violation | collision | rise | slope | len/Oracle "
            "median | rough exposure m2 | max rise m | max slope deg | gt_cost* |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam in bcfg.families:
        oracle = _select(records, fam, "oracle", a)
        for p in PLANNERS:
            sel = _select(records, fam, p, a)
            ok, n = _ok(sel), len(sel)

            def rate(kind, ok=ok, n=n) -> str:
                hit = sum(
                    kind in r["metrics"]["violations"] if kind else bool(r["metrics"]["violations"])
                    for r in ok
                )
                return f"{hit / n:.2f}" if n else "n/a"

            ratios = [
                r["metrics"]["length_m"] / oracle[s]["metrics"]["length_m"]
                for s, r in sel.items()
                if r["success"] and s in oracle and oracle[s]["success"]
            ]
            lines.append(
                f"| {fam} | {p} | {n} | {len(ok) / n if n else float('nan'):.2f} | {rate(None)} | "
                f"{rate('collision')} | {rate('rise')} | {rate('slope')} | "
                f"{f'{np.median(ratios):.3f}' if ratios else 'n/a'} | "
                f"{_mean([_value(r, 'rough_exposure_m2') for r in ok])} | "
                f"{_mean([_value(r, 'max_rise_m') for r in ok])} | "
                f"{_mean([_value(r, 'max_slope_deg') for r in ok])} | "
                f"{_mean([float(r['gt_cost_m']) for r in ok])} |"
            )
    return lines + [
        "",
        "Rates over all scenarios of the family (a failed plan counts as not violating).",
        "\\* gt_cost: integral of GT traversability cost along the path; DEFINITION-DEPENDENT",
        "(uses the same traversability definitions as the planners). All other metrics come from",
        "the independent evaluator (terranav.sim.evaluate).",
        "",
    ]


def paired(records, bcfg) -> list[str]:
    a = bcfg.default_alpha
    lines = [
        f"## (b) Paired differences at alpha = {a:g} (A - B; lower is better)",
        "",
        (
            f"Scenarios where both succeed. 95% CI: percentile bootstrap, {bcfg.bootstrap_resamples} "
            f"resamples, seed {bcfg.bootstrap_seed}. better/equal/worse: A lower / within 1e-9 / higher."
        ),
        "",
        "| family | A - B | n | metric | mean diff | 95% CI | better/equal/worse |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam in bcfg.families:
        for pa, pb in PAIRS:
            sa, sb = _select(records, fam, pa, a), _select(records, fam, pb, a)
            both = sorted(s for s in sa if s in sb and sa[s]["success"] and sb[s]["success"])
            for key, label in PAIRED:
                va = np.array([_value(sa[s], key) for s in both])
                vb = np.array([_value(sb[s], key) for s in both])
                mu, lo, hi = paired_bootstrap_ci(
                    va - vb, bcfg.bootstrap_resamples, bcfg.bootstrap_seed
                )
                b, e, w = compare_counts(va, vb)
                lines.append(
                    f"| {fam} | {pa} - {pb} | {len(both)} | {label} | {mu:+.4f} | "
                    f"[{lo:+.4f}, {hi:+.4f}] | {b}/{e}/{w} |"
                )
    return lines + [""]


def failures(records) -> list[str]:
    lines = ["## (c) Failures and violations", ""]
    for r in sorted(
        records, key=lambda r: (r["family"], r["seed"], r["planner"], r["alpha"] or -1)
    ):
        tag = f"{r['family']} seed {r['seed']} {r['planner']}"
        tag += f" alpha {r['alpha']:g}" if r["alpha"] is not None else ""
        if not r["success"]:
            lines.append(f"- {tag}: no path")
        elif r["metrics"]["violations"]:
            lines.append(f"- {tag}: {', '.join(r['metrics']['violations'])}")
    return lines + (["- none"] if len(lines) == 2 else []) + [""]


def sweep(records, bcfg) -> list[str]:
    lines = [
        "## (d) Alpha sweep (sensitivity only; alpha was fixed before the run)",
        "",
        "TerraNav length ratio is against Oracle at the same alpha (scenarios where both succeed).",
        "",
    ]
    for fam in bcfg.families:
        lines += [
            f"### {fam}",
            "",
            (
                "| alpha | TN len/Oracle | TN rough exp | TN max rise | TN viol | Oracle length m | "
                "Oracle rough exp | Oracle max rise | Oracle viol |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for a in bcfg.alphas:
            tn, orc = _select(records, fam, "terranav", a), _select(records, fam, "oracle", a)
            ratios = [
                tn[s]["metrics"]["length_m"] / orc[s]["metrics"]["length_m"]
                for s in tn
                if tn[s]["success"] and s in orc and orc[s]["success"]
            ]
            row = [f"{a:g}", _mean(ratios)]
            for sel, with_len in ((tn, False), (orc, True)):
                ok = _ok(sel)
                if with_len:
                    row.append(_mean([_value(r, "length_m") for r in ok]))
                row += [
                    _mean([_value(r, "rough_exposure_m2") for r in ok]),
                    _mean([_value(r, "max_rise_m") for r in ok]),
                    f"{np.mean([_value(r, 'violation') for r in ok]):.2f}" if ok else "n/a",
                ]
            lines.append("| " + " | ".join(row) + " |")
        bl = _ok(_select(records, fam, "baseline_occ", None))
        lines += [
            "",
            (
                f"Baseline-Occ (no alpha): mean length {_mean([_value(r, 'length_m') for r in bl])} m, "
                f"rough exposure {_mean([_value(r, 'rough_exposure_m2') for r in bl])}, max rise "
                f"{_mean([_value(r, 'max_rise_m') for r in bl])}, violation rate "
                f"{_mean([_value(r, 'violation') for r in bl])}."
            ),
            "",
        ]
    return lines


def invariant(records, bcfg) -> list[str]:
    lines = ["## (e) Sanity invariant", ""]
    if 0.0 not in bcfg.alphas:
        return lines + ["alpha = 0 not in the sweep: invariant not checked.", ""]
    tn0 = [r for r in records if r["planner"] == "terranav" and r["alpha"] == 0.0]
    bad = [r for r in tn0 if r.get("alpha0_matches_baseline") is not True]
    vacuous = sum(1 for r in tn0 if not r["success"])  # both failed: nothing to compare
    lines.append(
        "TerraNav at alpha = 0 must reproduce Baseline-Occ exactly. "
        + f"alpha = 0 invariant: {len(bad)} mismatches out of {len(tn0)} scenarios (expected 0); "
        + f"{vacuous} of them without a path (vacuous match)."
    )
    lines += [f"- mismatch: {r['family']} seed {r['seed']}" for r in bad]
    return lines + [""]


def read_latency_tables(latency_dir: str | Path, git: str) -> dict[str, str]:
    """Serial latency tables from latency_dir whose '# git:' line equals `git` exactly."""
    out = {}
    for name in LATENCY_FILES:
        path = Path(latency_dir) / name
        if not path.exists():
            continue
        text = path.read_text()
        heads = [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("# git:")]
        if heads and heads[0] == git:
            out[name] = text
    return out


def render_summary(records: list[dict], bcfg: BenchmarkConfig, meta: dict, latency: dict) -> str:
    lines = [
        "# TerraNav held-out benchmark",
        "",
        f"- git: `{meta.get('git')}`; config hash: `{meta.get('config_hash')}`",
        f"- started: {meta.get('started')}; wall time: {meta.get('wall_s', float('nan')):.1f} s",
        (
            f"- families: {', '.join(bcfg.families)}; seeds: {min(bcfg.seeds)}..{max(bcfg.seeds)} "
            f"({len(bcfg.seeds)} per family); alphas: {', '.join(f'{a:g}' for a in bcfg.alphas)}; "
            f"primary alpha: {bcfg.default_alpha:g}"
        ),
        "- protocol: docs/benchmark_protocol.md (pre-registered); all numbers measured.",
        "",
    ]
    lines += headline(records, bcfg) + paired(records, bcfg) + failures(records)
    lines += sweep(records, bcfg) + invariant(records, bcfg)
    lines += ["## (f) Latency", ""]
    if latency:
        lines.append("Serial latency tables measured at this commit (`make latency`):")
        for name, text in latency.items():
            lines += ["", f"### {name}", "", "```", text.rstrip(), "```"]
    else:
        lines.append(
            "latency not measured at this commit (no out/*latency*.txt with this git hash). "
            "Per-stage timings in raw.jsonl were measured under parallel load and are not latency."
        )
    return "\n".join(lines) + "\n"
