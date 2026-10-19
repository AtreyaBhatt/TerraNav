"""Benchmark figures: per-family alpha trade-off curves and the headline paired differences."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terranav.sim.plan_eval import paired_bootstrap_ci

# y-axis metric per family: rough exposure where rough patches exist, max rise otherwise
Y_METRIC = {"rough": "rough_exposure_m2", "mixed": "rough_exposure_m2"}
Y_LABEL = {"rough_exposure_m2": "rough exposure [m^2]", "max_rise_m": "max rise [m]"}
COLOURS = {"terranav": "tab:green", "oracle": "tab:purple", "baseline_occ": "tab:blue"}
METRICS = (
    ("rough_exposure_m2", "rough exposure [m^2]"),
    ("max_rise_m", "max rise [m]"),
    ("length_m", "length [m]"),
)


def _vals(records, family, planner, alpha, key) -> np.ndarray:
    return np.array(
        [
            r["metrics"][key]
            for r in records
            if r["family"] == family
            and r["planner"] == planner
            and (planner == "baseline_occ" or r["alpha"] == alpha)
            and r["success"]
            and r["metrics"][key] is not None
        ],
        dtype=float,
    )


def _mean_ci(v: np.ndarray, bcfg) -> tuple[float, float, float]:
    mu, lo, hi = paired_bootstrap_ci(v, bcfg.bootstrap_resamples, bcfg.bootstrap_seed)
    return mu, mu - lo, hi - mu


def plot_tradeoffs(records: list[dict], bcfg, out_dir: Path) -> None:
    """tradeoff_<family>.png: mean length vs mean y-metric, 95% CI error bars on both axes."""
    for fam in bcfg.families:
        key = Y_METRIC.get(fam, "max_rise_m")
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        for planner in ("terranav", "oracle"):
            pts = []
            for a in bcfg.alphas:
                x = _mean_ci(_vals(records, fam, planner, a, "length_m"), bcfg)
                y = _mean_ci(_vals(records, fam, planner, a, key), bcfg)
                pts.append((x, y, a))
            xs, ys = [p[0][0] for p in pts], [p[1][0] for p in pts]
            ax.errorbar(
                xs,
                ys,
                xerr=np.array([[p[0][1], p[0][2]] for p in pts]).T,
                yerr=np.array([[p[1][1], p[1][2]] for p in pts]).T,
                color=COLOURS[planner],
                marker="o",
                capsize=3,
                label=planner,
            )
            for (x, _, _), (y, _, _), a in pts:
                ax.annotate(
                    f"{a:g}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=COLOURS[planner],
                )
        x = _mean_ci(_vals(records, fam, "baseline_occ", None, "length_m"), bcfg)
        y = _mean_ci(_vals(records, fam, "baseline_occ", None, key), bcfg)
        ax.errorbar(
            [x[0]],
            [y[0]],
            xerr=[[x[1]], [x[2]]],
            yerr=[[y[1]], [y[2]]],
            color=COLOURS["baseline_occ"],
            marker="s",
            ms=8,
            capsize=3,
            ls="none",
            label="baseline_occ",
        )
        ax.set_xlabel("mean path length [m]")
        ax.set_ylabel(f"mean {Y_LABEL[key]}")
        ax.set_title(f"{fam}: trade-off over alpha (labels), 95% CI error bars")
        ax.legend()
        fig.savefig(Path(out_dir) / f"tradeoff_{fam}.png", dpi=100)
        plt.close(fig)


def plot_headline(records: list[dict], bcfg, out_dir: Path) -> None:
    """headline.png: per family, TerraNav - Baseline-Occ and TerraNav - Oracle at the default
    alpha, paired mean differences with 95% bootstrap CIs."""
    a = bcfg.default_alpha
    fams = list(bcfg.families)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.5), layout="constrained")
    for ax, (key, label) in zip(axes, METRICS, strict=True):
        for k, other in enumerate(("baseline_occ", "oracle")):
            mus, errs = [], []
            for fam in fams:

                def by_seed(planner, fam=fam, key=key):
                    return {
                        r["seed"]: r["metrics"][key]
                        for r in records
                        if r["family"] == fam
                        and r["planner"] == planner
                        and (planner == "baseline_occ" or r["alpha"] == a)
                        and r["success"]
                        and r["metrics"][key] is not None
                    }

                tn, ot = by_seed("terranav"), by_seed(other)
                d = np.array([tn[s] - ot[s] for s in tn if s in ot], dtype=float)
                mu, lo, hi = paired_bootstrap_ci(d, bcfg.bootstrap_resamples, bcfg.bootstrap_seed)
                mus.append(mu)
                errs.append((mu - lo, hi - mu))
            x = np.arange(len(fams)) + (k - 0.5) * 0.25
            ax.errorbar(
                x,
                mus,
                yerr=np.array(errs).T,
                fmt="o",
                capsize=4,
                color=COLOURS[other],
                label=f"terranav - {other}",
            )
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(fams)), fams, rotation=20)
        ax.set_title(f"{label}: paired difference (lower = TerraNav better)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Paired differences at alpha = {a:g}, 95% bootstrap CIs")
    fig.savefig(Path(out_dir) / "headline.png", dpi=100)
    plt.close(fig)
