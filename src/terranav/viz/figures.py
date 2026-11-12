"""Presentation figures from a committed benchmark run (raw.jsonl only, no reruns).

Every plotted statistic is a paired difference over the scenarios where both planners succeed,
computed exactly as summary.md computes it (terranav.sim.benchmark_report: same record
selection, same seed order, same bootstrap resamples and seed), so the figures can be checked
number by number against the committed summary.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terranav.sim.benchmark import BenchmarkConfig
from terranav.sim.benchmark_report import PAIRED, _select, _value
from terranav.sim.plan_eval import paired_bootstrap_ci

# Categorical planner colours (CVD-checked as a set); also used by the 3D renders.
PLANNER_COLOURS = {"terranav": "#1baf7a", "baseline_occ": "#d03b2a", "oracle": "#2a78d6"}
PLANNER_LABELS = {"terranav": "TerraNav", "baseline_occ": "Baseline-Occ", "oracle": "Oracle"}
PLANNER_MARKERS = {"terranav": "o", "baseline_occ": "s", "oracle": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# (metric key, short panel title, y-axis label, plot unit per metric unit) of the headline
# panels; values are plotted in the panel's unit (max rise in cm)
HEADLINE_METRICS = (
    ("rough_exposure_m2", "Rough exposure", "paired difference [m²]", 1.0),
    ("max_rise_m", "Max rise", "paired difference [cm]", 100.0),
    ("length_m", "Path length", "paired difference [m]", 1.0),
)
HEADLINE_SCALE = {key: scale for key, _, _, scale in HEADLINE_METRICS}
ORACLE_NOTE = (
    "Oracle plans on the true map (no unknown cells), so at low alpha it can be shorter than\n"
    "Baseline-Occ and cross more rough terrain."
)
# y-axis metric of each trade-off figure (flat_obstacles has no terrain cost: no figure)
TRADEOFF_Y = {
    "rough": "rough_exposure_m2",
    "mixed": "rough_exposure_m2",
    "slopes": "max_rise_m",
    "steps": "max_rise_m",
}
TRADEOFF_Y_LABEL = {
    "rough_exposure_m2": "reduction in rough exposure vs Baseline-Occ [m²]",
    "max_rise_m": "reduction in max rise vs Baseline-Occ [m]",
}
DPI = 200
# PNG metadata pinned so that regenerated files are byte-identical
PNG_METADATA = {"Software": None}

Stat = tuple[float, float, float]  # (mean, CI low, CI high)


def load_run(run_dir: str | Path) -> tuple[list[dict], BenchmarkConfig, dict]:
    """(records, BenchmarkConfig, meta) of a results directory; the config is the run's own."""
    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "raw.jsonl").read_text().splitlines()]
    meta = json.loads((run_dir / "meta.json").read_text())
    raw = meta["configs"]["benchmark"]
    bcfg = BenchmarkConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    return records, bcfg, meta


def _paired_values(records, family, pa, pb, alpha, key) -> tuple[np.ndarray, np.ndarray]:
    """(values of A, values of B) over seeds where both succeed, in summary.md's seed order."""
    sa, sb = _select(records, family, pa, alpha), _select(records, family, pb, alpha)
    both = sorted(s for s in sa if s in sb and sa[s]["success"] and sb[s]["success"])
    return (
        np.array([_value(sa[s], key) for s in both]),
        np.array([_value(sb[s], key) for s in both]),
    )


def paired_diff(records, bcfg, family: str, pa: str, pb: str, alpha: float, key: str) -> Stat:
    """Mean of A - B and its 95% percentile bootstrap CI, exactly as in summary.md (b)."""
    va, vb = _paired_values(records, family, pa, pb, alpha, key)
    return paired_bootstrap_ci(va - vb, bcfg.bootstrap_resamples, bcfg.bootstrap_seed)


def headline_data(records, bcfg) -> dict[tuple[str, str, str], Stat]:
    """{(metric, other planner, family): TerraNav - other at the primary alpha}, in the panel's
    plot unit (metric value * HEADLINE_SCALE[metric])."""
    return {
        (key, other, fam): tuple(
            scale * v
            for v in paired_diff(records, bcfg, fam, "terranav", other, bcfg.default_alpha, key)
        )
        for key, _, _, scale in HEADLINE_METRICS
        for other in ("baseline_occ", "oracle")
        for fam in bcfg.families
    }


def tradeoff_data(records, bcfg, family: str) -> dict[str, dict[float, tuple[Stat, Stat]]]:
    """{planner: {alpha: (x, y)}} for TerraNav and Oracle against Baseline-Occ, paired by seed.

    x: mean extra path length, 100 * (L_planner - L_baseline) / L_baseline [%].
    y: mean reduction of TRADEOFF_Y[family], baseline - planner. It is the bootstrap of
       planner - baseline with mean and bounds negated, so at the primary alpha TerraNav's y is
       exactly minus summary.md's (b) entry.
    """
    key = TRADEOFF_Y[family]
    out: dict[str, dict[float, tuple[Stat, Stat]]] = {}
    for planner in ("terranav", "oracle"):
        out[planner] = {}
        for a in bcfg.alphas:
            lp, lb = _paired_values(records, family, planner, "baseline_occ", a, "length_m")
            x = paired_bootstrap_ci(
                100.0 * (lp - lb) / lb, bcfg.bootstrap_resamples, bcfg.bootstrap_seed
            )
            mu, lo, hi = paired_diff(records, bcfg, family, planner, "baseline_occ", a, key)
            out[planner][a] = (x, (-mu, -hi, -lo))
    return out


def sample_note(records, bcfg) -> str:
    """ "held-out seeds A-B, n = N per family", derived from (and checked against) the records."""
    seeds = sorted({r["seed"] for r in records})
    per_family = {
        f: len({r["seed"] for r in records if r["family"] == f and r["planner"] == "baseline_occ"})
        for f in bcfg.families
    }
    if len(set(per_family.values())) != 1:
        raise ValueError(f"unequal scenarios per family: {per_family}")
    if seeds != list(range(seeds[0], seeds[-1] + 1)):
        raise ValueError("seeds are not a contiguous range")
    n = next(iter(per_family.values()))
    return f"held-out seeds {seeds[0]}-{seeds[-1]}, n = {n} per family"


def caption(records, bcfg) -> str:
    """Trade-off footer caption, derived from (and checked against) the records."""
    tn0 = [r for r in records if r["planner"] == "terranav" and r["alpha"] == 0.0]
    if not tn0 or any(r.get("alpha0_matches_baseline") is not True for r in tn0):
        raise ValueError("alpha = 0 invariant does not hold in these records")
    return f"{sample_note(records, bcfg)}; TerraNav at alpha = 0 equals Baseline-Occ"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_headline(data: dict, bcfg, note: str, out_path: str | Path) -> Path:
    """Three panels of TerraNav - Baseline-Occ and TerraNav - Oracle per family, 95% CIs.
    note: sample description for the footer (sample_note)."""
    fams = list(bcfg.families)
    fig, axes = plt.subplots(1, len(HEADLINE_METRICS), figsize=(12, 4))
    for ax, (key, title, ylabel, _) in zip(axes, HEADLINE_METRICS, strict=True):
        ax.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
        for k, other in enumerate(("baseline_occ", "oracle")):
            stats = [data[(key, other, f)] for f in fams]
            x = np.arange(len(fams)) + (k - 0.5) * 0.28
            mu = np.array([s[0] for s in stats])
            err = np.array([[s[0] - s[1], s[2] - s[0]] for s in stats]).T
            ax.errorbar(
                x,
                mu,
                yerr=err,
                fmt=PLANNER_MARKERS[other],
                ms=6,
                capsize=3,
                lw=1.5,
                color=PLANNER_COLOURS[other],
                label=f"TerraNav − {PLANNER_LABELS[other]}",
                zorder=3,
            )
        ax.set_xticks(np.arange(len(fams)), [f.replace("_", "\n") for f in fams], fontsize=9)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.01,
        f"Mean paired difference at alpha = {bcfg.default_alpha:g} with 95% bootstrap CI; "
        f"below zero = TerraNav lower; {note}.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=DPI, metadata=PNG_METADATA)
    plt.close(fig)
    return out_path


# Candidate label offsets (points): rings around the point, 30 degrees apart. Labels sit on a
# short leader line, so close points cannot swap labels visually.
LABEL_OFFSETS_PT = tuple(
    (round(r * math.cos(math.radians(t)), 1), round(r * math.sin(math.radians(t)), 1))
    for r in (16, 24, 34)
    for t in range(0, 360, 30)
)


def _sample_boxes(polylines_px, r: float, step: float = 3.0) -> list:
    """Small boxes every ~step px along pixel-space polylines (obstacles for labels)."""
    out = []
    for line in polylines_px:
        for a, b in itertools.pairwise(np.asarray(line)):
            n = max(2, int(np.hypot(*(b - a)) / step))
            for x, y in a + np.linspace(0, 1, n)[:, None] * (b - a):
                out.append(matplotlib.transforms.Bbox([[x - r, y - r], [x + r, y + r]]))
    return out


def _place_labels(
    fig,
    ax,
    labels: list[tuple[tuple[float, float], str]],
    markers: list[tuple[float, float]],
    curves: list[np.ndarray],
    bars: list[np.ndarray],
) -> None:
    """Place each label at the candidate offset (LABEL_OFFSETS_PT) of least cost.

    Hard obstacles (cost 100 per overlap): markers (all plotted points), curves (polylines
    through the means), earlier labels and their leader lines (both ways: the new text on an
    old leader, the new leader through an old text), leaving the axes. Soft obstacles (cost 1
    per overlap): CI bars. Ties go to the shortest leader line. Call after the final layout;
    curves and bars are (N, 2) data-coordinate polylines.
    """
    fig.canvas.draw()  # settle the layout so data -> pixel transforms are final
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    r = 6 * fig.dpi / 72  # marker half-size + margin, pixels
    hard = [matplotlib.transforms.Bbox([[x - r, y - r], [x + r, y + r]]) for x, y in to_px(markers)]
    hard += _sample_boxes([to_px(c) for c in curves], 2.0)
    soft = _sample_boxes([to_px(b) for b in bars], 1.0)
    if ax.get_legend() is not None:
        hard.append(ax.get_legend().get_window_extent(renderer))
    axes_box = ax.get_window_extent(renderer)

    def annotate(xy, text, offset):
        dx = offset[0]
        ha = "left" if dx > 1 else "right" if dx < -1 else "center"
        return ax.annotate(
            text,
            xy,
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7,
            color=MUTED,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkA": 1, "shrinkB": 3},
        )

    placed_text: list = []  # boxes of placed labels
    pt = fig.dpi / 72  # pixels per point

    def leader(xy_px, offset) -> list:
        # sample the leader line, skipping the marker end (it starts inside the marker)
        end = xy_px + pt * np.asarray(offset)
        return _sample_boxes([np.array([xy_px + 0.35 * (end - xy_px), end])], 1.0)

    def cost(box, lead, offset) -> float:
        inside = axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1
        inside = inside and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1
        n_hard = sum(box.overlaps(b) for b in hard) + (0 if inside else 1)
        n_hard += sum(any(seg.overlaps(b) for b in placed_text) for seg in lead) > 0
        n_soft = sum(box.overlaps(b) for b in soft)
        return 100.0 * n_hard + n_soft + 0.02 * math.hypot(*offset)

    for xy, text in labels:
        xy_px = to_px([xy])[0]
        best = None
        for offset in LABEL_OFFSETS_PT:
            t = annotate(xy, text, offset)
            # the text alone (an Annotation's own extent also covers its leader line); the
            # Annotation must place its text first, or the box sits at the bare offset
            t.update_positions(renderer)
            box = matplotlib.text.Text.get_window_extent(t, renderer)
            lead = leader(xy_px, offset)
            c = cost(box, lead, offset)
            t.remove()
            if best is None or c < best[0]:
                best = (c, offset, box, lead)
        annotate(xy, text, best[1])
        hard += [best[2], *best[3]]
        placed_text.append(best[2])


def plot_tradeoff(data: dict, family: str, bcfg, footer: str, out_path: str | Path) -> Path:
    """Paired extra length (%) vs paired reduction, per alpha, Baseline-Occ at the origin."""
    key = TRADEOFF_Y[family]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)
    ax.axvline(0.0, color=GRID, lw=1.0, zorder=1)
    labels: list[tuple[tuple[float, float], str]] = []
    markers: list[tuple[float, float]] = [(0.0, 0.0)]
    curves: list[np.ndarray] = []
    bars: list[np.ndarray] = []
    for planner, curve in data.items():
        alphas = list(curve)
        xs = np.array([curve[a][0] for a in alphas])  # rows (mean, lo, hi)
        ys = np.array([curve[a][1] for a in alphas])
        colour = PLANNER_COLOURS[planner]
        ax.errorbar(
            xs[:, 0],
            ys[:, 0],
            xerr=np.array([xs[:, 0] - xs[:, 1], xs[:, 2] - xs[:, 0]]),
            yerr=np.array([ys[:, 0] - ys[:, 1], ys[:, 2] - ys[:, 0]]),
            color=colour,
            marker=PLANNER_MARKERS[planner],
            ms=5,
            lw=1.5,
            elinewidth=0.8,
            capsize=2,
            # translucent CIs keep the curves readable; Oracle (the reference) lighter still
            ecolor=colour + ("59" if planner == "oracle" else "80"),
            label=PLANNER_LABELS[planner],
            zorder=3,
        )
        markers += list(zip(xs[:, 0], ys[:, 0], strict=True))
        curves.append(np.column_stack([xs[:, 0], ys[:, 0]]))
        for (mx, lo_x, hi_x), (my, lo_y, hi_y) in zip(xs, ys, strict=True):
            bars += [np.array([[lo_x, my], [hi_x, my]]), np.array([[mx, lo_y], [mx, hi_y]])]
        if planner != "terranav":
            continue
        # alpha labels on TerraNav only; alphas giving the same paths share one point and label
        groups: dict[tuple[float, float], list[float]] = {}
        for a, x, y in zip(alphas, xs[:, 0], ys[:, 0], strict=True):
            groups.setdefault((round(x, 6), round(y, 9)), []).append(a)
        labels += [((x, y), "α=" + ",".join(f"{a:g}" for a in g)) for (x, y), g in groups.items()]
        if bcfg.default_alpha in curve:
            x, y = curve[bcfg.default_alpha][0][0], curve[bcfg.default_alpha][1][0]
            ax.plot(x, y, "o", ms=15, mfc="none", mec=INK, mew=1.5, zorder=4)
    ax.plot(
        0.0,
        0.0,
        PLANNER_MARKERS["baseline_occ"],
        ms=8,
        color=PLANNER_COLOURS["baseline_occ"],
        label="Baseline-Occ (reference)",
        zorder=5,
    )
    ax.plot(
        [],
        [],
        "o",
        ms=12,
        mfc="none",
        mec=INK,
        mew=1.5,
        ls="none",
        label=f"TerraNav at pre-registered α = {bcfg.default_alpha:g}",
    )
    ax.set_xlabel("extra path length vs Baseline-Occ [%]", color=INK)
    ax.set_ylabel(TRADEOFF_Y_LABEL[key], color=INK)
    ax.set_title(f"{family}: terrain cost vs path length over alpha", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(
        0.5,
        0.01,
        f"{footer}; 95% paired bootstrap CIs.\n{ORACLE_NOTE}",
        ha="center",
        fontsize=7.5,
        color=MUTED,
        linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    _place_labels(fig, ax, labels, markers, curves, bars)  # after the final layout
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=DPI, metadata=PNG_METADATA)
    plt.close(fig)
    return out_path


# ---- consistency with the committed summary.md -------------------------------------------

_ROW = re.compile(
    r"^\| (\w+) \| (\w+) - (\w+) \| (\d+) \| ([^|]+?) \| ([+-]\d+\.\d+) \| "
    r"\[([+-]\d+\.\d+), ([+-]\d+\.\d+)\] \|"
)


def summary_paired_table(summary_text: str) -> dict[tuple[str, str, str, str], tuple[str, ...]]:
    """{(family, A, B, metric label): (mean, lo, hi) strings} from summary.md section (b)."""
    out = {}
    for line in summary_text.splitlines():
        m = _ROW.match(line)
        if m:
            fam, a, b, _, label, mu, lo, hi = m.groups()
            out[(fam, a, b, label)] = (mu, lo, hi)
    return out


def check_headline_against_summary(data: dict, summary_text: str) -> int:
    """Every plotted (mean, lo, hi), converted back to the summary's unit and printed as
    summary.md prints it, must equal the summary's
    string, and lie within half a unit of its last digit (+1e-9). Returns the number checked."""
    table = summary_paired_table(summary_text)
    label_of = dict(PAIRED)
    n = 0
    for (key, other, fam), stat in data.items():
        expected = table[(fam, "terranav", other, label_of[key])]
        for v_plot, s in zip(stat, expected, strict=True):
            v = v_plot / HEADLINE_SCALE[key]  # back to the summary's unit
            if f"{v:+.4f}" != s or not math.isclose(v, float(s), abs_tol=5e-5 + 1e-9):
                raise AssertionError(f"{fam} terranav - {other} {key}: plotted {v!r}, summary {s}")
            n += 1
    return n


def make_figures(run_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """headline.png and tradeoff_<family>.png from run_dir's raw.jsonl into out_dir, after
    checking every headline value against run_dir/summary.md."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    records, bcfg, _ = load_run(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = headline_data(records, bcfg)
    check_headline_against_summary(data, (run_dir / "summary.md").read_text())
    paths = [plot_headline(data, bcfg, sample_note(records, bcfg), out_dir / "headline.png")]
    footer = caption(records, bcfg)
    for fam in TRADEOFF_Y:
        curve = tradeoff_data(records, bcfg, fam)
        paths.append(plot_tradeoff(curve, fam, bcfg, footer, out_dir / f"tradeoff_{fam}.png"))
    return paths
