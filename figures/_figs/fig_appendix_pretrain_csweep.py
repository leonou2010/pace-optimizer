"""Appendix: GPT-2-124M pretraining pullback-strength sweep across schedules.

3-seed (seeds {0,1,42}) version: returned-model validation loss versus pullback
strength ``c``, one line per learning-rate schedule (Constant | Cosine | WSD), on
the strictly-fair shared grid at eta = 2e-3. Each plotted point is the best over
the shared kappa grid; the two leftmost ticks show the AdamW run and the
best-kappa EMA (c = 0). Error bars are the population std over the three seeds.

Schedule is encoded by colour and method by marker shape (PACE circle, AdamW
square, EMA diamond).

Data: ``<data>/fig_appendix_pretrain_csweep/summary_master2_3seed.csv``.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D

import plot_utils as pu

SCHED_COLOR = {
    "constant": "#E69F00",   # orange
    "cosine":   "#17BECF",   # teal
    "WSD":      "#C71585",   # magenta
}
SCHEDULES = [
    ("Constant_noWSD", "constant"),
    ("Cosine",         "cosine"),
    ("WSD",            "WSD"),
]
C_GRID = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
C_LABELS = ["3e-4", "1e-3", "2e-3", "3e-3", "5e-3", "7e-3", "1e-2"]
XTICKS = ["AdamW", "EMA"] + C_LABELS


def _best_over_k(rows):
    if rows.empty:
        return None
    r = rows.loc[rows["mean"].idxmin()]
    return float(r["mean"]), float(r["std"])


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig_appendix_pretrain_csweep"
    df = pd.read_csv(data_dir / "summary_master2_3seed.csv")

    fig, axes = pu.make_wide_panel()
    ax = axes[0, 0]

    report = {}
    all_ys = []
    for sched_csv, label in SCHEDULES:
        col = SCHED_COLOR[label]
        g = df[df.schedule == sched_csv]
        aw = _best_over_k(g[g.method == "adamw"])
        em = _best_over_k(g[g.method == "ema"])
        pace = [_best_over_k(g[(g.method == "pace") & (g.c == c)]) for c in C_GRID]

        xs = list(range(1, len(pace) + 2))
        ys = [em[0]] + [p[0] for p in pace]
        es = [em[1]] + [p[1] for p in pace]
        ax.errorbar(xs, ys, yerr=es, color=col, linestyle="-", marker="o",
                    markeredgecolor="white", markeredgewidth=0.6,
                    capsize=2.5, elinewidth=1.0,
                    zorder=5 if sched_csv == "WSD" else 4)
        ax.errorbar([0], [aw[0]], yerr=[aw[1]], color=col, marker="s",
                    markersize=11, markeredgecolor="black", capsize=2.5,
                    elinewidth=1.0, zorder=6, linestyle="None")
        ax.errorbar([1], [em[0]], yerr=[em[1]], color=col, marker="D",
                    markersize=10, markeredgecolor="black", capsize=2.5,
                    elinewidth=1.0, zorder=6, linestyle="None")
        all_ys += [aw[0], em[0]] + [p[0] for p in pace]
        cmin = min(range(len(pace)), key=lambda i: pace[i][0])
        report[label] = {"adamw": round(aw[0], 4), "ema": round(em[0], 4),
                         "pace_min": round(pace[cmin][0], 4),
                         "pace_min_c": C_LABELS[cmin]}

    ax.set_xticks(range(len(XTICKS)))
    ax.set_xticklabels(XTICKS, rotation=45, ha="right")
    pu.flush_xaxis(ax, x_min=-0.4, x_max=len(XTICKS) - 0.6)
    ax.set_ylim(min(all_ys) - 0.02, max(all_ys) + 0.02)
    ax.set_xlabel(r"Pullback strength $c$")
    ax.set_ylabel("Val Loss")
    ax.set_title("GPT-2-124M pretraining", pad=10)

    grey = "0.35"
    sched_handles = [Line2D([0], [0], color=SCHED_COLOR[label], linestyle="-",
                            linewidth=3.5)
                     for _, label in SCHEDULES]
    sched_labels = [label for _, label in SCHEDULES]
    method_handles = [
        Line2D([0], [0], color=grey, linestyle="-", marker="o", linewidth=3,
               markersize=9, markeredgecolor="white"),
        Line2D([0], [0], color=grey, linestyle="None", marker="s",
               markersize=11, markeredgecolor="black"),
        Line2D([0], [0], color=grey, linestyle="None", marker="D",
               markersize=10, markeredgecolor="black"),
    ]
    method_labels = ["PACE", "AdamW", "EMA"]
    pu.place_inside_legend(ax, sched_handles + method_handles,
                           sched_labels + method_labels, loc="upper right", ncol=2)

    out = pu.savefig_pdf(fig, "fig_appendix_pretrain_csweep")
    return {"pdf": str(out), "envelope": report}
