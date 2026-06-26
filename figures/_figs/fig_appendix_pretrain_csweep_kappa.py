"""Appendix: GPT-2-124M pretraining per-kappa pullback sweep, one panel per
schedule.

1 row x 3 cols (Constant | Cosine | WSD). Each panel: returned-model validation
loss versus pullback strength ``c``, one line per EMA power kappa in
{0.1, 0.2, 0.3, 0.5} (Reds gradient, light = small kappa), on the strictly-fair
shared grid per schedule at eta = 2e-3, seed 42. The leftmost ticks show the
AdamW run (black square, kappa-independent) and each kappa's EMA (c = 0).

Metric convention: AdamW on vanilla_val_loss; EMA and PACE on ema_val_loss.

Data: ``<data>/fig_appendix_pretrain_csweep_kappa/summary.csv`` (the two fair
experiments from the post_neurips_pretrain bundle).
"""
from __future__ import annotations

import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D

import plot_utils as pu

SCHEDULES = [("Constant LR", "Constant"), ("Cosine", "Cosine"), ("WSD", "WSD")]
KAPPAS = [0.1, 0.2, 0.3, 0.5]
KAPPA_COLOR = {k: mpl.cm.Reds(x) for k, x in
               zip(KAPPAS, (0.35, 0.55, 0.73, 0.92))}
C_GRID = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
C_LABELS = ["3e-4", "1e-3", "2e-3", "3e-3", "5e-3", "7e-3", "1e-2"]
XTICKS = ["AdamW", "EMA"] + C_LABELS


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig_appendix_pretrain_csweep_kappa"
    fair = pd.read_csv(data_dir / "summary.csv")
    fair["ret"] = fair.apply(
        lambda r: r.vanilla_val_loss if r.method == "adamw" else r.ema_val_loss,
        axis=1)

    fig, axes = pu.make_panels(1, 3)
    report = {}
    all_ys = []
    for col, (sched, title) in enumerate(SCHEDULES):
        ax = axes[0, col]
        g = fair[fair.schedule == sched]
        aw = float(g[g.method == "adamw"].ret.min())
        ax.scatter([0], [aw], color="black", marker="s", s=130,
                   edgecolor="black", linewidth=1.2, zorder=6)
        all_ys.append(aw)
        report[f"{title}|AdamW"] = round(aw, 4)
        for kappa in KAPPAS:
            em = g[(g.method == "ema") & (g.k == kappa)]
            ys = [float(em.ret.min())] if len(em) else []
            xs = [1] if len(em) else []
            for i, c in enumerate(C_GRID):
                cell = g[(g.method == "pace") & (g.k == kappa) & (g.c == c)]
                if len(cell):
                    xs.append(2 + i)
                    ys.append(float(cell.ret.min()))
            ax.plot(xs, ys, color=KAPPA_COLOR[kappa], linestyle="-",
                    marker="o", markeredgecolor="white", markeredgewidth=0.6)
            all_ys += ys
            if ys:
                report[f"{title}|k{kappa}|min"] = round(min(ys), 4)
        ax.set_title(title, pad=10)
        ax.set_xticks(range(len(XTICKS)))
        ax.set_xticklabels(XTICKS, rotation=45, ha="right")
        pu.flush_xaxis(ax, x_min=-0.4, x_max=len(XTICKS) - 0.6)
        ax.set_xlabel(r"Pullback strength $c$")
        ax.set_ylabel("Val Loss" if col == 0 else "")

    y_lo, y_hi = min(all_ys) - 0.012, max(all_ys) + 0.012
    for col in range(3):
        axes[0, col].set_ylim(y_lo, y_hi)

    handles = [Line2D([0], [0], color="black", marker="s", linestyle="None",
                      markersize=11, markeredgecolor="black")]
    labels = ["AdamW"]
    for kappa in KAPPAS:
        handles.append(Line2D([0], [0], color=KAPPA_COLOR[kappa], marker="o",
                              linestyle="-", linewidth=3, markersize=9,
                              markeredgecolor="white"))
        labels.append(rf"$\kappa = {kappa}$")
    pu.place_inside_legend(axes[0, 2], handles, labels, loc="upper left")

    fig.align_xlabels(axes[0])
    fig.align_ylabels(axes[0])
    out = pu.savefig_pdf(fig, "fig_appendix_pretrain_csweep_kappa")
    return {"pdf": str(out), "minima": report}
