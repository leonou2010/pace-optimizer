"""Appendix: GPT-2-124M pretraining EMA-Power sweep.

Final validation loss versus pullback strength ``c``, one panel per EMA power
kappa in {0.2, 0.5, 0.7}, with an lr-gradient family of lines. AdamW is shown as
a per-lr coloured square at the leftmost x-tick; EMA is the ``c = 0`` tick.

Data: ``<data>/fig_appendix_pretrain_kappa_sweep/summary.csv`` (bundled compact
summary of the GPT-2 pretraining sweep, restricted to the PACE / AdamW / EMA
methods). Columns: method, lr, c, kappa, final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_utils as pu

KAPPAS = [0.2, 0.5, 0.7]


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if s.startswith(("1.0e", "2.0e", "3.0e", "5.0e")):
        s = s[0] + s[3:]
    return s


def _fmt_c(c: float) -> str:
    if c == 0:
        return "EMA"
    return f"{c:.0e}".replace("e-0", "e-")


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig_appendix_pretrain_kappa_sweep"
    df = pd.read_csv(data_dir / "summary.csv")
    df = df[df["final"] < 4.5]

    adamw_df = df[df["method"] == "AdamW"]
    ema_df = df[df["method"] == "EMA"]
    ours_df = df[df["method"] == "PACE"]

    fig, axes = pu.make_panels(1, len(KAPPAS), sharey=False)
    axes_row = axes[0]

    report = {}
    all_seen_lrs = []
    for col, k in enumerate(KAPPAS):
        ax = axes_row[col]
        ax.set_title(f"EMA Power = {k:g}", pad=10)
        ours_k = ours_df[np.isclose(ours_df["kappa"].fillna(-1), k)]
        ema_k = ema_df[np.isclose(ema_df["kappa"].fillna(-1), k)]

        keep_lrs = sorted([lr for lr in ours_k["lr"].dropna().unique()
                           if ours_k[np.isclose(ours_k["lr"], lr)]["c"].nunique() >= 2])
        all_seen_lrs.append(keep_lrs)

        c_values = sorted(ours_k["c"].unique())
        x_labels = ["AdamW", "EMA"] + [_fmt_c(c) for c in c_values]
        x_pos = {-1: -1, 0: 0}
        for i, c in enumerate(c_values):
            x_pos[c] = i + 1

        for lr in keep_lrs:
            color = pu.lr_color(lr, keep_lrs)
            sub = ours_k[np.isclose(ours_k["lr"], lr)]
            best_at_c = sub.groupby("c")["final"].min().sort_index()
            xs, ys = [], []
            em = ema_k[np.isclose(ema_k["lr"], lr)]
            if not em.empty:
                xs.append(0)
                ys.append(float(em["final"].min()))
            for c_val, v in best_at_c.items():
                pos = x_pos.get(c_val)
                if pos is None:
                    continue
                xs.append(pos)
                ys.append(float(v))
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            if xs:
                ax.plot(xs, ys, marker="o", color=color, markeredgecolor="white",
                        markeredgewidth=0.6)
                report[f"k{k}|lr{_fmt_lr(lr)}|min"] = round(min(ys), 4)
            aw = adamw_df[np.isclose(adamw_df["lr"], lr)]
            if not aw.empty:
                ax.scatter([-1], [float(aw["final"].min())],
                           color=color, marker="s", s=100,
                           edgecolor="black", linewidth=1.0, zorder=5)

        xticks = [-1, 0] + list(range(1, len(c_values) + 1))
        ax.set_xticks(xticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        pu.flush_xaxis(ax, x_min=-1.3, x_max=len(c_values) + 0.3)
        ax.set_xlabel("Pullback strength $c$")
        ax.set_ylabel("Val Loss" if col == 0 else "")

        sub_pull = ours_k[ours_k["lr"].isin(keep_lrs)]["final"].dropna()
        sub_ema = ema_k[ema_k["lr"].isin(keep_lrs)]["final"].dropna()
        sub_aw = adamw_df[adamw_df["lr"].isin(keep_lrs)]["final"].dropna()
        if not sub_pull.empty:
            lo = float(sub_pull.min()) - 0.01
            hi = float(max(sub_ema.max() if not sub_ema.empty else lo,
                           sub_aw.max() if not sub_aw.empty else lo)) * 1.02
            ax.set_ylim(lo, hi)

    last_lrs = all_seen_lrs[-1] if all_seen_lrs else []
    leg_h = [Line2D([0], [0], color="0.15", marker="s", linestyle="None",
                    markersize=12, markeredgecolor="black")]
    leg_l = ["AdamW"]
    for lr in last_lrs[:5]:
        leg_h.append(Line2D([0], [0], color=pu.lr_color(lr, last_lrs),
                            marker="o", linestyle="-", linewidth=3, markersize=9))
        leg_l.append(f"lr = {_fmt_lr(lr)}")
    pu.place_inside_legend(axes_row[-1], leg_h, leg_l, loc="lower right")

    out = pu.savefig_pdf(fig, "fig_appendix_pretrain_kappa_sweep")
    return {"pdf": str(out), "n_runs": int(len(df)), "pace_minima": report}
