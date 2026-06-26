"""Appendix — full 3x3 SmolLM2-135M grid (update frequency x EMA power).

Rows = update frequency {1, 5, 10}; columns = EMA power kappa {0.3, 0.5, 0.7}.
Column super-headers on the top row only, row labels in the leftmost ylabel only.
Per panel: one lr-gradient line over the c-grid (light = small lr), with EMA at
its x-tick and AdamW as an lr-coloured square at the leftmost "AdamW" x-tick.

Data: SmolLM2-135M exp13 chunked results
(``<data>/fig_appendix_135m_full_grid/*_chunk*.json``).
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

C_VALUES = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
KAPPAS = [0.3, 0.5, 0.7]
UFS = [1, 5, 10]

X_LABELS = ["AdamW", "EMA"] + [
    f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+") for v in C_VALUES
]
X_POS = {"AdamW": 0, "EMA": 1}
for _i, _c in enumerate(C_VALUES):
    X_POS[_c] = _i + 2


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if s.startswith(("1.0e", "2.0e", "3.0e", "5.0e")):
        s = s[0] + s[3:]
    return s


def _c_to_x(c):
    for ref, pos in X_POS.items():
        if isinstance(ref, (int, float)) and isinstance(c, (int, float)):
            if abs(float(ref) - float(c)) < 1e-12:
                return pos
    return None


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig_appendix_135m_full_grid"
    df = pu.load_chunks(data_dir)
    df = df.drop_duplicates(subset=["run_id"], keep="first")
    if df.empty:
        raise SystemExit("no exp13 data found")

    kappas = list(cfg.get("kappas") or KAPPAS)
    ufs = list(cfg.get("ufs") or UFS)

    adamw_df = df[df["method"] == "AdamW"]
    ema_df = df[(df["method"] == "EMA") | ((df["c"] == 0) & (df["method"] != "AdamW"))]
    ours_df = df[(df["method"] == "PACE") & (df["c"] != 0)]

    eligible_lrs = sorted(ours_df["lr"].dropna().unique())
    if eligible_lrs and len(eligible_lrs) > 1:
        eligible_lrs = eligible_lrs[:-1]  # drop highest (often divergent)

    fig, axes = pu.make_panels(len(ufs), len(kappas),
                               sharex=True, sharey="row", extra_left_in=0.9)

    seen_lrs = set()
    report = {"kappas": kappas, "ufs": ufs, "cells": {}}
    for row, uf in enumerate(ufs):
        for col, k in enumerate(kappas):
            ax = axes[row, col]
            ax.set_title(f"EMA Power = {k:g}" if row == 0 else "", pad=10)

            for lr in eligible_lrs:
                color = pu.lr_color(lr, eligible_lrs)
                sub = ours_df[
                    np.isclose(ours_df["lr"].astype(float).fillna(-1), lr)
                    & np.isclose(ours_df["kappa"].astype(float).fillna(-1), k)
                    & np.isclose(ours_df["uf"].astype(float).fillna(-1), uf)
                ]
                if sub.empty:
                    continue
                best_at_c = sub.groupby("c")["final_val_loss"].min()
                ema_sub = ema_df[
                    np.isclose(ema_df["lr"].astype(float).fillna(-1), lr)
                    & np.isclose(ema_df["kappa"].astype(float).fillna(-1), k)
                    & np.isclose(ema_df["uf"].astype(float).fillna(-1), uf)
                ]
                ema_best = float(ema_sub["final_val_loss"].min()) if not ema_sub.empty else np.nan
                adamw_best = adamw_df[np.isclose(
                    adamw_df["lr"].astype(float).fillna(-1), lr)]["final_val_loss"].min()

                xs, ys = [], []
                if not np.isnan(ema_best):
                    xs.append(X_POS["EMA"]); ys.append(ema_best)
                for c, v in best_at_c.items():
                    pos = _c_to_x(c)
                    if pos is None:
                        continue
                    xs.append(pos); ys.append(float(v))
                order = np.argsort(xs)
                xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
                if xs:
                    ax.plot(xs, ys, marker="o", color=color, markeredgecolor="white",
                            markeredgewidth=0.6)
                if not np.isnan(adamw_best):
                    ax.scatter([X_POS["AdamW"]], [float(adamw_best)], color=color,
                               marker="s", s=100, edgecolor="black", linewidth=1.0, zorder=5)
                seen_lrs.add(lr)
                if not best_at_c.empty:
                    report["cells"][f"uf={uf}|k={k}|lr={_fmt_lr(lr)}"] = round(
                        float(best_at_c.min()), 4)

            if col == 0:
                ax.set_ylabel(f"Frequency = {uf}\n\nVal Loss")
            else:
                ax.set_ylabel("")
            if row == len(ufs) - 1:
                ax.set_xticks(list(range(len(X_LABELS))))
                ax.set_xticklabels(X_LABELS, rotation=45, ha="right")
                ax.set_xlabel("Pullback strength $c$")
            else:
                ax.set_xlabel("")
            pu.flush_xaxis(ax, x_min=-0.3, x_max=len(X_LABELS) - 0.7)

    # Per-row y-zoom (shared across kappa columns via sharey='row').
    for row, uf in enumerate(ufs):
        sub = ours_df[np.isclose(ours_df["uf"].astype(float).fillna(-1), uf)]["final_val_loss"].dropna()
        if sub.empty:
            continue
        y_lo = float(sub.quantile(0.02)) - 0.003
        y_hi = float(sub.quantile(0.75)) + 0.01
        for col in range(len(kappas)):
            axes[row, col].set_ylim(y_lo, y_hi)

    for row in range(len(ufs)):
        for col in range(1, len(kappas)):
            axes[row, col].tick_params(labelleft=False)

    handles, labels = [], []
    for lr in sorted(seen_lrs):
        handles.append(Line2D([0], [0], color=pu.lr_color(lr, sorted(seen_lrs)),
                              marker="o", linestyle="-", linewidth=3, markersize=9))
        labels.append(f"lr = {_fmt_lr(lr)}")
    pu.place_inside_legend(axes[0, -1], handles, labels, fontsize=11)

    out = pu.savefig_pdf(fig, "fig_appendix_135m_full_grid")
    report["pdf"] = str(out)
    report["n_lrs"] = len(seen_lrs)
    return report
