"""Appendix: GPT-2-124M pretraining token-budget comparison.

AdamW WSD decay branches at token budgets {50%, 75%, 100%} overlaid on the PACE
constant-learning-rate trajectory; x-axis in percent of the full token budget.
The branch endpoints land on (and above) the PACE line, illustrating that PACE's
constant-LR run matches the loss of WSD-decayed AdamW at every budget without a
schedule.

Metric convention: AdamW on vanilla_val_loss; PACE on ema_val_loss.

Data: ``<data>/fig_appendix_pretrain_river/raw_curves/...`` (the four referenced
curve JSONs from the post_neurips_pretrain bundle).
"""
from __future__ import annotations

import json

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

FULL = 4750
BUDGETS = [2375, 3563, 4750]
PCT = {2375: 50, 3563: 75, 4750: 100}
BRANCHES = {
    2375: "pretrain_gpt2_fineweb_wsd_sweep/wsdbudget_adamw_b2375_p50.json",
    3563: "pretrain_gpt2_fineweb_wsd_sweep/wsdbudget_adamw_b3563_p75.json",
    4750: "pretrain_gpt2_fineweb_wsd_sweep/wsd_adamw_lr0.002.json",
}
PACE_CONST = "pretrain_gpt2_fineweb_sched_kc/const_pace_c0.002_k0.5_lr0.002.json"
BUDGET_SHADE = {2375: 0.45, 3563: 0.65, 4750: 0.88}


def build(cfg) -> dict:
    pu.set_plot_style()
    raw = pu.DATA_ROOT / "fig_appendix_pretrain_river" / "raw_curves"

    def _curve(rel: str, key: str):
        d = json.loads((raw / rel).read_text())
        pts = [(e["step"], e.get(key)) for e in d["eval_loss"] if e.get(key) is not None]
        return (np.array([p[0] for p in pts], dtype=float),
                np.array([p[1] for p in pts], dtype=float))

    fig, axes = pu.make_wide_panel()
    ax = axes[0, 0]

    endpoints = {}
    for b in BUDGETS:
        s, v = _curve(BRANCHES[b], "vanilla_val_loss")
        col = mpl.cm.Blues(BUDGET_SHADE[b])
        ax.plot(100 * s / FULL, v, color=col, linestyle="-", marker="o",
                markevery=max(1, len(s) // 6), markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
        ax.scatter([100 * s[-1] / FULL], [v[-1]], color=col,
                   s=pu.ENDPOINT_MARKER_SIZE, edgecolor="black",
                   linewidth=1.2, zorder=6)
        endpoints[b] = float(v[-1])

    ps, pv = _curve(PACE_CONST, "ema_val_loss")
    pace_red = pu.METHOD_STYLE["PACE"]["color"]
    ax.plot(100 * ps / FULL, pv, color=pace_red, linestyle="-",
            marker="o", markevery=max(1, len(ps) // 8),
            markeredgecolor="white", markeredgewidth=0.6, zorder=7)
    ax.scatter([100 * ps[-1] / FULL], [pv[-1]], color=pace_red,
               s=pu.ENDPOINT_MARKER_SIZE + 60, edgecolor="black",
               linewidth=1.5, zorder=8)

    report = {}
    for b in BUDGETS:
        pace_at = float(np.interp(min(b - 1, ps[-1]), ps, pv))
        report[f"{PCT[b]}%"] = {"adamw_wsd_endpoint": round(endpoints[b], 4),
                                "pace_const": round(pace_at, 4),
                                "gap": round(endpoints[b] - pace_at, 4)}

    ax.set_ylim(min(pv.min(), min(endpoints.values())) - 0.03, 4.3)
    pu.flush_xaxis(ax, x_min=0, x_max=100)
    ax.set_xlabel("Token budget (%)")
    ax.set_ylabel("Val Loss")
    ax.set_title("GPT-2-124M pretraining", pad=10)

    handles = [Line2D([0], [0], color=pace_red, linestyle="-", marker="o",
                      linewidth=3.5, markersize=11, markeredgecolor="white")] + \
              [Line2D([0], [0], color=mpl.cm.Blues(BUDGET_SHADE[b]),
                      linestyle="-", marker="o", linewidth=3.0, markersize=9,
                      markeredgecolor="white") for b in BUDGETS]
    labels = ["PACE (constant)"] + [f"AdamW WSD {PCT[b]}%" for b in BUDGETS]
    pu.place_inside_legend(ax, handles, labels, loc="upper right")

    out = pu.savefig_pdf(fig, "fig_appendix_pretrain_river")
    return {"pdf": str(out), "budgets": report}
