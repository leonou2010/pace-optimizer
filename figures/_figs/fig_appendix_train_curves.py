"""Appendix: train-loss trajectories dropped from the body method-comparison figure.

1 row x 2 cols: SmolLM2-135M | Qwen3-1.7B. Same best-of-method picks as the body
figure, plotting smoothed ``train_loss`` from ``eval_logs`` for AdamW, EMA, and
PACE at each panel's matched learning rate.

Data: ``<data>/train_curves/manifest.json`` + the per-run JSONs it references
(trimmed to step + train_loss; bundled).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

METHODS = ["AdamW", "EMA", "PACE"]
LEGEND_SPLIT = {"SmolLM2-135M": ["AdamW", "EMA"], "Qwen3-1.7B": ["PACE"]}


def _train_curve(eval_logs, max_step=None):
    if not eval_logs:
        return np.array([]), np.array([])
    steps = np.array([e.get("step", i) for i, e in enumerate(eval_logs)], dtype=float)
    y = np.array([e.get("train_loss") if e.get("train_loss") is not None else np.nan
                  for e in eval_logs], dtype=float)
    mask = ~np.isnan(y)
    if max_step is not None:
        mask &= steps <= max_step
    return steps[mask], y[mask]


def _smooth(y, win=11):
    if len(y) < win:
        return y
    k = np.ones(win) / win
    out = np.convolve(y, k, mode="same")
    edge = win // 2
    out[:edge] = y[:edge]
    out[-edge:] = y[-edge:]
    return out


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "train_curves"
    manifest = json.load(open(data_dir / "manifest.json"))
    panels = list(manifest.keys())

    fig, axes = pu.make_panels(1, len(panels))
    report = {}
    for j, title in enumerate(panels):
        ax = axes[0][j]
        cells = manifest[title]["cells"]
        loaded = {}
        for m in METHODS:
            fn = cells.get(m)
            if not fn:
                continue
            loaded[m] = json.load(open(data_dir / fn)).get("eval_logs", [])
        if not loaded:
            ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center",
                    va="center", color="#888888")
            ax.set_title(title)
            continue
        common = min(int(logs[-1].get("step", 0)) for logs in loaded.values() if logs)
        finals = []
        for m in METHODS:
            if m not in loaded:
                continue
            sx, sy = _train_curve(loaded[m], max_step=common)
            if not len(sx):
                continue
            sy_s = _smooth(sy, win=11)
            st = pu.METHOD_STYLE[m]
            ax.plot(sx, sy_s, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], markevery=max(1, len(sx) // 8),
                    markeredgecolor="white", markeredgewidth=0.6)
            finals.append(float(sy_s[-1]))
            report[f"{title}|{m}"] = round(float(sy_s[-1]), 4)

        ax.set_title(title, pad=10)
        pu.flush_xaxis(ax, x_min=0, x_max=common)
        if finals:
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(cur_lo, min(cur_hi, max(finals) * 1.15 + 0.1))

        leg_methods = LEGEND_SPLIT.get(title, METHODS)
        leg_h = [Line2D([0], [0], color=pu.METHOD_STYLE[m]["color"],
                        marker=pu.METHOD_STYLE[m]["marker"],
                        linestyle=pu.METHOD_STYLE[m]["linestyle"],
                        linewidth=3, markersize=9) for m in leg_methods]
        pu.place_inside_legend(ax, leg_h, leg_methods)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss" if j == 0 else "")

    out = pu.savefig_pdf(fig, "fig_appendix_train_curves")
    return {"pdf": str(out), "final_train_loss": report}
