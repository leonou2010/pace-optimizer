"""Appendix multiseed robustness: all three Fig-1 models, both schedules.

2 rows x 3 cols: rows = Constant LR / WSD, columns = SmolLM2-1.7B, Qwen3-1.7B,
Gemma3-1B. Per panel: the mean validation curve plus a min-max band across seeds
{42, 1337, 2024} for each method (AdamW, EMA, PACE) at the model's headline cell.

Metric: AdamW ``vanilla_val_loss``; EMA/PACE ``ema_val_loss`` — same as Fig 1.

Data: ``<data>/multiseed/multiseed_files_3models.json`` and the per-seed JSONs it
references (paths are relative to that manifest; all bundled).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

MODELS = ["SmolLM2-1.7B", "Qwen3-1.7B", "Gemma3-1B"]
METHODS = ["AdamW", "EMA", "PACE"]
SEEDS = [42, 1337, 2024]
SCHEDULES = [("const", "Constant LR"), ("wsd", "WSD")]
TOTAL = {"SmolLM2-1.7B": 1778, "Qwen3-1.7B": 1779, "Gemma3-1B": 1779}


def _curve(jpath, prefer_ema: bool):
    o = json.load(open(jpath))
    logs = o.get("eval_logs", [])
    xs = np.array([e.get("step", 0) or 0 for e in logs], dtype=float)
    key = "ema_val_loss" if prefer_ema else "vanilla_val_loss"
    raw = [e.get(key) for e in logs]
    if all(v is None for v in raw):
        raw = [e.get("vanilla_val_loss") or e.get("val_loss") for e in logs]
    ys = np.array([np.nan if v is None else float(v) for v in raw], dtype=float)
    m = ~np.isnan(ys) & (xs > 0)
    return xs[m], ys[m]


def _band(files, data_dir, model, method, sched, grid):
    mats = []
    for s in SEEDS:
        v = files[model].get(f"{method}|{sched}|{s}")
        if v is None:
            continue
        xs, ys = _curve(data_dir / v[2], prefer_ema=(method != "AdamW"))
        if len(xs) < 3:
            continue
        mats.append(np.interp(grid, xs, ys))
    if not mats:
        return None
    M = np.array(mats)
    return M.mean(0), M.min(0), M.max(0), len(mats)


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "multiseed"
    files = json.load(open(data_dir / "multiseed_files_3models.json"))
    models = cfg.get("models") or MODELS

    fig, axes = pu.make_panels(2, 3)
    report = {}
    for col, model in enumerate(models):
        total = TOTAL[model]
        grid = np.append(np.arange(100, total - 70, 100), total)
        for row, (sched, sched_title) in enumerate(SCHEDULES):
            ax = axes[row][col]
            finals = []
            for meth in METHODS:
                res = _band(files, data_dir, model, meth, sched, grid)
                if res is None:
                    continue
                mean, lo, hi, n = res
                st = pu.METHOD_STYLE[meth]
                ax.fill_between(grid, lo, hi, color=st["color"], alpha=0.25,
                                linewidth=0, zorder=3)
                ax.plot(grid, mean, color=st["color"], marker=st["marker"],
                        linestyle=st["linestyle"],
                        markevery=max(1, len(grid) // 8),
                        markeredgecolor="white", markeredgewidth=0.6,
                        zorder=5 if meth == "PACE" else 4)
                finals.append(float(mean[-1]))
                report[f"{model}|{sched}|{meth}"] = round(float(mean[-1]), 4)
            ax.set_title(f"{model} ({sched_title})", pad=10)
            ax.set_xlabel("Step" if row == 1 else "")
            ax.set_ylabel("Val Loss" if col == 0 else "")
            pu.flush_xaxis(ax, grid[0], grid[-1])
            if finals:
                ax.set_ylim(min(finals) - 0.03, max(finals) + 0.28)

    handles = [Line2D([0], [0], color=pu.METHOD_STYLE[m]["color"],
                      marker=pu.METHOD_STYLE[m]["marker"],
                      linestyle=pu.METHOD_STYLE[m]["linestyle"],
                      linewidth=3.5, markersize=12) for m in METHODS]
    pu.place_inside_legend(axes[0][0], handles, METHODS, loc="upper right")

    fig.align_ylabels([axes[r][0] for r in range(2)])
    out = pu.savefig_pdf(fig, "fig_appendix_multiseed_robust")
    return {"pdf": str(out), "mean_final_val_loss": report}
