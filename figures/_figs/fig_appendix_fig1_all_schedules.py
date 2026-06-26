"""Appendix: baselines across all schedules vs PACE — 3x3 grid.

Rows = models (SmolLM2-1.7B, Qwen3-1.7B, Gemma3-1B); columns = learning-rate
schedule (Constant LR | Cosine | WSD). Per (model, schedule) cell the AdamW and
EMA baselines use that column's schedule, and the PACE constant-learning-rate run
is repeated in every column as the reference. Per model the headline learning
rate is fixed and the best config is picked at each (method, schedule) cell.
Y-limits are shared across each row so schedules compare directly.

Data: ``<data>/fig1_all_schedules/selection.json`` + the per-run JSONs it points
to (best (model, method, schedule) at the headline LR; bundled).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

MODELS = ["SmolLM2-1.7B", "Qwen3-1.7B", "Gemma3-1B"]
MODEL_STEPS = {"SmolLM2-1.7B": 1778, "Qwen3-1.7B": 1779, "Gemma3-1B": 1779}
SCHEDS = ["const", "cos", "wsd"]
SCHED_TITLE = {"const": "Constant LR", "cos": "Cosine", "wsd": "WSD"}


def _curve(jpath, prefer_ema: bool):
    o = json.load(open(jpath))
    logs = o["eval_logs"]
    steps = np.array([e.get("step", 0) or 0 for e in logs], dtype=float)
    key = "ema_val_loss" if prefer_ema else "vanilla_val_loss"
    raw = [e.get(key) for e in logs]
    if all(v is None for v in raw):
        raw = [e.get("vanilla_val_loss") or e.get("val_loss") for e in logs]
    y = np.array([np.nan if v is None else float(v) for v in raw], dtype=float)
    m = ~np.isnan(y) & (steps >= 0)
    return steps[m], y[m]


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig1_all_schedules"
    sel = json.load(open(data_dir / "selection.json"))
    models = cfg.get("models") or MODELS

    def _path(model, meth, sched):
        e = sel.get(f"{model}|{meth}|{sched}")
        return (data_dir / e["file"]) if e else None

    fig, axes = pu.make_panels(3, 3)
    report = {}
    for row, model in enumerate(models):
        row_finals = []
        pace = _path(model, "PACE", "const")
        for col, sched in enumerate(SCHEDS):
            ax = axes[row][col]
            for meth in ("AdamW", "EMA"):
                p = _path(model, meth, sched)
                if p is None:
                    continue
                xs, ys = _curve(p, prefer_ema=(meth != "AdamW"))
                if not len(xs):
                    continue
                st = pu.METHOD_STYLE[meth]
                ax.plot(xs, ys, color=st["color"], linestyle=st["linestyle"],
                        marker="o", markevery=max(1, len(xs) // 7),
                        markeredgecolor="white", markeredgewidth=0.5)
                ax.scatter([xs[-1]], [ys[-1]], color=st["color"],
                           s=pu.ENDPOINT_MARKER_SIZE - 20, edgecolor="black",
                           linewidth=1.0, zorder=6)
                row_finals.append(float(ys[-1]))
                report[f"{model}|{meth}|{sched}"] = round(float(ys[-1]), 4)
            if pace is not None:
                xs, ys = _curve(pace, prefer_ema=True)
                st = pu.METHOD_STYLE["PACE"]
                ax.plot(xs, ys, color=st["color"], linestyle=st["linestyle"],
                        marker="o", markevery=max(1, len(xs) // 7),
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)
                ax.scatter([xs[-1]], [ys[-1]], color=st["color"],
                           s=pu.ENDPOINT_MARKER_SIZE - 20, edgecolor="black",
                           linewidth=1.0, zorder=6)
                row_finals.append(float(ys[-1]))
            if row == 0:
                ax.set_title(SCHED_TITLE[sched], pad=10)
            ax.set_xlabel("Step" if row == 2 else "")
            ax.set_ylabel(f"{model}\nVal Loss" if col == 0 else "")
            pu.flush_xaxis(ax, x_min=0, x_max=MODEL_STEPS[model])
        if pace is not None:
            _, pys = _curve(pace, prefer_ema=True)
            report[f"{model}|PACE|const"] = round(float(pys[-1]), 4)
        lo = min(row_finals) - 0.03
        hi = 0.90 if model.startswith("SmolLM2") else max(row_finals) + 0.15
        for col in range(3):
            axes[row][col].set_ylim(lo, hi)

    handles = [Line2D([0], [0], color=pu.METHOD_STYLE[m]["color"],
                      linestyle=pu.METHOD_STYLE[m]["linestyle"], linewidth=3.5)
               for m in ("AdamW", "EMA", "PACE")]
    axes[0][0].legend(handles, ["AdamW", "EMA", "PACE (constant)"],
                      loc="upper right", framealpha=0.92)

    fig.align_ylabels([axes[r][0] for r in range(3)])
    out = pu.savefig_pdf(fig, "fig_appendix_fig1_all_schedules")
    return {"pdf": str(out), "final_val_loss": report}
