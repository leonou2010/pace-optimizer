"""Appendix: WSD fine-tuning validation trajectories on SmolLM2-135M.

1x3 layout, one panel per learning rate in {2e-3, 3e-3, 5e-3}. Each panel shows
validation-loss trajectories for AdamW+WSD, EMA+WSD, and PACE (constant LR).

Data: ``<data>/wsd_finetune/manifest.json`` + the per-run JSONs it references
(bundled). AdamW+WSD reads ``vanilla_val_loss``; EMA+WSD and PACE read
``val_loss``.
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

LRS_PANEL = [2e-3, 3e-3, 5e-3]
_METHODS = ["AdamW_best", "EMA_best", "PACE"]
_LABELS = {"AdamW_best": "AdamW", "EMA_best": "EMA", "PACE": "PACE"}
_VKEY = {"AdamW_best": "vanilla_val_loss", "EMA_best": "val_loss", "PACE": "val_loss"}


def _style(method):
    base = {"AdamW_best": "AdamW", "EMA_best": "EMA", "PACE": "PACE"}[method]
    color = pu.METHOD_STYLE[base]["color"]
    linestyle = "--" if method == "PACE" else "-"
    return dict(color=color, linestyle=linestyle)


def _curve(jpath, key):
    o = json.load(open(jpath))
    logs = o.get("eval_logs") or []
    steps = [e["step"] for e in logs if e.get(key) is not None]
    vals = [e[key] for e in logs if e.get(key) is not None]
    return steps, vals


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "wsd_finetune"
    manifest = json.load(open(data_dir / "manifest.json"))
    cells = manifest["cells"]

    fig, axes = pu.make_panels(1, len(LRS_PANEL))
    report = {}
    common_max_step = 0
    for j, lr in enumerate(LRS_PANEL):
        ax = axes[0][j]
        cell = cells.get(repr(lr), {})
        for method in _METHODS:
            fn = cell.get(method)
            if not fn:
                continue
            steps, vals = _curve(data_dir / fn, _VKEY[method])
            if not steps:
                continue
            steps = np.asarray(steps, dtype=float)
            vals = np.asarray(vals, dtype=float)
            mask = vals < 5.0
            steps, vals = steps[mask], vals[mask]
            if not len(steps):
                continue
            st = _style(method)
            ax.plot(steps, vals, color=st["color"], linestyle=st["linestyle"],
                    marker="o", markevery=max(1, len(steps) // 8),
                    markeredgecolor="white", markeredgewidth=0.6)
            common_max_step = max(common_max_step, int(steps[-1]))
            report[f"{lr:g}|{_LABELS[method]}"] = round(float(vals[-1]), 4)
        ax.set_title(f"Learning rate = {lr:.0e}", pad=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Val Loss" if j == 0 else "")

    if common_max_step > 0:
        for j in range(len(LRS_PANEL)):
            axes[0][j].set_xlim(0, common_max_step)

    leg = [Line2D([0], [0], color=_style(m)["color"], linestyle=_style(m)["linestyle"],
                  linewidth=4, label=_LABELS[m]) for m in _METHODS]
    pu.place_inside_legend(axes[0][-1], leg, [h.get_label() for h in leg],
                           loc="upper right")

    out = pu.savefig_pdf(fig, "fig_appendix_wsd_val_curves")
    return {"pdf": str(out), "final_val_loss": report}
