"""Appendix: best validation loss across learning rates for the WSD fine-tuning
ablation on SmolLM2-135M x smol-smoltalk.

Single wide panel. Sweeps eta in {1, 1.5, 2, 3, 5} x 1e-3 and plots final
validation cross-entropy for AdamW+WSD, EMA+WSD, and PACE (constant LR).

Data: ``<data>/wsd_finetune/manifest.json`` + the per-run JSONs it references
(bundled). Final loss = ``summary.best_val_loss`` (falling back to
``final_val_loss``).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

_METHODS = ["AdamW_best", "EMA_best", "PACE"]
_LABELS = {"AdamW_best": "AdamW", "EMA_best": "EMA", "PACE": "PACE"}


def _style(method):
    base = {"AdamW_best": "AdamW", "EMA_best": "EMA", "PACE": "PACE"}[method]
    color = pu.METHOD_STYLE[base]["color"]
    linestyle = "--" if method == "PACE" else "-"
    return dict(color=color, linestyle=linestyle)


def _best_val(jpath):
    o = json.load(open(jpath))
    s = o.get("summary") or {}
    v = s.get("best_val_loss") or s.get("final_val_loss")
    return float(v) if v is not None else None


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "wsd_finetune"
    manifest = json.load(open(data_dir / "manifest.json"))
    lrs = manifest["LRS"]
    cells = manifest["cells"]

    fig, axes = pu.make_wide_panel()
    ax = axes[0][0]

    x = np.arange(len(lrs))
    report = {}
    for method in _METHODS:
        ys = []
        for lr in lrs:
            fn = cells.get(repr(lr), {}).get(method)
            ys.append(_best_val(data_dir / fn) if fn else None)
        st = _style(method)
        valid = [(xi, yi) for xi, yi in zip(x, ys) if yi is not None]
        if not valid:
            continue
        xv = np.array([p[0] for p in valid])
        yv = np.array([p[1] for p in valid])
        ax.plot(xv, yv, color=st["color"], linestyle=st["linestyle"], marker="o",
                markeredgecolor="black", markeredgewidth=1.0, markersize=10)
        report[_LABELS[method]] = [round(v, 4) if v is not None else None for v in ys]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax.set_title("SmolLM2-135M", pad=8)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Val Loss")

    leg = [Line2D([0], [0], color=_style(m)["color"], linestyle=_style(m)["linestyle"],
                  linewidth=4, label=_LABELS[m]) for m in _METHODS]
    pu.place_inside_legend(ax, leg, [h.get_label() for h in leg], loc="upper left")

    out = pu.savefig_pdf(fig, "fig_appendix_wsd_finetune")
    return {"pdf": str(out), "final_val_loss_by_lr": report, "lrs": lrs}
