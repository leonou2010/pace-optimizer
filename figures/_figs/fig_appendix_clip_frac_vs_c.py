"""Appendix — end-of-training lambda-clip fraction vs. pullback strength c.

Companion to the clip-fraction time-series figure: fixes the step (end of
training, mean over the last ~100 logged steps) and sweeps c on a log x-axis,
one line per learning rate, one panel per EMA power kappa, on SmolLM2-1.7B.

Data: cached telemetry at ``<data>/clip_frac/_cache_clip_frac.json``.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

KAPPAS = [0.2, 0.5, 0.7]


def _load():
    cache = pu.DATA_ROOT / "clip_frac" / "_cache_clip_frac.json"
    return json.loads(cache.read_text())


def build(cfg) -> dict:
    data = _load()
    pu.set_plot_style()
    kappas = list(cfg.get("kappas") or KAPPAS)

    by_lr_k = defaultdict(lambda: defaultdict(dict))  # [lr][k][c] = mean frac
    for key, cell in data.items():
        try:
            lr_s, k_s, c_s = key.split("|")
            lr = float(lr_s); k = float(k_s); c = float(c_s)
        except Exception:
            continue
        steps = cell.get("step", [])
        fracs = cell.get("frac", [])
        if not steps:
            continue
        last_step = max(steps)
        tail_idx = [i for i, s in enumerate(steps) if s >= last_step - 100]
        if not tail_idx:
            continue
        by_lr_k[lr][k][c] = float(np.mean([fracs[i] for i in tail_idx]))

    LRS = sorted(by_lr_k.keys())
    fig, axes = pu.make_panels(1, len(kappas))

    def _lr_alpha(lr):
        return 0.30 + 0.60 * (LRS.index(lr) / max(1, len(LRS) - 1))

    report = {}
    for col, kappa in enumerate(kappas):
        ax = axes[0, col]
        for lr in LRS:
            cells = by_lr_k[lr].get(kappa, {})
            if not cells:
                continue
            cs = sorted(cells.keys())
            ys = [cells[c] for c in cs]
            color = pu.LR_CMAP(_lr_alpha(lr))
            ax.plot(cs, ys, color=color, marker="o", linestyle="-",
                    markeredgecolor="black", markeredgewidth=1.0,
                    markersize=10, linewidth=4)
            for c, y in zip(cs, ys):
                report[f"k={kappa}|lr={lr}|c={c}"] = round(float(y), 4)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.05)
        all_cs_set = sorted({c for lr in LRS for c in by_lr_k[lr].get(kappa, {}).keys()})
        if all_cs_set:
            ax.set_xticks(all_cs_set)
            ax.set_xticklabels([f"{c:g}" for c in all_cs_set])
        ax.set_title(f"EMA Power = {kappa:g}", pad=8)
        ax.set_xlabel(r"Pullback strength $c$")
        ax.set_ylabel(r"Fraction $\lambda$ clipped at 1" if col == 0 else "")

    leg_h, leg_l = [], []
    for lr in LRS:
        leg_h.append(Line2D([0], [0], color=pu.LR_CMAP(_lr_alpha(lr)),
                            marker="o", linestyle="-", linewidth=4, markersize=10,
                            markeredgecolor="black", markeredgewidth=1.0))
        leg_l.append(f"Learning rate = {lr:.0e}")
    pu.place_inside_legend(axes[0, -1], leg_h, leg_l, fontsize=11, loc="upper right")

    out = pu.savefig_pdf(fig, "fig_appendix_clip_frac_vs_c")
    return {"pdf": str(out), "lrs": LRS, "kappas": kappas, "end_clip_frac": report}
