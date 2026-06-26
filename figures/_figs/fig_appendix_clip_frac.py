"""Appendix — fraction of parameters with saturated lambda over training.

Shows the proportion of coordinates whose per-parameter pullback strength
lambda is clipped to 1, over training, for SmolLM2-1.7B. Layout: 2 rows
(learning rate in {3e-4, 1e-3}) x 3 cols (EMA power kappa in {0.2, 0.5, 0.7}).
Lines per panel are coloured on a c-gradient (light = small c, dark = large c).

Data: cached telemetry at ``<data>/clip_frac/_cache_clip_frac.json`` (keys are
``"<lr>|<kappa>|<c>"`` -> {step, frac}).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

LRS = [3e-4, 1e-3]
KAPPAS = [0.2, 0.5, 0.7]
C_VALUES = [0.005, 0.015, 0.05, 0.15, 0.5]


def _load() -> dict:
    cache = pu.DATA_ROOT / "clip_frac" / "_cache_clip_frac.json"
    return json.loads(cache.read_text())


def _smooth(y, win=51):
    if len(y) < win:
        return y
    kk = np.ones(win) / win
    out = np.convolve(y, kk, mode="same")
    edge = win // 2
    out[:edge] = y[:edge]
    out[-edge:] = y[-edge:]
    return out


def _fmt_lr(lr: float) -> str:
    if lr == 1e-3:
        return r"10^{-3}"
    if lr == 3e-4:
        return r"3\!\times\!10^{-4}"
    if lr == 5e-4:
        return r"5\!\times\!10^{-4}"
    return f"{lr:g}"


def build(cfg) -> dict:
    data = _load()
    pu.set_plot_style()
    lrs = list(cfg.get("lrs") or LRS)
    kappas = list(cfg.get("kappas") or KAPPAS)
    cs = sorted(cfg.get("c_values") or C_VALUES)

    nrows, ncols = len(lrs), len(kappas)
    fig, axes = pu.make_panels(nrows, ncols)
    report = {}

    for i, lr in enumerate(lrs):
        for j, kappa in enumerate(kappas):
            ax = axes[i, j]
            for c in cs:
                cell = data.get(f"{lr}|{kappa}|{c}")
                if not cell:
                    continue
                steps = np.array(cell["step"], dtype=float)
                frac = np.array(cell["frac"], dtype=float)
                order = np.argsort(steps)
                steps, frac = steps[order], frac[order]
                frac = _smooth(frac, win=51)
                t = (np.log10(c) - np.log10(cs[0])) / max(
                    1e-9, np.log10(cs[-1]) - np.log10(cs[0]))
                color = pu.LR_CMAP(0.30 + 0.60 * t)
                ax.plot(steps, frac, color=color, marker="o",
                        markevery=max(1, len(steps) // 8),
                        markeredgecolor="white", markeredgewidth=0.6)
                report[f"lr={lr}|k={kappa}|c={c}"] = round(float(frac[-1]), 4)
            ax.set_ylim(0.0, 1.0)
            pu.flush_xaxis(ax, x_min=0)
            if i == 0:
                ax.set_title(f"EMA Power = {kappa:g}", pad=8)
            if j == 0:
                ax.set_ylabel(rf"Learning rate = ${_fmt_lr(lr)}$" + "\n"
                              + r"Fraction $\lambda$ clipped at 1")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Step" if i == nrows - 1 else "")

    leg_h, leg_l = [], []
    for c in cs:
        t = (np.log10(c) - np.log10(cs[0])) / max(
            1e-9, np.log10(cs[-1]) - np.log10(cs[0]))
        leg_h.append(Line2D([0], [0], color=pu.LR_CMAP(0.30 + 0.60 * t),
                            marker="o", linestyle="-", linewidth=3, markersize=9))
        leg_l.append(f"$c = {c:g}$")
    pu.place_inside_legend(axes[0, -1], leg_h, leg_l, fontsize=11, loc="upper right")

    out = pu.savefig_pdf(fig, "fig_appendix_clip_frac")
    return {"pdf": str(out), "final_clip_frac": report}
