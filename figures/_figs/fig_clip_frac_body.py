"""Body — fraction of saturated lambda over training (top row of the appendix
clip-fraction figure, learning rate 3e-4 only).

1x3 panel grid at eta=3e-4 with kappa in {0.2, 0.5, 0.7}, giving direct evidence
for the per-coordinate variance-scaling claim. Lines per panel are coloured on a
c-gradient (light = small c, dark = large c).

Data: cached telemetry at ``<data>/clip_frac/_cache_clip_frac.json``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu
from _figs.fig_appendix_clip_frac import _load, _smooth, C_VALUES, KAPPAS

LR = 3e-4


def build(cfg) -> dict:
    data = _load()
    pu.set_plot_style()
    lr = float(cfg.get("lr", LR))
    kappas = list(cfg.get("kappas") or KAPPAS)
    cs = sorted(cfg.get("c_values") or C_VALUES)

    fig, axes = pu.make_panels(1, len(kappas))
    report = {}
    for j, kappa in enumerate(kappas):
        ax = axes[0, j]
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
            report[f"k={kappa}|c={c}"] = round(float(frac[-1]), 4)
        ax.set_ylim(0.0, 1.0)
        pu.flush_xaxis(ax, x_min=0)
        ax.set_title(f"EMA Power = {kappa:g}", pad=8)
        ax.set_xlabel("Step")
        ax.set_ylabel(r"Fraction $\lambda$ clipped at 1" if j == 0 else "")

    leg_h, leg_l = [], []
    for c in cs:
        t = (np.log10(c) - np.log10(cs[0])) / max(
            1e-9, np.log10(cs[-1]) - np.log10(cs[0]))
        leg_h.append(Line2D([0], [0], color=pu.LR_CMAP(0.30 + 0.60 * t),
                            marker="o", linestyle="-", linewidth=3, markersize=9))
        leg_l.append(f"$c = {c:g}$")
    pu.place_inside_legend(axes[0, -1], leg_h, leg_l, fontsize=11, loc="upper right")

    out = pu.savefig_pdf(fig, "fig_clip_frac_body")
    return {"pdf": str(out), "lr": lr, "final_clip_frac": report}
