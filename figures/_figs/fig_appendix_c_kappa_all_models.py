"""Appendix: combined c x EMA-power ablation for Gemma3, SmolLM2, and Qwen3.

A 3x3 grid: rows = models (Gemma3-1B, SmolLM2-1.7B, Qwen3-1.7B), columns = EMA
power kappa in {0.2, 0.5, 0.7}. Each panel plots, per learning rate, the best
PACE validation loss as a function of pullback strength c (with the matched-lr
EMA baseline at c=0 and the AdamW baseline as a square at the left), so the
pullback's effect over c is visible at each (model, kappa).

Data: ``<data>/c_kappa_all_models/rows.json`` — one record per run with
(method, final, lr, c, kappa) for all three models (bundled, summary-only).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

KAPPAS = [0.2, 0.5, 0.7]
C_GRID = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
MODELS = ["Gemma3-1B", "SmolLM2-1.7B", "Qwen3-1.7B"]


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if s.startswith(("1.0e", "2.0e", "3.0e", "5.0e", "7.0e")):
        s = s[0] + s[3:]
    return s


def _x_pos(c: float) -> int:
    if c == 0:
        return 0
    return min(range(len(C_GRID)), key=lambda i: abs(np.log10(c) - np.log10(C_GRID[i]))) + 1


def _eligible_lrs(title: str, rows) -> list:
    all_lrs = sorted({r["lr"] for r in rows if r["method"] == "PACE" and r["lr"] is not None})
    counts = {
        lr: len({r["c"] for r in rows if r["method"] == "PACE" and abs(r["lr"] - lr) < 1e-12})
        for lr in all_lrs
    }
    if title == "Gemma3-1B":
        return [lr for lr in all_lrs if counts[lr] >= 4]

    eligible = []
    for lr in all_lrs:
        if counts[lr] < 3:
            continue
        improved = False
        for kappa in KAPPAS:
            ours = [r for r in rows if r["method"] == "PACE" and r["kappa"] is not None
                    and abs(r["kappa"] - kappa) < 1e-9 and abs(r["lr"] - lr) < 1e-12]
            ema = [r for r in rows if r["method"] == "EMA" and r["kappa"] is not None
                   and abs(r["kappa"] - kappa) < 1e-9 and abs(r["lr"] - lr) < 1e-12]
            if ours and ema and (min(r["final"] for r in ema) - min(r["final"] for r in ours)) > 0.005:
                improved = True
                break
        if improved:
            eligible.append(lr)
    if len(eligible) > 4:
        eligible = sorted({eligible[0], eligible[len(eligible) // 3],
                           eligible[2 * len(eligible) // 3], eligible[-1]})
    if title == "SmolLM2-1.7B":
        eligible = [lr for lr in eligible if abs(lr - 3e-3) > 1e-9]
    return eligible


def _draw_panel(ax, rows, kappa, lrs):
    pull = [r for r in rows if r["method"] == "PACE" and r["kappa"] is not None
            and abs(r["kappa"] - kappa) < 1e-9]
    ema = [r for r in rows if r["method"] == "EMA" and r["kappa"] is not None
           and abs(r["kappa"] - kappa) < 1e-9]
    adamw = [r for r in rows if r["method"] == "AdamW"]

    for lr in lrs:
        color = pu.lr_color(lr, lrs)
        sub = [r for r in pull if abs(r["lr"] - lr) < 1e-12]
        if not sub:
            continue
        best_at = {}
        for r in sub:
            pos = _x_pos(r["c"])
            if pos not in best_at or r["final"] < best_at[pos]:
                best_at[pos] = r["final"]
        em = [r for r in ema if abs(r["lr"] - lr) < 1e-12]
        if em:
            best_at[0] = min(r["final"] for r in em)
        xs = sorted(best_at)
        ys = [best_at[x] for x in xs]
        ax.plot(xs, ys, color=color, marker="o", markeredgecolor="white",
                markeredgewidth=0.6)

        aw = [r for r in adamw if abs(r["lr"] - lr) < 1e-12]
        if aw:
            ax.scatter([-1], [min(r["final"] for r in aw)], color=color, marker="s",
                       s=100, edgecolor="black", linewidth=1.0, zorder=5)

    xticks = [-1, 0] + list(range(1, len(C_GRID) + 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels(["AdamW", "EMA"] + [f"{v:.0e}".replace("e-0", "e-") for v in C_GRID],
                       rotation=45, ha="right")
    pu.flush_xaxis(ax, x_min=-1.3, x_max=len(C_GRID) + 0.3)


def build(cfg) -> dict:
    pu.set_plot_style()
    data = json.load(open(pu.DATA_ROOT / "c_kappa_all_models" / "rows.json"))
    models = cfg.get("models") or MODELS

    fig, axes = pu.make_panels(3, 3, sharex=True, extra_left_in=0.9)
    report = {}
    for row_idx, title in enumerate(models):
        rows = data[title]
        lrs = _eligible_lrs(title, rows)
        report[f"{title}|n_lrs"] = len(lrs)
        row_vals = []
        for col_idx, kappa in enumerate(KAPPAS):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(f"EMA Power = {kappa:g}", pad=8)
            _draw_panel(ax, rows, kappa, lrs)
            ax.set_ylabel(f"{title}\nVal Loss" if col_idx == 0 else "")
            ax.set_xlabel("Pullback strength $c$" if row_idx == len(models) - 1 else "")

            for r in rows:
                if r["lr"] not in lrs:
                    continue
                if r["method"] == "AdamW" or (r["kappa"] is not None and abs(r["kappa"] - kappa) < 1e-9):
                    row_vals.append(r["final"])
        if row_vals:
            lo = min(row_vals) - 0.01
            hi = max(row_vals) * 1.02 + 0.01
            for ax in axes[row_idx]:
                ax.set_ylim(lo, hi)
        for ax in axes[row_idx][1:]:
            ax.tick_params(labelleft=False)

        leg_h, leg_l = [], []
        for lr in lrs[:4]:
            leg_h.append(Line2D([0], [0], color=pu.lr_color(lr, lrs), marker="o",
                                linestyle="-", linewidth=3, markersize=9))
            leg_l.append(f"lr = {_fmt_lr(lr)}")
        pu.place_inside_legend(axes[row_idx][-1], leg_h, leg_l, loc="best")

    out = pu.savefig_pdf(fig, "fig_appendix_c_kappa_all_models")
    return {"pdf": str(out), "lrs_per_model": report}
