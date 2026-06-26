"""Appendix figure: Qwen3-1.7B on Tulu-3.

Sparse validation-loss sweep from exp27. The top row shows final validation loss
by method (AdamW / EMA / PACE), one line per learning rate; the bottom row shows
the corresponding validation trajectories at lr = 3e-4. One column per EMA power
kappa in {0.2, 0.5, 0.7}.

Metric convention: AdamW on vanilla_val_loss; EMA and PACE on ema_val_loss.

Data: ``<data>/fig_appendix_tulu_qwen/*.json`` (14 trimmed exp27 run JSONs).
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

KAPPAS = [0.2, 0.5, 0.7]
STEP0_VAL_LOSS = 1.6095420797366928
TRAJ_LR = 3e-4


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if s.startswith(("3.0e", "5.0e")):
        s = s[0] + s[3:]
    return s


def _load_rows(results):
    rows = []
    for path in sorted(results.glob("*.json")):
        data = json.load(open(path))
        summary = data.get("summary") or {}
        opt = data.get("config", {}).get("optimizer", {})
        final = summary.get("final_val_loss")
        if final is None:
            continue
        c = float(opt.get("lambda_pullback") or 0.0)
        if c > 0:
            method = "PACE"
        elif opt.get("use_ema_eval"):
            method = "EMA"
        else:
            method = "AdamW"
        rows.append({
            "method": method,
            "lr": float(opt["lr"]),
            "kappa": float(opt.get("ema_kappa", 0.0)),
            "c": c,
            "final": float(final),
            "eval_logs": data.get("eval_logs") or [],
        })
    return rows


def _best(rows, method, lr, kappa=None):
    cands = [r for r in rows if r["method"] == method
             and abs(r["lr"] - lr) < 1e-12
             and (kappa is None or abs(r["kappa"] - kappa) < 1e-12)]
    return min(cands, key=lambda r: r["final"]) if cands else None


def _curve(row):
    steps, vals = [0.0], [STEP0_VAL_LOSS]
    for i, event in enumerate(row.get("eval_logs") or []):
        step = event.get("step", i)
        if row["method"] == "AdamW":
            val = event.get("vanilla_val_loss", event.get("val_loss"))
        else:
            val = event.get("ema_val_loss", event.get("val_loss"))
        if step is None or val is None:
            continue
        if float(step) < 500:
            continue
        steps.append(float(step))
        vals.append(float(val))
    return np.asarray(steps), np.asarray(vals)


def build(cfg) -> dict:
    pu.set_plot_style()
    results = pu.DATA_ROOT / "fig_appendix_tulu_qwen"
    rows = _load_rows(results)
    lrs = sorted({r["lr"] for r in rows})

    fig, axes = pu.make_panels(2, len(KAPPAS), sharey="row")
    x = np.array([-1, 0, 1])
    xlabels = ["AdamW", "EMA", "PACE"]
    report = {}
    for col, kappa in enumerate(KAPPAS):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        ax_top.set_title(f"EMA Power = {kappa:g}", pad=10)
        for lr in lrs:
            adamw = _best(rows, "AdamW", lr)
            ema = _best(rows, "EMA", lr, kappa)
            pace = _best(rows, "PACE", lr, kappa)
            vals = [
                adamw["final"] if adamw else np.nan,
                ema["final"] if ema else np.nan,
                pace["final"] if pace else np.nan,
            ]
            ax_top.plot(x, vals, color=pu.lr_color(lr, lrs), marker="o",
                        markeredgecolor="black", markeredgewidth=1.0)
            if pace is not None:
                report[f"k{kappa}|lr{_fmt_lr(lr)}|PACE"] = round(pace["final"], 4)

            if abs(lr - TRAJ_LR) > 1e-12:
                continue
            for run in [adamw, ema, pace]:
                if run is None:
                    continue
                steps, curve = _curve(run)
                if len(steps) == 0:
                    continue
                style = pu.METHOD_STYLE[run["method"]]
                ax_bottom.plot(steps, curve, color=style["color"],
                               linestyle=style["linestyle"], marker=style["marker"],
                               markeredgecolor="white", markeredgewidth=0.6,
                               markevery=max(1, len(steps) // 8))

        ax_top.set_xticks(x)
        ax_top.set_xticklabels(xlabels)
        ax_top.set_xlabel("Method" if col == 1 else "")
        ax_top.set_ylabel("Val Loss" if col == 0 else "")
        pu.flush_xaxis(ax_top, x_min=-1.3, x_max=1.3)

        ax_bottom.set_xlabel("Step" if col == 1 else "")
        ax_bottom.set_ylabel("Val Loss" if col == 0 else "")
        pu.flush_xaxis(ax_bottom)

    leg_h = [Line2D([0], [0], color=pu.lr_color(lr, lrs), marker="o",
                    linestyle="-", linewidth=3, markersize=9) for lr in lrs]
    leg_l = [f"lr = {_fmt_lr(lr)}" for lr in lrs]
    pu.place_inside_legend(axes[0, -1], leg_h, leg_l, loc="upper right")

    curve_h = [Line2D([0], [0], color=pu.METHOD_STYLE[m]["color"],
                      marker=pu.METHOD_STYLE[m]["marker"],
                      linestyle=pu.METHOD_STYLE[m]["linestyle"], linewidth=3,
                      markersize=9) for m in ["AdamW", "EMA", "PACE"]]
    curve_l = ["AdamW", "EMA", "PACE"]
    pu.place_inside_legend(axes[1, -1], curve_h, curve_l, loc="upper right")

    out = pu.savefig_pdf(fig, "fig_appendix_tulu_qwen")
    return {"pdf": str(out), "n_runs": len(rows), "pace_final": report}
