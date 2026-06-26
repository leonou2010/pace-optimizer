"""Fig (pretraining) — GPT-2 small (124M) on FineWeb.

1 row x 3 cols:
  Left   -- best-config validation curves per method (AdamW / EMA / PACE),
            CONSTANT LR, from the bundled raw curves.
  Middle -- c sweep at kappa=0.5: x = c grid with leftmost ticks AdamW + EMA;
            one lr-gradient line per kept lr.
  Right  -- best-config validation curves per method, WSD schedule, zoomed.

This repository reproduces the PACE / AdamW / EMA methods only.

Data:
  ``<data>/fig5_pretraining/raw_curves/`` -- bundled best-config curves.
  ``<data>/fig5_pretraining/exp25``, ``.../exp26`` -- nanoGPT c/lr/kappa sweep
  (pruned: config.optimizer + summary + eval_logs).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_utils as pu

METHODS = ["AdamW", "EMA", "PACE"]


def _method_from_name(name: str) -> str:
    n = name.lower()
    if "_adamw_" in n:
        return "AdamW"
    if "_ema_" in n:
        return "EMA"
    if "_pb_" in n:
        return "PACE"
    return "unknown"


def _load(root) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for p in sorted(root.glob("*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        s = d.get("summary") or {}
        fv = s.get("final_val_loss")
        if fv is None:
            continue
        method = _method_from_name(p.name)
        if method == "unknown":
            continue
        eval_logs = d.get("eval_logs") or []
        opt = (d.get("config") or {}).get("optimizer", {})
        if not isinstance(opt, dict):
            opt = {}
        rows.append({
            "path": p, "method": method,
            "lr": float(opt.get("lr")) if opt.get("lr") is not None else None,
            "c": float(opt.get("lambda_pullback", 0) or 0),
            "kappa": float(opt.get("ema_kappa")) if opt.get("ema_kappa") is not None else None,
            "final": float(fv),
            "eval_logs": eval_logs,
        })
    return rows


def _curve(eval_logs: list, method: str):
    if not eval_logs:
        return np.array([]), np.array([])
    steps = np.array([e.get("step", i) for i, e in enumerate(eval_logs)], dtype=float)
    if method in ("EMA", "PACE"):
        y = np.array([e.get("ema_val_loss") if e.get("ema_val_loss") is not None
                      else e.get("vanilla_val_loss") for e in eval_logs], dtype=float)
    else:
        y = np.array([e.get("vanilla_val_loss") if e.get("vanilla_val_loss") is not None
                      else e.get("ema_val_loss") for e in eval_logs], dtype=float)
    mask = ~np.isnan(y)
    return steps[mask], y[mask]


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if s.startswith(("1.0e", "2.0e", "3.0e", "5.0e")):
        s = s[0] + s[3:]
    return s


BUNDLE_PANELS = {
    "Constant LR": [
        ("AdamW", "pretrain_gpt2_fineweb_sched_kc/const_adamw_lr0.002.json"),
        ("EMA", "pretrain_gpt2_fineweb_sched_kc/const_ema_k0.5_lr0.002.json"),
        ("PACE", "pretrain_gpt2_fineweb_sched_kc/const_pace_c0.002_k0.5_lr0.002.json"),
    ],
    "WSD": [
        ("AdamW", "pretrain_gpt2_fineweb_wsd_kc/wsd_adamw_lr0.002.json"),
        ("EMA", "pretrain_gpt2_fineweb_wsd_kc/wsd_ema_k0.2_lr0.002.json"),
        ("PACE", "pretrain_gpt2_fineweb_wsd_kc/wsd_pace_c0.002_k0.2_lr0.002.json"),
    ],
}


def _bundle_curve(bundle_root, rel: str, method: str):
    d = json.load(open(bundle_root / rel))
    key = "vanilla_val_loss" if method == "AdamW" else "ema_val_loss"
    pts = [(e["step"], e.get(key)) for e in d["eval_loss"] if e.get(key) is not None]
    return (np.array([p[0] for p in pts], dtype=float),
            np.array([p[1] for p in pts], dtype=float))


def draw_bundle_panel(ax, bundle_root, sched: str, ymax: float, xmin: float = 0):
    finals = {}
    for method, rel in BUNDLE_PANELS[sched]:
        s_, v = _bundle_curve(bundle_root, rel, method)
        st = pu.METHOD_STYLE[method]
        ax.plot(s_, v, color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markevery=max(1, len(s_) // 8),
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=5 if method == "PACE" else 4)
        ax.scatter([s_[-1]], [v[-1]], color=st["color"], s=pu.ENDPOINT_MARKER_SIZE,
                   edgecolor="black", linewidth=1.5, zorder=6)
        finals[method] = float(v[-1])
    ax.set_title(sched, pad=10)
    pu.flush_xaxis(ax, x_min=xmin, x_max=4750)
    ax.set_ylim(min(finals.values()) - 0.03, ymax)
    return finals


C_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.03]
C_LABELS = ["EMA"] + ["1e-3", "2e-3", "3e-3", "5e-3", "7e-3", "1e-2", "3e-2"]


def _nearest_c_pos(c: float):
    for i, cg in enumerate(C_GRID):
        if cg == 0.0:
            continue
        if abs(c - cg) / cg < 0.05:
            return i
    return None


def draw_c_sweep(ax, rows, plotted_lrs, outlier_thresh: float = 4.5):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "eval_logs"} for r in rows])
    df = df[df["final"] < outlier_thresh]
    ours_k05 = df[(df["method"] == "PACE") & np.isclose(df["kappa"].astype(float).fillna(-1), 0.5)]
    ema_k05 = df[(df["method"] == "EMA") & np.isclose(df["kappa"].astype(float).fillna(-1), 0.5)]
    adamw = df[df["method"] == "AdamW"]

    eligible_lrs = [lr for lr, _ in plotted_lrs]
    swept = {}
    for lr in eligible_lrs:
        color = pu.lr_color(lr, eligible_lrs)
        sub = ours_k05[np.isclose(ours_k05["lr"], lr)]
        pos_to_best = {}
        for c_val, grp in sub.groupby("c"):
            pos = _nearest_c_pos(float(c_val))
            if pos is None:
                continue
            v = float(grp["final"].min())
            if pos not in pos_to_best or v < pos_to_best[pos]:
                pos_to_best[pos] = v
        em = ema_k05[np.isclose(ema_k05["lr"], lr)]
        xs, ys = [], []
        if not em.empty:
            xs.append(0); ys.append(float(em["final"].min()))
        for pos in sorted(pos_to_best):
            xs.append(pos); ys.append(pos_to_best[pos])
        if xs:
            ax.plot(xs, ys, marker="o", color=color, markeredgecolor="white",
                    markeredgewidth=0.6)
        aw = adamw[np.isclose(adamw["lr"], lr)]
        if not aw.empty:
            ax.scatter([-1], [float(aw["final"].min())], color=color,
                       marker="s", s=110, edgecolor="black", linewidth=1.2, zorder=5)
        swept[_fmt_lr(lr)] = {
            "adamw": round(float(aw["final"].min()), 4) if not aw.empty else None,
            "ema": round(float(em["final"].min()), 4) if not em.empty else None,
            "best_pace": round(min(pos_to_best.values()), 4) if pos_to_best else None,
        }

    xticks = [-1] + list(range(len(C_LABELS)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(["AdamW"] + C_LABELS, rotation=45, ha="right")
    pu.flush_xaxis(ax, x_min=-1.3, x_max=len(C_LABELS) - 0.7)
    cur_lo, _ = ax.get_ylim()
    ax.set_ylim(cur_lo, 3.8)
    ax.set_title("EMA Power = 0.5")
    return swept


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig5_pretraining"
    bundle_root = data_dir / "raw_curves"
    rows = _load(data_dir / "exp25") + _load(data_dir / "exp26")

    fig, axes = pu.make_panels(1, 3)
    axes_row = axes[0]

    report = {}
    report["const_finals"] = draw_bundle_panel(axes_row[0], bundle_root, "Constant LR", ymax=4.0)

    # Middle c-sweep: keep only lrs where pullback (c>0) beats matched EMA (c=0).
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "eval_logs"} for r in rows])
    ours = df[(df["method"] == "PACE") & np.isclose(df["kappa"].astype(float).fillna(-1), 0.5)]
    ema = df[(df["method"] == "EMA") & np.isclose(df["kappa"].astype(float).fillna(-1), 0.5)]
    eligible_lrs = []
    for lr in sorted(ours["lr"].dropna().unique()):
        ours_lr = ours[np.isclose(ours["lr"], lr)]
        ema_lr = ema[np.isclose(ema["lr"], lr)]
        if ours_lr.empty or ema_lr.empty:
            continue
        ours_pos = ours_lr[ours_lr["c"] > 0]
        if ours_pos.empty:
            continue
        if (float(ema_lr["final"].min()) - float(ours_pos["final"].min())) > 0.005:
            eligible_lrs.append(lr)

    if len(eligible_lrs) > 4:
        scored = []
        for lr in eligible_lrs:
            ours_lr = ours[np.isclose(ours["lr"], lr) & (ours["c"] > 0)]
            ema_lr = ema[np.isclose(ema["lr"], lr)]
            gap = float(ema_lr["final"].min()) - float(ours_lr["final"].min())
            scored.append((gap, lr))
        scored.sort(reverse=True)
        eligible_lrs = sorted(lr for _, lr in scored[:4])

    plotted_lrs = [(lr, pu.lr_color(lr, eligible_lrs)) for lr in eligible_lrs]
    report["c_sweep_lrs"] = [_fmt_lr(lr) for lr in eligible_lrs]
    report["c_sweep"] = draw_c_sweep(axes_row[1], rows, plotted_lrs)

    report["wsd_finals"] = draw_bundle_panel(axes_row[2], bundle_root, "WSD", ymax=3.72, xmin=2000)

    axes_row[0].set_ylabel("Val Loss")
    axes_row[1].set_ylabel("")
    axes_row[2].set_ylabel("")
    axes_row[0].set_xlabel("Step")
    axes_row[1].set_xlabel("Pullback strength $c$")
    axes_row[2].set_xlabel("Step")

    leg0_h = [Line2D([0], [0], color=pu.METHOD_STYLE[m]["color"],
                     marker=pu.METHOD_STYLE[m]["marker"],
                     linestyle=pu.METHOD_STYLE[m]["linestyle"],
                     linewidth=3, markersize=9) for m in METHODS]
    pu.place_inside_legend(axes_row[0], leg0_h, METHODS)

    leg2_h = [Line2D([0], [0], color="0.15", marker="s", linestyle="None",
                     markersize=12, markeredgecolor="black")]
    leg2_l = ["AdamW"]
    for lr, color in plotted_lrs:
        leg2_h.append(Line2D([0], [0], color=color, marker="o",
                             linestyle="-", linewidth=3, markersize=9))
        leg2_l.append(f"lr = {_fmt_lr(lr)}")
    pu.place_inside_legend(axes_row[1], leg2_h, leg2_l)

    fig.align_xlabels(axes_row)
    fig.align_ylabels(axes_row)
    out = pu.savefig_pdf(fig, "fig5_pretraining")
    report["pdf"] = str(out)
    return report
