"""Fig 3 (body) — 135M granular ablation, EMA power kappa=0.3 row only.

1 row x 3 cols, one panel per update frequency in {1, 5, 10}, all at kappa=0.3.
Per panel: x-axis is ["AdamW", "EMA", c-grid...]; one lr-gradient line per lr
(light = small lr), AdamW as a black/lr-coloured square at the "AdamW" x-tick,
EMA at the "EMA" position. Inside top-right legends distributed across panels.

Data: SmolLM2-135M exp13 chunked results (``<data>/fig3_135m_grid/*_chunk*.json``).
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

import plot_utils as pu

C_VALUES = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
KAPPA = 0.3
UFS = [1, 5, 10]

X_LABELS = ["AdamW", "EMA"] + [
    f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+") for v in C_VALUES
]
X_POS = {"AdamW": 0, "EMA": 1}
for _i, _c in enumerate(C_VALUES):
    X_POS[_c] = _i + 2


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if s.startswith(("1.0e", "2.0e", "3.0e", "5.0e")):
        s = s[0] + s[3:]
    return s


def _c_to_x(c):
    for ref, pos in X_POS.items():
        if isinstance(ref, (int, float)) and isinstance(c, (int, float)):
            if abs(float(ref) - float(c)) < 1e-12:
                return pos
    return None


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig3_135m_grid"
    df = pu.load_chunks(data_dir)
    df = df.drop_duplicates(subset=["run_id"], keep="first")
    if df.empty:
        raise SystemExit("no exp13 data found")

    kappa = float(cfg.get("kappa", KAPPA))
    ufs = list(cfg.get("ufs") or UFS)

    # Drop divergent runs (135M smoltalk val ~1.0-1.3 nats; >=1.5 is a spike).
    df = df[df["final_val_loss"] < 1.5]

    adamw_df = df[df["method"] == "AdamW"]
    ema_df = df[(df["method"] == "EMA") | ((df["c"] == 0) & (df["method"] != "AdamW"))]
    ours_df = df[(df["method"] == "PACE") & (df["c"] != 0)]

    # Sparse-lr filter: keep an lr only if best PACE at that lr beats matched
    # EMA and AdamW by >0.005 nats; then sub-sample to ~3 lrs.
    all_lrs_kappa = sorted(ours_df[np.isclose(
        ours_df["kappa"].astype(float).fillna(-1), kappa)]["lr"].dropna().unique())
    keep = []
    for lr in all_lrs_kappa:
        ours_lr = ours_df[
            np.isclose(ours_df["lr"].astype(float).fillna(-1), lr)
            & np.isclose(ours_df["kappa"].astype(float).fillna(-1), kappa)
        ]
        ema_lr = ema_df[
            np.isclose(ema_df["lr"].astype(float).fillna(-1), lr)
            & np.isclose(ema_df["kappa"].astype(float).fillna(-1), kappa)
        ]
        aw_lr = adamw_df[np.isclose(adamw_df["lr"].astype(float).fillna(-1), lr)]
        if ours_lr.empty or ema_lr.empty or aw_lr.empty:
            continue
        o_best = float(ours_lr["final_val_loss"].min())
        e_best = float(ema_lr["final_val_loss"].min())
        a_best = float(aw_lr["final_val_loss"].min())
        if (e_best - o_best) > 0.005 and (a_best - o_best) > 0.005:
            keep.append(lr)
    if len(keep) > 3:
        keep = [keep[0], keep[len(keep) // 2], keep[-1]]
    lrs_at_kappa = keep

    fig, axes = pu.make_panels(1, len(ufs), sharey=False)
    axes_row = axes[0]

    panel_lrs = [[] for _ in ufs]
    for i, lr in enumerate(lrs_at_kappa):
        panel_lrs[i % len(ufs)].append(lr)

    report = {"kappa": kappa, "lrs": [_fmt_lr(lr) for lr in lrs_at_kappa], "panels": {}}

    for col, uf in enumerate(ufs):
        ax = axes_row[col]
        ax.set_title(f"Frequency = {uf}", pad=10)
        for lr in lrs_at_kappa:
            color = pu.lr_color(lr, lrs_at_kappa)
            sub = ours_df[
                np.isclose(ours_df["lr"].astype(float).fillna(-1), lr)
                & np.isclose(ours_df["kappa"].astype(float).fillna(-1), kappa)
                & np.isclose(ours_df["uf"].astype(float).fillna(-1), uf)
            ]
            if sub.empty:
                continue
            best_at_c = sub.groupby("c")["final_val_loss"].min()
            ema_sub = ema_df[
                np.isclose(ema_df["lr"].astype(float).fillna(-1), lr)
                & np.isclose(ema_df["kappa"].astype(float).fillna(-1), kappa)
                & np.isclose(ema_df["uf"].astype(float).fillna(-1), uf)
            ]
            ema_best = float(ema_sub["final_val_loss"].min()) if not ema_sub.empty else np.nan
            xs, ys = [], []
            if not np.isnan(ema_best):
                xs.append(X_POS["EMA"]); ys.append(ema_best)
            for c, v in best_at_c.items():
                pos = _c_to_x(c)
                if pos is None:
                    continue
                xs.append(pos); ys.append(float(v))
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            if xs:
                ax.plot(xs, ys, marker="o", color=color, markeredgecolor="white",
                        markeredgewidth=0.6)
            adamw_best = adamw_df[np.isclose(
                adamw_df["lr"].astype(float).fillna(-1), lr)]["final_val_loss"].min()
            if not np.isnan(adamw_best):
                ax.scatter([X_POS["AdamW"]], [float(adamw_best)], color=color,
                           marker="s", s=110, edgecolor="black", linewidth=1.2, zorder=5)
            report["panels"].setdefault(f"uf={uf}", {})[_fmt_lr(lr)] = {
                "best_pace": round(float(best_at_c.min()), 4) if not best_at_c.empty else None,
                "ema": round(ema_best, 4) if not np.isnan(ema_best) else None,
                "adamw": round(float(adamw_best), 4) if not np.isnan(adamw_best) else None,
            }

        ax.set_xticks(list(range(len(X_LABELS))))
        ax.set_xticklabels(X_LABELS, rotation=45, ha="right")
        pu.flush_xaxis(ax, x_min=-0.3, x_max=len(X_LABELS) - 0.7)
        ax.set_xlabel("Pullback strength $c$")
        if col == 0:
            ax.set_ylabel("Val Loss")

        leg_h, leg_l = [], []
        for lr in panel_lrs[col]:
            leg_h.append(Line2D([0], [0], color=pu.lr_color(lr, lrs_at_kappa),
                                marker="o", linestyle="-", linewidth=3, markersize=9))
            leg_l.append(f"lr = {_fmt_lr(lr)}")
        if leg_h:
            pu.place_inside_legend(ax, leg_h, leg_l)

    # Per-panel y-zoom to include every plotted point.
    for col, uf in enumerate(ufs):
        ax = axes_row[col]
        ours_pts = ours_df[
            np.isclose(ours_df["kappa"].astype(float).fillna(-1), kappa)
            & np.isclose(ours_df["uf"].astype(float).fillna(-1), uf)
            & ours_df["lr"].isin(lrs_at_kappa)
        ]["final_val_loss"].dropna()
        ema_pts = ema_df[
            np.isclose(ema_df["kappa"].astype(float).fillna(-1), kappa)
            & np.isclose(ema_df["uf"].astype(float).fillna(-1), uf)
            & ema_df["lr"].isin(lrs_at_kappa)
        ]["final_val_loss"].dropna()
        aw_pts = adamw_df[adamw_df["lr"].isin(lrs_at_kappa)]["final_val_loss"].dropna()
        import pandas as pd
        all_pts = pd.concat([ours_pts, ema_pts, aw_pts])
        if all_pts.empty:
            continue
        ax.set_ylim(float(all_pts.min()) - 0.005, float(all_pts.max()) + 0.015)

    out = pu.savefig_pdf(fig, "fig3_135m_grid")
    report["pdf"] = str(out)
    return report
