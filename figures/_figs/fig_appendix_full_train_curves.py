"""Appendix: full finetuning validation curves with a final-quarter zoom.

Companion to Fig 1. For each model (SmolLM2-1.7B, Qwen3-1.7B, Gemma3-1B) the
validation cross-entropy over training for AdamW, EMA, and PACE, each at its best
configuration. The top row shows the full trajectory; the bottom row zooms the
x-axis to the final 25% of training steps, where the method gap is decided.

Data: reuses Fig 1's bundled per-run JSONs at ``<data>/fig1/`` (no re-bundling).
"""
import json

import plot_utils as pu

ZOOM_FRAC = 0.25  # show the final 25% of steps in the bottom row


def _val_curve(eval_logs):
    steps, vals = [], []
    for e in eval_logs:
        s = e.get("step")
        v = e.get("val_loss", e.get("vanilla_val_loss"))
        if s is None or v is None:
            continue
        steps.append(s)
        vals.append(v)
    return steps, vals


def build(cfg) -> dict:
    pu.set_plot_style()
    data_dir = pu.DATA_ROOT / "fig1"
    manifest = json.load(open(data_dir / "selection.json"))
    meta = manifest["_meta"]
    models = cfg.get("models") or meta["models"]
    methods = cfg.get("methods") or meta["methods"]

    fig, axes = pu.make_panels(2, len(models), sharey="col")
    report = {}
    for j, model in enumerate(models):
        ax_full = axes[0][j]
        ax_zoom = axes[1][j]
        max_step = 0
        zoom_lo = None
        # First pass: find the global max step for this model panel.
        curves = {}
        for method in methods:
            entry = manifest.get(f"{model}|{method}")
            if entry is None:
                continue
            d = json.load(open(data_dir / entry["file"]))
            steps, vals = _val_curve(d.get("eval_logs", []))
            curves[method] = (steps, vals)
            if steps:
                max_step = max(max_step, max(steps))
        zoom_lo = max_step * (1.0 - ZOOM_FRAC)

        for method, (steps, vals) in curves.items():
            style = pu.METHOD_STYLE[method]
            for ax in (ax_full, ax_zoom):
                ax.plot(steps, vals, color=style["color"],
                        linestyle=style["linestyle"], marker=None, label=method)
                ax.plot([steps[-1]], [vals[-1]], marker="o", color=style["color"],
                        markersize=9, markeredgecolor="black", markeredgewidth=1.5,
                        zorder=5)
            report[f"{model}|{method}"] = round(vals[-1], 4)

        ax_full.set_title(model)
        pu.flush_xaxis(ax_full, 0, max_step)
        pu.flush_xaxis(ax_zoom, zoom_lo, max_step)

        # Bottom-row y-limits: tighten to the values within the zoom window so the
        # final-quarter gap is legible.
        zvals = [v for steps, vals in curves.values()
                 for s, v in zip(steps, vals) if s >= zoom_lo]
        if zvals:
            lo, hi = min(zvals), max(zvals)
            pad = (hi - lo) * 0.08 or 0.01
            ax_zoom.set_ylim(lo - pad, hi + pad)

        pu.place_inside_legend(ax_full)

    pu.single_axis_labels(axes, xlabel="Step", ylabel="Validation loss")
    out = pu.savefig_pdf(fig, "fig_appendix_full_train_curves")
    return {"pdf": str(out), "final_val_loss": report,
            "zoom_frac": ZOOM_FRAC}
