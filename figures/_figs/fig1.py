"""Fig 1 — best-per-method validation-loss trajectories.

Reproduces the paper's headline finetuning figure: for each model
(SmolLM2-1.7B, Qwen3-1.7B, Gemma3-1B) the validation cross-entropy over training
for AdamW, EMA, and PACE, each at its best configuration. Baselines use their
best decay schedule (cosine / WSD); PACE uses a constant learning rate.

Data: ``<data>/fig1/selection.json`` + the per-run JSONs it points to (bundled).
"""

import json
from pathlib import Path

import plot_utils as pu


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

    fig, axes = pu.make_panels(1, len(models))
    report = {}
    for j, model in enumerate(models):
        ax = axes[0][j]
        for method in methods:
            entry = manifest.get(f"{model}|{method}")
            if entry is None:
                continue
            d = json.load(open(data_dir / entry["file"]))
            steps, vals = _val_curve(d.get("eval_logs", []))
            style = pu.METHOD_STYLE[method]
            ax.plot(steps, vals, color=style["color"], linestyle=style["linestyle"],
                    marker=None, label=method)
            # endpoint marker at the final value
            ax.plot([steps[-1]], [vals[-1]], marker="o", color=style["color"],
                    markersize=9, markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            report[f"{model}|{method}"] = round(vals[-1], 4)
        ax.set_title(model)
        pu.flush_xaxis(ax, 0, max(steps))
        pu.place_inside_legend(ax)
    pu.single_axis_labels(axes, xlabel="Step", ylabel="Validation loss")
    out = pu.savefig_pdf(fig, "fig1_best_per_method")
    return {"pdf": str(out), "final_val_loss": report,
            "paper_pace_fv": meta["paper_pace_fv"]}
