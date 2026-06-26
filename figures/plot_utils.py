"""Shared plotting utilities for the PACE paper figures.

A single source of truth for the figure style: serif fonts, a light-grey panel
background with a thick white grid, bold titles/labels, thick lines with
black-ringed circle markers, and per-learning-rate color gradients for the
ablation panels. Every figure builder calls :func:`set_plot_style` once and uses
:func:`make_panels` so that each panel renders at exactly the same physical size
on the page regardless of the grid shape.

Do NOT override fontsize / linewidth / markersize inside individual figure
scripts — change them here so the whole paper stays consistent.

Data loaders are schema-agnostic: they read the chunked ``*_chunk*.json`` format
and the per-run JSON format (with ``eval_logs``) used by the bundled results.
The results directory defaults to ``<repo>/data/results`` and can be overridden
with the ``PACE_DATA`` environment variable; the figure output directory defaults
to ``<repo>/figures/rendered`` and can be overridden with ``PACE_FIG_OUT``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PACE_DATA", REPO_ROOT / "data" / "results"))
FIG_DIR = Path(os.environ.get("PACE_FIG_OUT", REPO_ROOT / "figures" / "rendered"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Font sizes (canonical paper style)
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 15
LABEL_FONTSIZE = 19
TICK_FONTSIZE = 15

# --- Global panel sizing -----------------------------------------------------
# figsize = (16/3 * ncols, 4 * nrows) per panel ~ 5.33 x 4 inches.
PANEL_W = 16.0 / 3.0
PANEL_H = 4.0
# Absolute (inches) margins used by `make_panels()`. Keeping these constant in
# inches across figures is the only way every panel renders at exactly the same
# size on the page, regardless of nrows / ncols.
MARGIN_LEFT_IN = 1.30     # ylabel + y-tick numerical labels
MARGIN_RIGHT_IN = 0.55    # so the last x-tick label can extend past the panel
MARGIN_TOP_IN = 0.80      # bold title + legend if at top
MARGIN_BOTTOM_IN = 1.50   # xlabel + 45deg x-tick labels (long: "AdamW", "5e-3")
WSPACE_IN = 1.30          # column N+1 labels must clear column N's right edge
HSPACE_IN = 1.70          # row N+1 title must clear row N's xlabel + tick labels


def make_panels(nrows: int, ncols: int, extra_left_in: float = 0.0, **kw):
    """Create (fig, axes) where every axis is exactly PANEL_W x PANEL_H inches
    regardless of grid shape. Margins and inter-panel spacing are fixed in inches
    (not as fractions of figsize), so a 1x3 panel and a 2x3 panel have axes of
    identical physical size on the rendered page.

    `extra_left_in` adds extra left-side whitespace (in inches) without changing
    the panel box size (for figures whose leftmost ylabel is unusually wide).
    """
    left_in = MARGIN_LEFT_IN + extra_left_in
    fig_w = left_in + ncols * PANEL_W + (ncols - 1) * WSPACE_IN + MARGIN_RIGHT_IN
    fig_h = MARGIN_BOTTOM_IN + nrows * PANEL_H + (nrows - 1) * HSPACE_IN + MARGIN_TOP_IN
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, **kw)
    fig.subplots_adjust(
        left=left_in / fig_w,
        right=1 - MARGIN_RIGHT_IN / fig_w,
        bottom=MARGIN_BOTTOM_IN / fig_h,
        top=1 - MARGIN_TOP_IN / fig_h,
        wspace=WSPACE_IN / PANEL_W,
        hspace=HSPACE_IN / PANEL_H,
    )
    return fig, axes


# --- Method styling ----------------------------------------------------------
#   PACE  = our method  -> cm.Reds
#   EMA   = EMA baseline -> cm.Blues
#   AdamW = no EMA       -> black, dashed
# Shade 0.75 gives a clear mid-saturation pull that stands out on the light-grey
# #EFEFEF panel background.
METHOD_STYLE = {
    "AdamW": dict(color="black", marker="o", linestyle="--"),
    "EMA":   dict(color=mpl.cm.Blues(0.75), marker="o", linestyle="-"),
    "PACE":  dict(color=mpl.cm.Reds(0.75), marker="o", linestyle="-"),
}
METHOD_COLORS = {k: v["color"] for k, v in METHOD_STYLE.items()}

REF_ADAMW = "black"
REF_EMA = mpl.cm.Blues(0.75)

# Default lr palette: cm.Reds shading. Light shade = smallest lr, dark = largest.
LR_CMAP = mpl.cm.Reds


def _resolve_cmap(cmap):
    if cmap is None:
        return LR_CMAP
    if hasattr(cmap, "__call__"):
        return cmap
    return mpl.colormaps[cmap]


def lr_color(lr: float, all_lrs, cmap=None):
    """Map an lr to a color on the lr gradient using LogNorm over the lr range.

    Output is restricted to [0.35, 0.90] so the lightest line stays visible on
    the grey panel background. Light = small lr, dark = large lr.
    """
    cm = _resolve_cmap(cmap)
    if not all_lrs:
        return cm(0.7)
    lo, hi = float(min(all_lrs)), float(max(all_lrs))
    if lo == hi:
        return cm(0.7)
    t = (np.log10(float(lr)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    t = 0.35 + 0.55 * t
    return cm(t)


def flush_xaxis(ax, x_min: float | None = None, x_max: float | None = None) -> None:
    """Pin the data range to the spine edges (no left/right matplotlib margin)."""
    ax.margins(x=0)
    cur_lo, cur_hi = ax.get_xlim()
    ax.set_xlim(left=x_min if x_min is not None else cur_lo,
                right=x_max if x_max is not None else cur_hi)


def single_axis_labels(axes_2d, xlabel: str, ylabel: str) -> None:
    """Apply x/y labels only on the outer edge of a 2-D axes array."""
    arr = np.atleast_2d(axes_2d)
    nrows, ncols = arr.shape
    for i in range(nrows):
        for j in range(ncols):
            ax = arr[i, j]
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.set_xlabel(xlabel if i == nrows - 1 else "")


def place_inside_legend(ax, handles=None, labels=None, **kw):
    """Inside-the-axes legend at the corner with the most empty space."""
    defaults = dict(loc="best", framealpha=0.92, edgecolor="0.4",
                    fancybox=True, borderpad=0.5)
    defaults.update(kw)
    if handles is not None and labels is not None:
        return ax.legend(handles, labels, **defaults)
    return ax.legend(**defaults)


def adamw_point(ax, x_pos: float, val: float, label: str = "AdamW") -> None:
    """Plot the AdamW baseline as a single black square at `x_pos`."""
    style = METHOD_STYLE["AdamW"]
    ax.plot([x_pos], [val], marker=style["marker"], color=style["color"],
            markersize=14, markeredgewidth=1.5, linestyle="None",
            zorder=5, label=label)


def set_plot_style() -> None:
    """Canonical PACE figure style: serif font, light-grey panel background
    (#EFEFEF), thick white grid, bold titles + labels, thick lines (lw=7) with
    black-ringed circle markers. Single source of truth — DO NOT override
    fontsize/linewidth/markersize inside individual scripts.
    """
    mpl.rcParams.update({
        # font
        "font.size": 11,
        "font.family": "serif",
        # panel background + thick white grid
        "axes.grid": True,
        "axes.facecolor": "#EFEFEF",
        "grid.color": "white",
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "grid.linewidth": 3.0,
        # spine + label/title weight
        "axes.linewidth": 1.5,
        "axes.labelsize": LABEL_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        # ticks
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        # lines + markers (thick, with black-ringed dots)
        "lines.linewidth": 7.0,
        "lines.markersize": 10,
        "lines.markeredgewidth": 2.0,
        "lines.markeredgecolor": "black",
        # legend
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.title_fontsize": LEGEND_FONTSIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.fancybox": False,
        "legend.edgecolor": "0.4",
        # figure-level
        "figure.titlesize": TITLE_FONTSIZE,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
        "figure.subplot.left": 0.10,
        "figure.subplot.right": 0.97,
        "figure.subplot.top": 0.90,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.wspace": 0.22,
        "figure.subplot.hspace": 0.30,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Shared sizing constants for annotations that rcParams can't cover
ANNOT_FONTSIZE = 15
PLACEHOLDER_FONTSIZE = 26
PLACEHOLDER_TAG_FONTSIZE = 16
ENDPOINT_MARKER_SIZE = 90


def uniform_grid(ax, nx: int = 5, ny: int = 5):
    """Force a consistent number of ticks (gridlines) on both axes."""
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nx))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=ny))


def make_wide_panel(**kw):
    """Single panel spanning the width of a 1x2 grid (for single-plot figures)."""
    wide_w = 2 * PANEL_W + WSPACE_IN
    fig_w = MARGIN_LEFT_IN + wide_w + MARGIN_RIGHT_IN
    fig_h = MARGIN_BOTTOM_IN + PANEL_H + MARGIN_TOP_IN
    fig, axes = plt.subplots(1, 1, figsize=(fig_w, fig_h), squeeze=False, **kw)
    fig.subplots_adjust(
        left=MARGIN_LEFT_IN / fig_w,
        right=1 - MARGIN_RIGHT_IN / fig_w,
        bottom=MARGIN_BOTTOM_IN / fig_h,
        top=1 - MARGIN_TOP_IN / fig_h,
    )
    return fig, axes


def savefig_pdf(fig, name: str) -> Path:
    """Save `fig` to ``<FIG_DIR>/<name>.pdf`` (and .png) with dimensions exactly
    equal to figsize (no tight-bbox auto-cropping), so every figure renders at
    identical per-panel size after \\includegraphics.
    """
    out = FIG_DIR / f"{name}.pdf"
    bb = mpl.transforms.Bbox([[0, 0], list(fig.get_size_inches())])
    fig.savefig(out, bbox_inches=bb, pad_inches=0)
    fig.savefig(out.with_suffix(".png"), bbox_inches=bb, pad_inches=0)
    return out


_NAME_RE = re.compile(
    r"lr(?P<lr>[0-9.e+-]+)"
    r"(?:_k(?P<k>[0-9.]+))?"
    r"(?:_c(?P<c>[0-9.e+-]+))?"
    r"(?:_uf(?P<uf>[0-9]+))?"
    r"(?:_bs(?P<bs>[0-9]+))?"
    r"(?:_s(?P<s>[0-9]+))?"
)


def _parse_name(name: str) -> dict:
    """Extract (method, lr, c, kappa, uf, seed) from a run name string."""
    s = name or ""
    out = {"method": None, "lr": None, "c": None, "kappa": None, "uf": None, "seed": None}
    if "_PB_" in s or "pullback" in s.lower():
        out["method"] = "PACE"
    elif "_EMA_" in s or "adamw_ema" in s.lower():
        out["method"] = "EMA"
    elif "adamw" in s.lower() or "AdamW" in s:
        out["method"] = "AdamW"
    m = _NAME_RE.search(s)
    if m:
        for k in ("lr", "c", "s"):
            v = m.group(k)
            if v is not None:
                try:
                    out["seed" if k == "s" else k] = float(v)
                except ValueError:
                    pass
        kv = m.group("k")
        if kv is not None:
            try:
                out["kappa"] = float(kv)
            except ValueError:
                pass
        ufv = m.group("uf")
        if ufv is not None:
            try:
                out["uf"] = int(ufv)
            except ValueError:
                pass
    return out


def _method_from_config(opt: dict, c, name: str) -> str:
    """Classify a run as PACE / EMA / AdamW from its optimizer config."""
    parsed = _parse_name(name)
    if parsed["method"] is not None:
        return parsed["method"]
    if c and float(c) > 0:
        return "PACE"
    if opt.get("use_ema_eval"):
        return "EMA"
    return "AdamW"


def load_chunks(results_dir: Path | str, pattern: str = "*_chunk*.json") -> pd.DataFrame:
    """Load chunked result JSONs into a tidy DataFrame.

    Each chunk JSON is a list of run dicts with at least `name`, `lr`, `kappa`,
    `summary.best_val_loss`.
    """
    results_dir = Path(results_dir)
    rows: list[dict] = []
    if not results_dir.exists():
        return pd.DataFrame()
    for f in sorted(results_dir.glob(pattern)):
        try:
            data = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict):
                continue
            summary = r.get("summary") or {}
            best = summary.get("best_val_loss")
            if best is None:
                continue
            name = r.get("name", "")
            parsed = _parse_name(name)
            method = parsed["method"]
            if method is None:
                c = r.get("c")
                if r.get("adamw") or (c in (0, 0.0, None) and not r.get("baseline")):
                    method = "AdamW"
                elif r.get("baseline") or c in (0, 0.0, None):
                    method = "EMA"
                else:
                    method = "PACE"
            rows.append({
                "run_id": name,
                "source": str(f),
                "method": method,
                "lr": float(r.get("lr")) if r.get("lr") is not None else parsed["lr"],
                "c": r.get("c") if r.get("c") is not None else parsed["c"],
                "kappa": float(r["kappa"]) if r.get("kappa") is not None else parsed["kappa"],
                "uf": r.get("uf") if r.get("uf") is not None else parsed["uf"],
                "best_val_loss": float(best),
                "final_val_loss": summary.get("final_val_loss"),
                "role": r.get("role"),
                "adamw": bool(r.get("adamw", False)),
                "baseline": bool(r.get("baseline", False)),
            })
    return pd.DataFrame(rows)


def load_per_run_jsons(results_dir: Path | str) -> pd.DataFrame:
    """Load per-run JSON files (per-config subdirs) with eval_logs."""
    results_dir = Path(results_dir)
    rows: list[dict] = []
    if not results_dir.exists():
        return pd.DataFrame()
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        for jf in sub.glob("*.json"):
            try:
                d = json.load(open(jf))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(d, dict):
                continue
            summary = d.get("summary") or {}
            cfg = d.get("config") or {}
            opt = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), dict) else {}
            name = sub.name
            parsed = _parse_name(name)
            lr = opt.get("lr", parsed["lr"])
            c = opt.get("lambda_pullback", opt.get("c", parsed["c"]))
            kappa = opt.get("ema_kappa", opt.get("kappa", parsed["kappa"]))
            uf = opt.get("ema_update_freq", opt.get("update_freq", parsed["uf"]))
            method = _method_from_config(opt, c, name)
            rows.append({
                "run_id": name,
                "source": str(jf),
                "method": method,
                "lr": float(lr) if lr is not None else None,
                "c": float(c) if c is not None else None,
                "kappa": float(kappa) if kappa is not None else None,
                "uf": int(uf) if uf is not None else None,
                "best_val_loss": summary.get("best_val_loss"),
                "final_val_loss": summary.get("final_val_loss"),
                "eval_logs": d.get("eval_logs") or [],
                "step_logs": d.get("step_logs") or [],
            })
    return pd.DataFrame(rows)


@dataclass
class TrainingCurve:
    method: str
    lr: float | None
    c: float | None
    kappa: float | None
    steps: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    ema_val_loss: np.ndarray | None
    best_val_loss: float


def curve_from_eval_logs(row: pd.Series) -> TrainingCurve:
    logs = row.get("eval_logs") or []
    steps = np.array([e.get("step", i) for i, e in enumerate(logs)], dtype=float)
    train = np.array([e.get("train_loss", np.nan) for e in logs], dtype=float)
    val = np.array([e.get("val_loss", e.get("vanilla_val_loss", np.nan)) for e in logs], dtype=float)
    ema = np.array([e.get("ema_val_loss", np.nan) for e in logs], dtype=float)
    if np.all(np.isnan(ema)):
        ema = None
    return TrainingCurve(
        method=row["method"],
        lr=row.get("lr"),
        c=row.get("c"),
        kappa=row.get("kappa"),
        steps=steps,
        train_loss=train,
        val_loss=val,
        ema_val_loss=ema,
        best_val_loss=float(row.get("best_val_loss") or np.nan),
    )


def legible_c_label(c) -> str:
    """Format a c value like '1e-04', '3e-03', '3e-01'."""
    if c is None:
        return "—"
    if isinstance(c, str):
        return c
    if c == 0:
        return "0"
    return f"{c:.0e}".replace("e-0", "e-").replace("e+0", "e+")
