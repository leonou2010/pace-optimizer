"""Consistency tests for the figure-reproduction map.

Verifies that `pace-figure` produces valid, complete reproduction commands:

- test_every_fig_config_is_in_the_catalog: every config under conf/fig/ is listed in
  FIGMAP.yaml.
- test_every_figure_expands_to_wellformed_commands: each figure's curves (and any
  sweeps) expand to commands that begin with `pace-train` and include the required
  overrides (model, optimizer, schedule, seed, optimizer.lr).
- test_fig1_has_nine_curves: fig1 declares the expected nine curves (three models x
  three methods).

Offline; requires no GPU, network, or bundled data.
"""

import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "src" / "pace_figures"
sys.path.insert(0, str(FIGURES))

import reproduce  # noqa: E402


def _config_ids():
    return {p.stem for p in (FIGURES / "conf" / "fig").glob("*.yaml") if p.stem != "list"}


def _figmap():
    return yaml.safe_load(open(FIGURES / "FIGMAP.yaml"))


def test_every_fig_config_is_in_the_catalog():
    catalog = set(_figmap())
    for fid in _config_ids():
        assert fid in catalog, f"figure config '{fid}' missing from FIGMAP.yaml"


def _all_curves(cfg):
    """Every run a figure expands to: explicit curves + expanded sweep(s)."""
    curves = list(cfg.get("curves", []) or [])
    sweeps = ([cfg["sweep"]] if cfg.get("sweep") else []) + list(cfg.get("sweeps", []) or [])
    for sw in sweeps:
        curves += reproduce._expand_sweep(sw)
    return curves


def test_every_figure_expands_to_wellformed_commands():
    for fid in _config_ids():
        cfg = OmegaConf.load(FIGURES / "conf" / "fig" / f"{fid}.yaml")
        curves = _all_curves(cfg)
        assert curves, f"{fid}: no curves and no sweep -> nothing to reproduce"
        for c in curves:
            cmd = reproduce._command(cfg, c, [])
            assert cmd[0] == "pace-train"
            joined = " ".join(cmd)
            for needed in ("model=", "optimizer=", "schedule=", "seed=", "optimizer.lr="):
                assert needed in joined, f"{fid}: command missing {needed}: {joined}"


def test_fig1_has_nine_curves():
    cfg = OmegaConf.load(FIGURES / "conf" / "fig" / "fig1.yaml")
    assert len(cfg.curves) == 9, "Fig 1 should specify 9 curves (3 models x 3 methods)"
