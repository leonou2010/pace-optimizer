"""The reproduction map must be self-consistent: every figure config is in the
catalog, and each curve expands to a well-formed pace-train command.
"""

import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
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


def test_curves_expand_to_wellformed_commands():
    for fid in _config_ids():
        cfg = OmegaConf.load(FIGURES / "conf" / "fig" / f"{fid}.yaml")
        for c in cfg.get("curves", []) or []:
            cmd = reproduce._command(cfg, c, [])
            assert cmd[0] == "pace-train"
            joined = " ".join(cmd)
            for needed in ("model=", "optimizer=", "schedule=", "seed=", "optimizer.lr="):
                assert needed in joined, f"{fid}: command missing {needed}: {joined}"


def test_fig1_has_nine_curves():
    cfg = OmegaConf.load(FIGURES / "conf" / "fig" / "fig1.yaml")
    assert len(cfg.curves) == 9, "Fig 1 should specify 9 curves (3 models x 3 methods)"
