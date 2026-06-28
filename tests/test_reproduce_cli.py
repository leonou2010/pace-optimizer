"""Check the documented reproduction commands work: `pace-figure fig=list` shows the
catalog and `pace-figure fig=fig1` prints Figure 1's training commands. Offline -- this
only expands the map to commands, it doesn't train.
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
FIGMAP = yaml.safe_load(open(REPO / "src" / "pace_figures" / "FIGMAP.yaml"))


def _figure(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, "-m", "pace_figures.reproduce", *args],
                          cwd=str(REPO), env=env, capture_output=True, text=True)


def test_fig_list_shows_every_figure():
    r = _figure("fig=list")
    assert r.returncode == 0, r.stderr[-2000:]
    for fid in FIGMAP:
        assert fid in r.stdout, f"{fid} missing from the catalog"


def test_fig1_prints_its_training_commands():
    r = _figure("fig=fig1")
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("pace-train")]
    assert len(cmds) == 9, f"expected 9 commands, got {len(cmds)}"
    for cmd in cmds:
        for token in ("model=", "optimizer=", "schedule=", "seed=", "optimizer.lr="):
            assert token in cmd, f"missing {token}: {cmd}"
