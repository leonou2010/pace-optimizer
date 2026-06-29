"""Smoke-check the console commands are wired and respond. Offline: no training, no GPU.

Exercises the entry points from pyproject (pace-data / pace-figure / pace-train) as real
CLI invocations -- `pace-figure` actually expands the map, the others answer `--help`.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FIGMAP = yaml.safe_load(open(REPO / "src" / "pace_figures" / "FIGMAP.yaml"))


def _run(cmd):
    exe = shutil.which(cmd[0])
    if not exe:
        pytest.skip(f"{cmd[0]} not on PATH (run via `uv run pytest`)")
    return subprocess.run([exe] + cmd[1:], cwd=str(REPO), capture_output=True, text=True)


def test_pace_figure_lists_the_catalog():
    r = _run(["pace-figure", "fig=list"])
    assert r.returncode == 0, r.stderr[-2000:]
    for fid in FIGMAP:
        assert fid in r.stdout, f"{fid} missing from catalog"


def test_pace_figure_expands_fig1():
    r = _run(["pace-figure", "fig=fig1"])
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("pace-train")]
    assert len(cmds) == 9
    for cmd in cmds:
        for token in ("model=", "optimizer=", "schedule=", "seed=", "optimizer.lr="):
            assert token in cmd, f"missing {token}: {cmd}"


def test_pace_data_help():
    r = _run(["pace-data", "--help"])
    assert r.returncode == 0
    assert "--out" in r.stdout and "--tokenizer" in r.stdout


def test_pace_train_help():
    r = _run(["pace-train", "--help"])
    assert r.returncode == 0
