"""Tests for the console commands -- the documented CLI actually runs.

- `pace-figure` expands the figure map (offline).
- the README quickstart runs end to end: `pace-data` prepares a small dataset, then
  `pace-train` trains a few steps on the small model. Needs network and (ideally) a GPU.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FIGMAP = yaml.safe_load(open(REPO / "src" / "pace_figures" / "FIGMAP.yaml"))


def _exe(name):
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not on PATH (run via `uv run pytest`)")
    return path


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, **kw)


def test_pace_figure_lists_the_catalog():
    r = _run([_exe("pace-figure"), "fig=list"])
    assert r.returncode == 0, r.stderr[-2000:]
    for fid in FIGMAP:
        assert fid in r.stdout, f"{fid} missing from catalog"


def test_pace_figure_expands_fig1():
    r = _run([_exe("pace-figure"), "fig=fig1"])
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("pace-train")]
    assert len(cmds) == 9
    for cmd in cmds:
        for token in ("model=", "optimizer=", "schedule=", "seed=", "optimizer.lr="):
            assert token in cmd, f"missing {token}: {cmd}"


def test_readme_quickstart(tmp_path):
    """Run the documented `pace-data` then `pace-train` commands on the small model."""
    data = tmp_path / "smoltalk"
    r = _run([_exe("pace-data"), "--out", str(data),
              "--tokenizer", "HuggingFaceTB/SmolLM2-135M",
              "--max-seq-len", "256", "--max-train", "512", "--max-val", "128"])
    if r.returncode != 0:
        pytest.skip(f"pace-data could not fetch from HuggingFace:\n{r.stderr[-1500:]}")
    assert (data / "train").exists()

    run_dir = tmp_path / "run"
    r = _run([_exe("pace-train"),
              "model=smollm2_135m", "optimizer=pace", "schedule=const",
              "optimizer.lr=1e-3", "optimizer.lambda_pullback=0.15", "optimizer.ema_kappa=0.2",
              f"dataset.path={data}",
              "training.batch_size=8", "training.gradient_accumulation_steps=1",
              "training.max_steps=5", "training.eval_interval=2", "training.eval_samples=64",
              "schedule.warmup_steps=1", "seed=42"],
             env=dict(os.environ, PACE_RUN_DIR=str(run_dir)))
    assert r.returncode == 0, r.stdout[-2000:] + "\n" + r.stderr[-2000:]

    js = [j for j in run_dir.rglob("*.json") if "hydra" not in str(j)]
    assert js, "pace-train wrote no result JSON"
    fv = json.load(open(js[-1]))["summary"]["final_val_loss"]
    assert fv == fv and fv < 100, f"bad final_val_loss {fv}"
