"""End-to-end flow test: run the real pipeline on the small model and check it works.

Prepares data with ``pace.data`` and then drives the figure map through the real
``pace-figure`` -> ``pace-train`` flow, so a single test exercises every module
(data, reproduce, train, schedules, model, optimizer). A few steps by default; needs
network and a GPU. The committed reference runs live in ``tests/fixtures/results/``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STEPS = int(os.environ.get("PACE_TEST_STEPS", "5"))
# enough data for a few optimizer steps at the config's effective batch size (128)
TRAIN = int(os.environ.get("PACE_TEST_TRAIN", "1024"))
CACHE = Path(os.environ.get("PACE_TEST_DATA_CACHE", Path.home() / ".cache" / "pace_test"))


def test_pipeline_runs_small_model(tmp_path):
    sys.path.insert(0, str(REPO / "src"))
    # 1. prepare a tiny dataset (exercises pace.data)
    data = CACHE / f"pipeline_135m_{TRAIN}"
    if not (data / "train").exists():
        try:
            from pace.data import prepare_sft
            prepare_sft(str(data), tokenizer_name="HuggingFaceTB/SmolLM2-135M",
                        max_seq_len=256, max_train=TRAIN, max_val=128)
        except Exception as e:
            pytest.skip(f"could not fetch model/data from HuggingFace: {e}")

    # 2. run the figure map end-to-end (pace-figure -> pace-train) for the 135M figure.
    #    limit=6 covers the AdamW baselines and the first PACE run.
    out = tmp_path / "runs"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    cmd = [sys.executable, "-m", "pace_figures.reproduce",
           "fig=fig3_135m_grid", "run=true", f"data={data}", f"out={out}",
           "limit=6", f"max_steps={STEPS}", f"eval_interval={max(1, STEPS // 2)}",
           "eval_samples=64"]
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-3000:] + "\n" + r.stderr[-3000:])

    # 3. each run wrote a result with a finite loss, and PACE actually ran
    results = [j for j in out.rglob("*.json") if "hydra" not in str(j)]
    assert len(results) >= 6, f"expected >=6 result files, got {len(results)}"
    pace_ran = False
    for j in results:
        d = json.load(open(j))
        opt, fv = d["config"]["optimizer"], d["summary"]["final_val_loss"]
        steps = d["summary"]["total_steps"]
        print(f"  trained {opt['name']:5} lr={opt['lr']} c={opt.get('lambda_pullback')} "
              f"-> {steps} steps, final_val_loss={fv:.4f}")
        assert fv == fv and fv < 100, f"{j.name}: bad final_val_loss {fv}"
        if opt.get("lambda_pullback", 0) > 0:
            pace_ran = True
    assert pace_ran, "no PACE run produced in the pipeline"
