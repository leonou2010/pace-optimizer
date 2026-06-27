"""End-to-end training test for a freshly cloned repository.

Downloads a model and the smol-smoltalk dataset from HuggingFace, builds the training
data with the standard recipe, and runs PACE through the real `pace.train` entry point.
It then verifies that:

- the training process completes successfully and writes a result JSON with the
  {config, summary, eval_logs} structure;
- the run is a PACE run (lambda_pullback > 0), the EMA evaluation is populated, and the
  optimizer's lambda statistics are recorded;
- the final train and validation losses are finite and agree with a committed reference
  (tests/fixtures/smoke_reference.json) within PACE_SMOKE_TOL.

No bundled data or cluster paths; requires network access and uses a GPU when available.

The run is configured through environment variables, all with defaults:

    PACE_TEST_STEPS=20           # number of optimizer steps to train
    PACE_TEST_TRAIN=512          # training examples ('all' = the full dataset)
    PACE_TEST_VAL=128            # validation examples ('all' = the full set)
    PACE_TEST_BIG=1              # also run SmolLM2-1.7B (default: SmolLM2-135M only)
    PACE_SMOKE_TOL=0.30          # tolerance for the final-loss check vs the reference

The prepared dataset is cached under ~/.cache/pace_test/, so subsequent runs repeat only
the training. The validation loss is computed over the full prepared validation set.

    uv run pytest tests/test_train_smoke.py
    PACE_TEST_STEPS=200 uv run pytest tests/test_train_smoke.py    # train longer
    PACE_TEST_TRAIN=all PACE_TEST_STEPS=1778 uv run pytest ...     # full-scale run
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

STEPS = int(os.environ.get("PACE_TEST_STEPS", "20"))
TRAIN = os.environ.get("PACE_TEST_TRAIN", "512")
VAL = os.environ.get("PACE_TEST_VAL", "128")
CACHE = Path(os.environ.get("PACE_TEST_DATA_CACHE", Path.home() / ".cache" / "pace_test"))

# Reference final losses for the DEFAULT smoke config (SmolLM2-135M, 20 steps,
# train=512, val=128, seq_len=512, seed=42, const schedule, warmup=1) live in the
# fixture file, not in code. Same-seed runs drift across hardware / library versions, so
# the check is within PACE_SMOKE_TOL and is skipped if any knob above is overridden.
import json as _json  # noqa: E402

_REF = _json.load(open(Path(__file__).parent / "fixtures" / "smoke_reference.json"))
SMOKE_REF = _REF["reference"]
SMOKE_TOL = float(os.environ.get("PACE_SMOKE_TOL", str(_REF.get("default_tol", 0.30))))
AT_DEFAULTS = (STEPS == 20 and TRAIN == "512" and VAL == "128")

MODELS = [("smollm2_135m", "HuggingFaceTB/SmolLM2-135M")]
if os.environ.get("PACE_TEST_BIG") == "1":
    MODELS.append(("smollm2_1.7b", "HuggingFaceTB/SmolLM2-1.7B"))


def _prepare_data(hf_model):
    """Prepare (and cache) a tokenized smol-smoltalk dataset for this model."""
    n_train = None if TRAIN.lower() == "all" else int(TRAIN)
    n_val = None if VAL.lower() == "all" else int(VAL)
    # `r2` is the data-recipe version: bump it whenever the tokenization changes so a
    # stale cache from an older recipe is never silently reused.
    tag = f"{hf_model.split('/')[-1]}_{TRAIN}_{VAL}_r2"
    out = CACHE / tag
    if (out / "train").exists() and (out / "val").exists():
        return out
    try:
        from pace.data import prepare_sft
        prepare_sft(str(out), tokenizer_name=hf_model,
                    dataset_name="HuggingFaceTB/smol-smoltalk",
                    max_seq_len=512, max_train=n_train, max_val=n_val)
    except Exception as e:
        pytest.skip(f"could not fetch model/data from HuggingFace: {e}")
    return out


@pytest.mark.parametrize("cfg_name,hf_model", MODELS)
def test_pace_trains(cfg_name, hf_model, tmp_path):
    data_dir = _prepare_data(hf_model)
    run_dir = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PACE_RUN_DIR=str(run_dir))
    cmd = [
        sys.executable, "-m", "pace.train",
        f"model={cfg_name}", "optimizer=pace", "schedule=const",
        f"dataset.path={data_dir}",
        "training.batch_size=2", "training.gradient_accumulation_steps=1",
        f"training.max_steps={STEPS}",
        f"training.eval_interval={max(1, STEPS // 2)}",
        # eval_samples left at its default (null) -> final eval over the FULL val set
        "schedule.warmup_steps=1", "seed=42",
    ]
    r = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, f"pace-train failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"

    js = [j for j in run_dir.rglob("*.json") if "hydra" not in str(j)]
    assert js, "no result JSON written"
    d = json.load(open(js[-1]))

    assert {"config", "summary", "eval_logs"} <= set(d), "bad result schema"
    assert d["config"]["optimizer"]["lambda_pullback"] > 0, "not a PACE run"
    fv = d["summary"]["final_val_loss"]
    ft = d["summary"]["final_train_loss"]
    assert fv == fv and fv < 100, f"final_val_loss not finite/sane: {fv}"
    assert ft == ft and ft < 100, f"final_train_loss not finite/sane: {ft}"
    assert d["eval_logs"], "no eval logs"
    last = d["eval_logs"][-1]
    assert last.get("ema_val_loss") is not None, "EMA eval not populated"
    assert "lambda_clipped_frac" in last, "PACE lambda stats missing"
    print(f"[{cfg_name}] steps={STEPS} train={TRAIN} val={VAL} -> "
          f"final_train_loss={ft:.4f} final_val_loss={fv:.4f} "
          f"lambda_clipped_frac={last['lambda_clipped_frac']:.4f}")

    # Numeric check against the committed reference (default config only).
    ref = SMOKE_REF.get(cfg_name) if AT_DEFAULTS else None
    if ref and ref.get("final_val_loss") is not None:
        assert abs(fv - ref["final_val_loss"]) <= SMOKE_TOL, (
            f"final_val_loss {fv:.4f} vs ref {ref['final_val_loss']:.4f} "
            f"exceeds PACE_SMOKE_TOL={SMOKE_TOL}")
        assert abs(ft - ref["final_train_loss"]) <= SMOKE_TOL, (
            f"final_train_loss {ft:.4f} vs ref {ref['final_train_loss']:.4f} "
            f"exceeds PACE_SMOKE_TOL={SMOKE_TOL}")


if __name__ == "__main__":
    for cfg_name, hf_model in MODELS:
        data_dir = _prepare_data(hf_model)
        print(f"prepared data for {cfg_name} at {data_dir}; run via pytest to train.")
