"""Validation of the bundled PACE result fixtures.

The repository ships two real PACE fine-tuning runs (SmolLM2-135M and SmolLM2-1.7B)
under tests/fixtures/results/ so the result format and the reported figures can be
checked offline. These tests verify, for each fixture:

- the JSON has the expected {config, summary, eval_logs} structure;
- it is a PACE run (config.optimizer.lambda_pullback > 0);
- summary.final_val_loss matches the recorded value (within 1e-3) and
  summary.final_train_loss is present and positive;
- the train and validation loss trajectories in eval_logs both decrease from start
  to finish, and all recorded train losses are finite.

test_both_models_present confirms both fixture files exist. Offline; no GPU or network.
"""

import json
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures" / "results"

# (file, expected final_val_loss) — real PACE runs.
FIXTURES = [
    ("smollm2_135m_pace.json", 1.0390),   # small model
    ("smollm2_1.7b_pace.json", 0.6911),   # big model (Fig 1 headline)
]


@pytest.mark.parametrize("fname,expected_fv", FIXTURES)
def test_pace_result_fixture(fname, expected_fv):
    d = json.load(open(FIX / fname))

    # schema
    assert {"config", "summary", "eval_logs"} <= set(d), f"{fname}: bad schema"
    # it is genuinely a PACE run (pullback active)
    assert d["config"]["optimizer"]["lambda_pullback"] > 0, f"{fname}: not a PACE run"
    # final val loss present and matches the recorded value
    fv = d["summary"]["final_val_loss"]
    assert abs(fv - expected_fv) < 1e-3, f"{fname}: final_val_loss {fv} != {expected_fv}"
    # final train loss recorded too
    ft = d["summary"]["final_train_loss"]
    assert ft == ft and ft > 0, f"{fname}: bad final_train_loss {ft}"  # not NaN, positive
    # real trajectories (both train and val) that improve end-to-end
    logs = d["eval_logs"]
    vals = [e["val_loss"] for e in logs if e.get("val_loss") is not None]
    trains = [e["train_loss"] for e in logs if e.get("train_loss") is not None]
    assert len(vals) >= 2 and vals[-1] < vals[0], f"{fname}: val trajectory not decreasing"
    assert len(trains) >= 2 and trains[-1] < trains[0], f"{fname}: train trajectory not decreasing"
    assert all(t == t and t > 0 for t in trains), f"{fname}: non-finite train_loss in eval_logs"


def test_both_models_present():
    names = {p.name for p in FIX.glob("*.json")}
    assert {"smollm2_135m_pace.json", "smollm2_1.7b_pace.json"} <= names
