"""Checks on the bundled PACE result fixtures (one small + one big model).

These are real PACE finetuning runs (`tests/fixtures/results/`) shipped so anyone
who clones the repo can exercise the result schema / numbers offline, without a
GPU or any download. They are the actual runs behind Fig 1's PACE points.
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
    # a real validation trajectory that improves end-to-end
    logs = d["eval_logs"]
    vals = [e["val_loss"] for e in logs if e.get("val_loss") is not None]
    assert len(vals) >= 2 and vals[-1] < vals[0], f"{fname}: trajectory not decreasing"


def test_both_models_present():
    names = {p.name for p in FIX.glob("*.json")}
    assert {"smollm2_135m_pace.json", "smollm2_1.7b_pace.json"} <= names
