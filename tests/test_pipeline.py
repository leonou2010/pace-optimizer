"""End-to-end pipeline check, by import (no subprocess).

Runs the repo's own code step by step on the small model:
  1. `pace.data.prepare_sft`            -> tokenize a small slice of smol-smoltalk
  2. `pace_figures.reproduce.expand_figure` -> read a figure yaml, pick a PACE run
  3. build that run's config from `conf/` (the model/optimizer/schedule yamls)
  4. `pace.train.run_training`          -> train it (uses pace.optimizer / schedules / model)

This verifies the whole pipeline *runs* and trains to a sane loss; it is fast (a small
slice, a few steps) and is NOT a paper reproduction. Reproducing a figure number requires
the full dataset and a full run (see the configs in `src/pace_figures/conf/fig/`); the
committed reference runs live in `tests/fixtures/results/`. Needs network and a GPU.
"""
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pace.data import prepare_sft
from pace.train import run_training
from pace_figures.reproduce import expand_figure

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "src" / "pace" / "conf"
FIG = REPO / "src" / "pace_figures" / "conf" / "fig" / "fig3_135m_grid.yaml"
TRAIN = int(os.environ.get("PACE_TEST_TRAIN", "1024"))
STEPS = int(os.environ.get("PACE_TEST_STEPS", "5"))
CACHE = Path(os.environ.get("PACE_TEST_DATA_CACHE", Path.home() / ".cache" / "pace_test"))


def _compose(curve, data_path):
    """Build the training config from the conf/ yamls plus this run's overrides."""
    from hydra import compose, initialize_config_dir

    o = curve["optimizer"]
    overrides = [
        f"model={curve['model']}", f"optimizer={o['name']}",
        f"schedule={curve['schedule']}", f"seed={curve['seed']}",
        f"optimizer.lr={o['lr']}",
        f"optimizer.lambda_pullback={o.get('lambda_pullback', 0.0)}",
        f"dataset.path={data_path}",
        "training.gradient_accumulation_steps=1",      # small data -> a few real steps
        f"training.max_steps={STEPS}",
        f"training.eval_interval={max(1, STEPS // 2)}",
        "training.eval_samples=64",
    ]
    if o.get("ema_kappa") is not None:
        overrides.append(f"optimizer.ema_kappa={o['ema_kappa']}")
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_pipeline_trains_small_model(tmp_path):
    data = CACHE / f"pipeline_135m_{TRAIN}"
    if not (data / "train").exists():
        try:
            prepare_sft(str(data), tokenizer_name="HuggingFaceTB/SmolLM2-135M",
                        max_seq_len=256, max_train=TRAIN, max_val=128)
        except Exception as e:
            pytest.skip(f"could not fetch model/data from HuggingFace: {e}")

    curves = expand_figure(OmegaConf.load(FIG))
    pace = next(c for c in curves if c["optimizer"]["name"] == "pace")

    os.environ["PACE_RUN_DIR"] = str(tmp_path / "run")
    best_val = run_training(_compose(pace, data))
    print(f"  trained PACE lr={pace['optimizer']['lr']} c={pace['optimizer']['lambda_pullback']}"
          f" -> best_val_loss={best_val:.4f}")
    assert best_val == best_val and best_val < 100, f"bad best_val_loss {best_val}"
