"""Offline checks for paper run specifications and packed data."""

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import prepare_data
from reproduce import command, expand_figure
from train import run_name


ROOT = Path(__file__).resolve().parents[1]


def test_figure_runs_have_distinct_outputs_and_paper_budgets():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        configs = {model: compose(config_name="config", overrides=[f"model={model}"])
                   for model in ("smollm2_135m", "smollm2_1.7b", "qwen3_1.7b", "gemma3_1b")}
    for path in (ROOT / "figures").glob("*.yaml"):
        fig = yaml.safe_load(path.read_text())
        runs = expand_figure(fig)
        names = []
        for run in runs:
            optimizer = yaml.safe_load((ROOT / "configs/optimizer" / f"{run['optimizer']['name']}.yaml").read_text())
            optimizer.update(run["optimizer"])
            cfg = OmegaConf.merge(configs[run["model"]], {
                "optimizer": optimizer, "schedule": {"name": run["schedule"]},
                "seed": run["seed"],
                "training": {**fig.get("training", {}), **run.get("training", {})},
            })
            names.append(run_name(cfg))
            assert cfg.training.batch_size * cfg.training.gradient_accumulation_steps == (
                128 if run["model"] == "smollm2_135m" else 256)
            if path.stem == "fig_appendix_train_curves" and run["model"] == "smollm2_135m":
                assert cfg.training.max_steps == 3557
                assert "training.max_steps=3557" in command(fig, run, [])
        assert len(names) == len(set(names)), path.name
        if path.stem == "fig1":
            assert len(runs) == 27
            assert {r["schedule"] for r in runs} == {"wsd"}
            assert {r["seed"] for r in runs} == {42, 1337, 2024}


def test_packing_preserves_tokens_and_masks_and_only_pads_the_tail(monkeypatch):
    ids = list(range(1, 13))
    labels = [-100, -100] + ids[2:]
    monkeypatch.setattr(prepare_data, "tokenize_conversation",
                        lambda *args: (ids, [1] * 12, labels))
    rows = [{"messages": [{"role": "assistant", "content": "example"}]}] * 2
    packed = prepare_data._build_split(rows, None, 16, 0, "test")
    assert len(packed) == 2
    assert packed[0]["input_ids"] + packed[1]["input_ids"][:8] == ids * 2
    assert packed[0]["labels"] + packed[1]["labels"][:8] == labels * 2
    assert packed[1]["attention_mask"] == [1] * 8 + [0] * 8
    assert packed[1]["labels"][8:] == [-100] * 8


def test_long_conversations_are_not_truncated_before_packing():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    ids, _, labels = prepare_data.tokenize_conversation(
        [{"role": "assistant", "content": "x" * 100}], Tokenizer(), 16)
    assert len(ids) > 100
    assert labels.count(ord("x")) == 100
